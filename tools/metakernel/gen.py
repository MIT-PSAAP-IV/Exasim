#!/usr/bin/env python3
"""Meta-kernel generator: schedules (schedules.py) -> one header per schedule.

A schedule is a list of kernels; a kernel is gemm("interp"|"integrate") or pt(...)/el(...) holding an ordered list
of pointwise stages. See DESIGN.md. Output: <out>/<name>/schedule.hpp, <out>/schedules.txt ("name scheduler" lines).
usage: gen.py schedules.py <out dir>
"""
import sys, os, importlib.util
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---- stage table: reads / writes value classes; comps per class --------------------------------------------------
COMPS = {"u": "MK_NC", "w": "MK_NCW", "s": "MK_NCU", "sg": "MK_NCU", "pr": "MK_NPR", "f": "MK_NF", "rg": "MK_NRG"}
STAGES = {
    "interp":    ((), ("u", "w")),
    "source":    (("u", "w"), ("s",)),
    "time":      (("u", "s"), ("sg",)),
    "flux":      (("u", "w"), ("f",)),
    "props":     (("u", "w"), ("pr",)),
    "fluxp":     (("u", "w", "pr"), ("f",)),
    "scale":     (("sg", "f"), ("rg",)),
    "integrate": (("rg",), ()),
}
BODY = {"source": "body_Source.inc", "flux": "body_Flux.inc", "props": "body_Props.inc", "fluxp": "body_FluxP.inc"}

class K:
    def __init__(self, kind, stages, E=0, waves=1, lb=1, hand=None, launder=False, phased=False, label=None, ksel=None, sf=False):
        """ksel=(i, n): this kernel computes only conserved components k in [i*NCU/n, (i+1)*NCU/n) of flux/scale -- the
        physics body is included whole and the compiler drops what feeds no selected output (fewer live registers)."""
        self.kind, self.stages, self.E, self.waves, self.lb = kind, list(stages), E, waves, lb
        self.hand, self.launder, self.phased, self.label, self.ksel = dict(hand or {}), launder, phased, label, ksel
        self.sf = sf      # fe(): source + flux from one combined body (body_SourceFlux.inc) sharing the temperature functions
def gemm(stage): return K("gemm", [stage])
def pt(*stages, **kw): return K("pt", stages, **kw)
def el(E, *stages, **kw): return K("el", stages, E=E, **kw)
def fe(E=2, **kw):
    """The whole element chain fused: MFMA interp -> LDS -> source/time/flux/scale -> LDS -> MFMA integrate.
    One wave per E elements; same MFMA chains (k groups of 4, ascending) as the GEMM kernels, same pointwise code."""
    return K("fe", ["interp", "source", "time", "flux", "scale", "integrate"], E=E, **kw)
class S:
    """backend: "hip" (plain __global__ kernels, the default) or "kokkos" (Kokkos TeamPolicy functors)."""
    def __init__(self, name, kernels, sched="default", note="", backend="hip", face=None, q=None, scatter=False, uhat=False, rqface=False, w=False, fsel=False, batch=False):
        """kernels=[] leaves the element residual to Exasim (e2e only); face=dict(...) generates the face residual (gen_face.py)."""
        assert "_K" not in name and name.replace("_", "").isalnum(), name
        assert backend in ("hip", "kokkos"), backend
        self.name, self.kernels, self.sched, self.note, self.backend, self.face, self.q = name, kernels, sched, note, backend, face, q
        self.scatter, self.uhat, self.rqface, self.w, self.fsel = scatter, uhat, rqface, w, fsel
        self.batch = batch      # also emit one-launch-per-range variants of the element and q kernels

def check_and_bind(s):
    """Per kernel: binding of every class it touches: 'reg' | 'lds' | 'glb' (+ 'ldsrg' for in-kernel integrate)."""
    produced_in = {}
    for j, k in enumerate(s.kernels):
        for st in k.stages:
            if st not in STAGES: sys.exit("%s: unknown stage %s" % (s.name, st))
            for c in STAGES[st][1]: produced_in[c] = j
    later_use = {}   # kernels that read c from OUTSIDE (not produced earlier in the same kernel)
    for j, k in enumerate(s.kernels):
        mine = set()
        for st in k.stages:
            for c in STAGES[st][0]:
                if c not in mine: later_use.setdefault(c, set()).add(j)
            mine |= set(STAGES[st][1])
    binds = []
    seen = set()
    for j, k in enumerate(s.kernels):
        b = {}
        if k.kind == "fe":
            if len(s.kernels) != 1: sys.exit("%s: fe() is the whole element chain" % s.name)
            binds.append({"u": "lds", "w": "lds", "s": "reg", "sg": "reg", "f": "reg", "rg": "ldsrg"}); continue
        if k.kind == "gemm":
            if k.stages[0] == "interp": seen |= {"u", "w"}
            elif k.stages[0] == "integrate":
                if "rg" not in seen: sys.exit("%s K%d: integrate before rg is produced" % (s.name, j))
            binds.append(b); continue
        if "integrate" in k.stages:
            if k.kind != "el": sys.exit("%s K%d: in-kernel integrate needs an el() map" % (s.name, j))
            if k.stages[-1] != "integrate": sys.exit("%s K%d: integrate must be the last stage" % (s.name, j))
        if k.kind == "el" and k.waves * 64 < k.E * 27: sys.exit("%s K%d: %d waves cannot hold %d elements" % (s.name, j, k.waves, k.E))
        local = set()
        for st in k.stages:
            for c in STAGES[st][0]:
                if c not in seen and c not in local: sys.exit("%s K%d: stage %s reads %s before it is produced" % (s.name, j, st, c))
                if c not in local: b[c] = "glb"
            for c in STAGES[st][1]:
                local.add(c)
                if c == "rg" and "integrate" in k.stages: b[c] = "ldsrg"
                elif any(u > j for u in later_use.get(c, ())) or (c == "rg"): b[c] = "glb"
                else:
                    h = k.hand.get(c, "glb" if k.phased else "reg")
                    if k.phased and h == "reg": sys.exit("%s K%d: phased kernels cannot hand %s off in registers" % (s.name, j, c))
                    b[c] = h
        seen |= local
        binds.append(b)
    return binds

def acc(c, bind, kern, k="k", m=None):
    """C++ lvalue for component k (and m for rg) of class c under a binding."""
    if c == "rg":
        idx = "mk_g + MK_NGE*(%s) + MK_NGE*4*(%s)" % (m, k)
        if bind == "ldsrg": return "mk_L_rg[%s + MK_NGE*4*MK_NCU*mk_el]" % idx
        return "b.g_rg[%s + (size_t)MK_NGE*4*MK_NCU*mk_e]" % idx
    if bind == "reg": return "mk_r_%s[%s]" % (c, k)
    if bind == "lds": return "mk_L_%s[(%s)*MK_TP + lane]" % (c, k)
    return "b.g_%s[(size_t)(%s)*b.ng + i]" % (c, k)

def stage_code(st, B, kern):
    R = lambda c, k="k", m=None: acc(c, B[c], kern, k, m)
    if st in BODY:
        rd, wr = STAGES[st]
        out = wr[0]
        lines = ["const dstype* param = b.par; const dstype* uinf = b.uinf; const dstype time = b.time; (void)param; (void)uinf; (void)time;"]
        for c, mac in (("u", "MK_RD_u"), ("w", "MK_RD_w"), ("pr", "MK_RD_pr")):
            if c in rd: lines.append("#define %s(k) %s" % (mac, R(c)))
        lines += ["#define MK_RD_x(k) b.x[(size_t)(k)*b.ng + i]", "#define MK_RD_o(k) b.o[(size_t)(k)*b.ng + i]",
                  ("#define MK_WR(k) %s" % R(out)) if not (kern.ksel and st in ("flux", "fluxp")) else
                  ("dstype mk_sink; (void)mk_sink;\n#define MK_WR(k) (*((((k) %% MK_NCU) >= MK_KLO && ((k) %% MK_NCU) < MK_KHI) ? &%s : &mk_sink))" % R(out)),
                  '#include "%s"' % BODY[st],
                  "#undef MK_RD_x", "#undef MK_RD_o", "#undef MK_WR"] + ["#undef %s" % m for c, m in (("u", "MK_RD_u"), ("w", "MK_RD_w"), ("pr", "MK_RD_pr")) if c in rd]
        return "\n".join(lines)
    if st == "interp":
        return ("#pragma unroll\nfor (int k = 0; k < MK_NC; k++) { dstype mk_s = 0;\n#pragma unroll\n"
                "  for (int p = 0; p < MK_NPE; p++) mk_s += b.gt[mk_g + MK_NGE*p] * b.udg[p + MK_NPE*k + (size_t)MK_NPE*MK_NC*mk_e];\n"
                "  %s = mk_s; }\n#pragma unroll\nfor (int k = 0; k < MK_NCW; k++) { dstype mk_s = 0;\n#pragma unroll\n"
                "  for (int p = 0; p < MK_NPE; p++) mk_s += b.gt[mk_g + MK_NGE*p] * b.wdg[p + MK_NPE*k + (size_t)MK_NPE*MK_NCW*mk_e];\n"
                "  %s = mk_s; }") % (R("u"), R("w"))
    if st == "time":
        return ("#pragma unroll\nfor (int k = 0; k < MK_NCU; k++) { const dstype mk_t = 1.0 * b.sdg[i + (size_t)b.ng*k] + (-b.dtf) * %s;\n"
                "  %s = 1.0 * mk_t + 1.0 * %s; }") % (R("u"), R("sg"), R("s"))
    if st == "scale":
        return ("#pragma unroll\nfor (int k = MK_KLO; k < MK_KHI; k++) {\n  %s = %s * b.jac[i];\n#pragma unroll\n"
                "  for (int m = 0; m < MK_ND; m++) { dstype mk_s = %s * b.Xx[i + (size_t)b.ng*(MK_ND*m)];\n#pragma unroll\n"
                "    for (int j = 1; j < MK_ND; j++) mk_s += %s * b.Xx[i + (size_t)b.ng*(j + MK_ND*m)];\n    %s = mk_s; } }") % (
                R("rg", "k", "0"), R("sg"), R("f", "k"), R("f", "k + MK_NCU*j"), R("rg", "k", "m + 1"))
    sys.exit("no code for stage " + st)

def kernel_body(s, j, k, B, hip):
    """Lines of the per-thread body; `lane`, `mk_blk` and the LDS base are provided by the wrapper."""
    tp = 64 * k.waves
    lds = [(c, "(size_t)%s*%d" % (COMPS[c], tp)) for c, v in B.items() if v == "lds"]
    if B.get("rg") == "ldsrg": lds.append(("rg", "(size_t)MK_NGE*4*MK_NCU*%d" % k.E))
    o = []
    if k.kind == "pt":
        o += ["    const size_t i = (size_t)mk_blk * MK_TP + lane; const bool mk_ok = i < (size_t)a.ng;",
              "    const int mk_g = (int)(i % MK_NGE); const size_t mk_e = i / MK_NGE; const int mk_el = 0; (void)mk_el; (void)mk_g; (void)mk_e;"]
    else:
        o += ["    const int mk_e0 = mk_blk * %d, mk_el = lane / MK_NGE, mk_g = lane %% MK_NGE; const size_t mk_e = (size_t)mk_e0 + mk_el;" % k.E,
              "    const bool mk_ok = lane < %d * MK_NGE && mk_e < (size_t)a.ne; const size_t i = mk_e * MK_NGE + mk_g; (void)mk_el;" % k.E]
    off = "0"
    for c, n in lds:
        o.append("    dstype* mk_L_%s = mk_shm + %s;" % (c, off)); off = "%s + %s" % (off, n)
    for c, v in B.items():
        if v == "reg": o.append("    dstype mk_r_%s[%s];" % (c, COMPS[c]))
    pw = [st for st in k.stages if st != "integrate"]
    binder = "const MkIn b = mk_launder(a);" if (k.launder or k.phased) else "const MkIn& b = a;"
    if k.phased:
        o += ["#pragma nounroll", "    for (int mk_ph = 0; mk_ph < %d; mk_ph++) {" % len(pw), "      %s" % binder, "      if (mk_ok) {"]
        for q, st in enumerate(pw):
            o += ["      %sif (mk_ph == %d) { // %s" % ("else " if q else "", q, st), stage_code(st, B, k), "      }"]
        o += ["      }", "      MK_CFENCE();", "    }"]
    else:
        for st in pw:
            o += ["    if (mk_ok) { // %s" % st, "    %s" % binder, stage_code(st, B, k), "    }"]
            if k.launder: o.append("    MK_CFENCE();")
    if "integrate" in k.stages:
        o += ["    MK_BARRIER();", "    { const MkIn& b = a;",
              "    for (int idx = lane; idx < %d * MK_NPE * MK_NCU; idx += MK_TP) {" % k.E,
              "      const int el = idx / (MK_NPE*MK_NCU), r = idx % (MK_NPE*MK_NCU), p = r % MK_NPE, k = r / MK_NPE; const size_t e = (size_t)mk_e0 + el;",
              "      if (e >= (size_t)b.ne) continue;",
              "      dstype mk_s = 0; for (int gd = 0; gd < MK_NGE*4; gd++) mk_s += b.gw[p + MK_NPE*gd] * mk_L_rg[gd + MK_NGE*4*k + MK_NGE*4*MK_NCU*el];",
              "      b.R[p + MK_NPE*(k + (size_t)MK_NCU*e)] = mk_s; } }"]
    shm = "(%s) * sizeof(dstype)" % off
    return o, shm

def kdefs(k):
    i, n = k.ksel or (0, 1)
    return ["#define MK_KLO ((%d * MK_NCU) / %d)" % (i, n), "#define MK_KHI ((%d * MK_NCU) / %d)" % (i + 1, n)]

def emit_kernel(s, j, k, B):
    nm = "MK_%s_K%d" % (s.name, j)
    tp = 64 * k.waves
    league = "(a.ng + %d - 1) / %d" % (tp, tp) if k.kind == "pt" else "(a.ne + %d - 1) / %d" % (k.E, k.E)
    body, shm = kernel_body(s, j, k, B, s.backend == "hip")
    if s.backend == "hip":
        # extern "C" keeps the symbol = the kernel name, so compiler remarks map straight back to (schedule, kernel)
        o = ["static constexpr size_t %s_SHM(const MkIn& a) { (void)a; return %s; }" % (nm, shm),
             'extern "C" __global__ void __attribute__((amdgpu_flat_work_group_size(1, %d), amdgpu_waves_per_eu(%d)))' % (tp, k.lb),
             "%s(const MkIn a) {" % nm, "    constexpr int MK_TP = %d;" % tp,
             "    const int lane = threadIdx.x, mk_blk = blockIdx.x; extern __shared__ dstype mk_shm[]; (void)mk_shm;",
             "#define MK_BARRIER() __syncthreads()"] + kdefs(k) + body + ["#undef MK_BARRIER", "#undef MK_KLO", "#undef MK_KHI", "}"]
        launch = ("hipLaunchKernelGGL(%s, dim3(%s), dim3(%d), %s_SHM(a), mk_stream(), a); MK_HIPCHECK(\"%s\");" % (nm, league, tp, nm, nm))
        return "\n".join(o), launch
    o = ["struct %s {" % nm, "  static constexpr int MK_TP = %d;" % tp,
         "  using P = Kokkos::TeamPolicy<Kokkos::LaunchBounds<%d, %d>>; MkIn a;" % (tp, k.lb),
         "  KOKKOS_INLINE_FUNCTION static size_t shmem(const MkIn& a) { (void)a; return %s; }" % shm,
         "  KOKKOS_INLINE_FUNCTION void operator()(const typename P::member_type& t) const {",
         "    const int lane = t.team_rank(), mk_blk = t.league_rank();",
         "    dstype* mk_shm = (dstype*)t.team_scratch(0).get_shmem(shmem(a)); (void)mk_shm;",
         "#define MK_BARRIER() t.team_barrier()"] + kdefs(k) + body + ["#undef MK_BARRIER", "#undef MK_KLO", "#undef MK_KHI", "  }", "};"]
    launch = 'Kokkos::parallel_for("%s", %s::P(%s, %d).set_scratch_size(0, Kokkos::PerTeam(%s::shmem(a))), %s{a});' % (nm, nm, league, tp, nm, nm)
    return "\n".join(o), launch

def label(k):
    if k.label: return k.label
    if k.kind == "gemm": return "gemm:" + k.stages[0]
    if k.kind == "fe": return "fused-elem E=%d lb=%d%s" % (k.E, k.lb, " srcflux" if k.sf else "")
    t = ("pt" if k.kind == "pt" else "el%d" % k.E) + "w%d/lb%d" % (k.waves, k.lb)
    f = [x for x, on in (("launder", k.launder), ("phased", k.phased)) if on] + (["k%d/%d" % k.ksel] if k.ksel else [])
    return "%s[%s]%s" % (t, "+".join(k.stages), ("{" + ",".join(f) + "}") if f else "")

def producer_prologue(E, unroll=4, mfma=False):
    """Face values from the producer: stage the element's nodal u|w (npe x (nc + ncw), the size of the u|w Gauss buffer for E = 1) in LDS,
    then write this element's side of each of its interior faces at the face Gauss points -- the face interpolation
    kernel's sums (ascending face node q, from 0), so bitwise identical -- straight into the face stage's buffer."""
    if E != 1: sys.exit("producer face values need fe(E=1)")
    return [
        "#ifndef MK_FACE_PRODUCER",
        "#define MK_FACE_PRODUCER 1",
        "#endif",
        "    static_assert(MK_NPE * (MK_NC + MK_NCW) <= (MK_LU > MK_LR ? MK_LU : MK_LR), \"nodal block must fit the LDS buffer\");",
        "    if (nel > 0) { const int Eg = b.pe1 + mk_e0;",
        "        for (int t = lane; t < MK_NPE * (MK_NC + MK_NCW); t += 64)",
        "            mk_shm[t] = t < MK_NPE * MK_NC ? b.udg[(size_t)MK_NPE * MK_NC * mk_e0 + t] : b.wdg[(size_t)MK_NPE * MK_NCW * mk_e0 + (t - MK_NPE * MK_NC)];",
        "        __syncthreads();",
        ] + ([
        "        // face values on the matrix cores: D(col, j) = sum_p N(p, col) B(p, j), j = (slot, g) over 54 rows,",
        "        // B(p, j) = gt_face(g, q) where face node q of slot sits at element node p (from pinv), else 0",
        "        __shared__ int mk_pi[6 * MK_NPE]; __shared__ dstype* mk_pd[6]; __shared__ int mk_pg[6]; __shared__ dstype mk_gtf[MK_NGF * MK_NPF + 1];",
        "        for (int t = lane; t < 6 * MK_NPE; t += 64) mk_pi[t] = b.pinv[(size_t)6 * MK_NPE * Eg + t];",
        "        for (int t = lane; t < MK_NGF * MK_NPF; t += 64) mk_gtf[t] = b.fgt[t];",
        "        if (lane == 0) mk_gtf[MK_NGF * MK_NPF] = 0.0;",
        "        if (lane < 6) { mk_pd[lane] = b.pbase[6 * Eg + lane]; mk_pg[lane] = b.pnga[6 * Eg + lane]; }",
        "        __syncthreads();",
        "#pragma unroll",
        "        for (int n0 = 0; n0 < MK_NC + MK_NCW; n0 += 16) { const int col = n0 + lr; const bool cin = col < MK_NC + MK_NCW;",
        "            const int cb = col < MK_NC ? MK_NPE*col : MK_NPE*MK_NC + MK_NPE*(col - MK_NC);",
        "            dstype fa[7];",
        "#pragma unroll",
        "            for (int st = 0; st < 7; st++) { const int p = 4*st + lk; fa[st] = (cin && p < MK_NPE) ? mk_shm[p + cb] : 0.0; }",
        "#pragma unroll",
        "            for (int m0 = 0; m0 < 6 * MK_NGF; m0 += 16) { const int j = m0 + lr, sl = j / MK_NGF, g = j % MK_NGF; const bool jin = j < 6 * MK_NGF;",
        "                mk_fd4 acc = (mk_fd4){0.0, 0.0, 0.0, 0.0};",
        "#pragma unroll",
        "                for (int st = 0; st < 7; st++) { const int p = 4*st + lk; const int q = (jin && p < MK_NPE) ? mk_pi[MK_NPE*sl + p] : -1;",
        "                    acc = __builtin_amdgcn_mfma_f64_16x16x4f64(fa[st], mk_gtf[q >= 0 ? g + MK_NGF*q : MK_NGF*MK_NPF], acc, 0, 0, 0); }",
        "                dstype* dst = jin ? mk_pd[sl] : nullptr;",
        "                if (dst) { const size_t ng_ = (size_t)mk_pg[sl];",
        "#pragma unroll",
        "                    for (int r = 0; r < 4; r++) { const int cc = n0 + 4*r + lk; if (cc < MK_NC + MK_NCW) dst[g + ng_ * cc] = acc[r]; } } } }",
        ] if mfma else [
        "        if (lane < 6 * MK_NGF) {   // lane = (face slot, Gauss point): its 9 face nodes stay in registers",
        "            const int sl = lane / MK_NGF, g = lane % MK_NGF; dstype* dst = b.pbase[6 * Eg + sl];",
        "            if (dst) { const size_t ng_ = (size_t)b.pnga[6 * Eg + sl]; const int* nd = b.pnode + (size_t)MK_NPF * (6 * Eg + sl);",
        "                int pp[MK_NPF]; dstype gq[MK_NPF];",
        "#pragma unroll",
        "                for (int q = 0; q < MK_NPF; q++) { pp[q] = nd[q]; gq[q] = b.fgt[g + MK_NGF*q]; }",
        "#pragma unroll %d" % unroll,
        "                for (int col = 0; col < MK_NC + MK_NCW; col++) { const int cb = col < MK_NC ? MK_NPE*col : MK_NPE*MK_NC + MK_NPE*(col - MK_NC);",
        "                    dstype mk_s = 0;",
        "#pragma unroll",
        "                    for (int q = 0; q < MK_NPF; q++) mk_s += gq[q] * mk_shm[pp[q] + cb];",
        "                    dst[g + ng_ * col] = mk_s; } } }",
        ]) + [
        "        __syncthreads(); }"]


def emit_fused_elem(s, j, k, B):
    """fe(E): one 64-lane wave per E elements (E*27 <= 64). LDS: u|w at Gauss points [c*TP + el*27 + g] (TP = E*27), then
    (after a barrier) rg in the integrate GEMM's B layout [gd + 108*(kc + ncu*el)] over the same bytes."""
    E = k.E; TP = E * 27; nm = "MK_%s_K%d" % (s.name, j)
    if TP > 64: sys.exit("%s: fe(E=%d) needs E*27 <= 64" % (s.name, E))
    pw = ["source", "time", "flux"]
    o = ["typedef double mk_fd4 __attribute__((ext_vector_type(4)));",
         "static constexpr size_t %s_SHM(const MkIn& a) { (void)a; return 0; }" % nm,
         'extern "C" __global__ void __attribute__((amdgpu_flat_work_group_size(1, 64), amdgpu_waves_per_eu(%d)))' % k.lb,
         "%s(const MkIn a) {" % nm,
         "    constexpr int MK_TP = %d, MK_FE = %d;   // LDS stride (Gauss points per wave), elements per wave" % (TP, E),
         "    constexpr int MK_LU = (MK_NC + MK_NCW) * MK_TP, MK_LR = MK_NGE * 4 * MK_NCU * MK_FE;",
         "    __shared__ dstype mk_shm[MK_LU > MK_LR ? MK_LU : MK_LR];",
         "    const int lane = threadIdx.x, mk_blk = blockIdx.x, lr = lane & 15, lk = lane >> 4;",
         "    const int mk_e0 = mk_blk * MK_FE, nel = min(MK_FE, a.ne - mk_e0);",
         "    dstype* mk_L_u = mk_shm; dstype* mk_L_w = mk_shm + MK_NC * MK_TP; dstype* mk_L_rg = mk_shm;",
         "    const MkIn& b = a;"] + kdefs(k) + (producer_prologue(E, int((s.face or {}).get("punroll", 4)), bool((s.face or {}).get("pmfma", False))) if (s.face or {}).get("producer") else []) + [
         "    // ---- interp on the matrix cores: G(g, col) = sum_p gt[g + nge p] un(p, col); col = c + nc*el (u), el (w)",
         "    //      a-fragment = un column (lanes lr = col, lk = k within the group), b = gt row: GEMM v2's chain, bitwise",
    ] + (([
         "    {   // producer: the nodal block is already in LDS -- take the MFMA operands from there (same values),",
         "        //      all of them into registers first, because the Gauss values below overwrite the same LDS",
         "        dstype fra[(MK_NC + 15) / 16][7], frw[7];",
         "#pragma unroll",
         "        for (int ch = 0; ch < (MK_NC + 15) / 16; ch++) { const int col = 16*ch + lr; const bool cin = nel > 0 && col < MK_NC;",
         "#pragma unroll",
         "            for (int st = 0; st < 7; st++) { const int p = 4*st + lk; fra[ch][st] = (cin && p < MK_NPE) ? mk_shm[p + MK_NPE*col] : 0.0; } }",
         "#pragma unroll",
         "        for (int st = 0; st < 7; st++) { const int p = 4*st + lk; frw[st] = (nel > 0 && lr < MK_NCW && p < MK_NPE) ? mk_shm[MK_NPE*MK_NC + p + MK_NPE*lr] : 0.0; }",
         "        __syncthreads();",
         "#pragma unroll",
         "        for (int ch = 0; ch < (MK_NC + 15) / 16; ch++) { const int n0 = 16*ch, ncol = MK_NC * nel;",
         "#pragma unroll",
         "            for (int m0 = 0; m0 < 32; m0 += 16) { mk_fd4 acc = (mk_fd4){0.0, 0.0, 0.0, 0.0}; const int g = m0 + lr;",
         "#pragma unroll",
         "                for (int st = 0; st < 7; st++) { const int p = 4*st + lk;",
         "                    acc = __builtin_amdgcn_mfma_f64_16x16x4f64(fra[ch][st], (g < MK_NGE && p < MK_NPE) ? b.gt[g + MK_NGE*p] : 0.0, acc, 0, 0, 0); }",
         "                if (g < MK_NGE) {",
         "#pragma unroll",
         "                    for (int r = 0; r < 4; r++) { const int cc = n0 + 4*r + lk; if (cc < ncol) mk_L_u[(cc % MK_NC)*MK_TP + (cc / MK_NC)*MK_NGE + g] = acc[r]; } } } }",
         "        if (MK_NCW > 0) { const int nw = MK_NCW * nel;",
         "#pragma unroll",
         "            for (int m0 = 0; m0 < 32; m0 += 16) { mk_fd4 acc = (mk_fd4){0.0, 0.0, 0.0, 0.0}; const int g = m0 + lr;",
         "#pragma unroll",
         "                for (int st = 0; st < 7; st++) { const int p = 4*st + lk;",
         "                    acc = __builtin_amdgcn_mfma_f64_16x16x4f64(frw[st], (g < MK_NGE && p < MK_NPE) ? b.gt[g + MK_NGE*p] : 0.0, acc, 0, 0, 0); }",
         "                if (g < MK_NGE) {",
         "#pragma unroll",
         "                    for (int r = 0; r < 4; r++) { const int cc = 4*r + lk; if (cc < nw) mk_L_w[(cc % MK_NCW)*MK_TP + (cc / MK_NCW)*MK_NGE + g] = acc[r]; } } } }",
         "    }",
         ]) if ((s.face or {}).get("producer") and (s.face or {}).get("plds", True)) else [
         "    {   const int ncol = MK_NC * nel;",
         "        for (int n0 = 0; n0 < ncol; n0 += 16) { const int col = n0 + lr; const bool cin = col < ncol;",
         "            const int c = cin ? col % MK_NC : 0, el = cin ? col / MK_NC : 0; const size_t e = (size_t)mk_e0 + el;",
         "            dstype fr[7];",
         "#pragma unroll",
         "            for (int st = 0; st < 7; st++) { const int p = 4*st + lk;",
         "                fr[st] = (cin && p < MK_NPE) ? b.udg_abs[b.eind[p + MK_NPE*(e + (size_t)b.ne*c)]] : 0.0; }",
         "#pragma unroll",
         "            for (int m0 = 0; m0 < 32; m0 += 16) { mk_fd4 acc = (mk_fd4){0.0, 0.0, 0.0, 0.0}; const int g = m0 + lr;",
         "#pragma unroll",
         "                for (int st = 0; st < 7; st++) { const int p = 4*st + lk;",
         "                    acc = __builtin_amdgcn_mfma_f64_16x16x4f64(fr[st], (g < MK_NGE && p < MK_NPE) ? b.gt[g + MK_NGE*p] : 0.0, acc, 0, 0, 0); }",
         "                if (g < MK_NGE) {",
         "#pragma unroll",
         "                    for (int r = 0; r < 4; r++) { const int cc = n0 + 4*r + lk; if (cc < ncol) mk_L_u[(cc % MK_NC)*MK_TP + (cc / MK_NC)*MK_NGE + g] = acc[r]; } } } }",
         "        if (MK_NCW > 0) { const int nw = MK_NCW * nel;",
         "        for (int n0 = 0; n0 < nw; n0 += 16) { const int col = n0 + lr; const bool cin = col < nw;",
         "            const int c = cin ? col % MK_NCW : 0, el = cin ? col / MK_NCW : 0;",
         "            dstype fr[7];",
         "#pragma unroll",
         "            for (int st = 0; st < 7; st++) { const int p = 4*st + lk;",
         "                fr[st] = (cin && p < MK_NPE) ? b.wdg[p + MK_NPE*c + (size_t)MK_NPE*MK_NCW*(mk_e0 + el)] : 0.0; }",
         "#pragma unroll",
         "            for (int m0 = 0; m0 < 32; m0 += 16) { mk_fd4 acc = (mk_fd4){0.0, 0.0, 0.0, 0.0}; const int g = m0 + lr;",
         "#pragma unroll",
         "                for (int st = 0; st < 7; st++) { const int p = 4*st + lk;",
         "                    acc = __builtin_amdgcn_mfma_f64_16x16x4f64(fr[st], (g < MK_NGE && p < MK_NPE) ? b.gt[g + MK_NGE*p] : 0.0, acc, 0, 0, 0); }",
         "                if (g < MK_NGE) {",
         "#pragma unroll",
         "                    for (int r = 0; r < 4; r++) { const int cc = n0 + 4*r + lk; if (cc < nw) mk_L_w[(cc % MK_NCW)*MK_TP + (cc / MK_NCW)*MK_NGE + g] = acc[r]; } } } } }",
         "    }",
    ]) + [
         "    __syncthreads();",
         "    // ---- pointwise: one lane per Gauss point",
         "    const int mk_el = lane / MK_NGE, mk_g = lane % MK_NGE; const size_t mk_e = (size_t)mk_e0 + mk_el;",
         "    const bool mk_ok = lane < MK_TP && mk_el < nel; const size_t i = mk_e * MK_NGE + mk_g;"]
    for c, v in B.items():
        if v == "reg": o.append("    dstype mk_r_%s[%s];" % (c, COMPS[c]))
    if k.sf:
        sfc = stage_code("flux", B, k).replace('#include "body_Flux.inc"',
              '#define MK_WS(k) %s\n#include "body_SourceFlux.inc"\n#undef MK_WS' % acc("s", B["s"], k, "k"))
        o += ["    if (mk_ok) { // source + flux (one body)", sfc, "    }", "    if (mk_ok) { // time", stage_code("time", B, k), "    }"]
    else:
        for st in pw:
            o += ["    if (mk_ok) { // %s" % st, stage_code(st, B, k), "    }"]
    o += ["    __syncthreads();   // u/w are dead: rg reuses their LDS",
          "    if (mk_ok) { // scale", stage_code("scale", B, k), "    }",
          "    __syncthreads();",
          "    // ---- integrate on the matrix cores: R(p, n) = sum_gd gw[p + npe gd] rg(gd, n), n = kc + ncu*el, K = nge*(nd+1) = 108",
          "    {   const int ncol = MK_NCU * nel;",
          "        for (int n0 = 0; n0 < ncol; n0 += 16) { const int n = n0 + lr; const bool nin = n < ncol;",
          "            dstype fr[27];",
          "#pragma unroll",
          "            for (int st = 0; st < 27; st++) { const int gd = 4*st + lk; fr[st] = nin ? mk_L_rg[gd + MK_NGE*4*n] : 0.0; }",
          "#pragma unroll",
          "            for (int m0 = 0; m0 < 32; m0 += 16) { mk_fd4 acc = (mk_fd4){0.0, 0.0, 0.0, 0.0}; const int p = m0 + lr;",
          "#pragma unroll",
          "                for (int st = 0; st < 27; st++) { const int gd = 4*st + lk;",
          "                    acc = __builtin_amdgcn_mfma_f64_16x16x4f64(fr[st], p < MK_NPE ? b.gw[p + MK_NPE*gd] : 0.0, acc, 0, 0, 0); }",
          "                if (p < MK_NPE) {",
          "#pragma unroll",
          "                    for (int r = 0; r < 4; r++) { const int nn = n0 + 4*r + lk; if (nn < ncol) b.R[p + MK_NPE*(nn + (size_t)MK_NCU*mk_e0)] = acc[r]; } } } }",
          "    }", "#undef MK_KLO", "#undef MK_KHI",
          "}"]
    launch = 'hipLaunchKernelGGL(%s, dim3((a.ne + %d - 1) / %d), dim3(64), 0, mk_stream(), a); MK_HIPCHECK("%s");' % (nm, E, E, nm)
    return "\n".join(o), launch

def emit(s, out):
    B = check_and_bind(s)
    code, launches = [], []
    for j, k in enumerate(s.kernels):
        if k.kind == "fe":
            c, l = emit_fused_elem(s, j, k, B[j]); code.append(c); launches.append(l); continue
        if k.kind == "gemm":
            launches.append("mk_gemm_%s(a);" % k.stages[0]); continue
        c, l = emit_kernel(s, j, k, B[j]); code.append(c); launches.append(l)
    h = ["// generated by gen.py -- schedule %s  (%s)" % (s.name, s.note), "#pragma once",
         '#if __has_include("body_Flux_prelude.inc")', '#include "body_Flux_prelude.inc"   // file-scope part of a replacement flux body', "#endif",
         'static const char* MK_SCHED = "%s";' % s.name, 'static const char* MK_BACKEND = "%s";' % s.backend, "static const int MK_NK = %d;" % len(s.kernels),
         "static const char* MK_KLABEL[] = {%s};" % (", ".join('"%s"' % label(k) for k in s.kernels) or '""')] + code
    h += ["static void mk_run_kernel(const MkIn& a, int j) {", "  switch (j) {"]
    h += ["  case %d: { %s } break;" % (j, l) for j, l in enumerate(launches)] + ["  }", "}",
          "static void mk_run(const MkIn& a) { for (int j = 0; j < MK_NK; j++) mk_run_kernel(a, j); }"]
    if s.scatter: h.append("#define MK_GEN_SCATTER 1")
    if s.fsel: h.append("#define MK_FSEL 1")
    if s.uhat or s.rqface or s.w:
        import gen_rest
        h.append(gen_rest.emit_rest(s.name, uhat=s.uhat, rqface=s.rqface, w=s.w))
    if s.q is not None:
        import gen_q
        h.append(gen_q.emit_q(s.name, s.q))
    if s.face is not None:
        import gen_face
        h.append(gen_face.emit_face(s.name, s.face))
    if getattr(s, "batch", False):
        src = "\n".join(h)
        extra = ["// ---- batched (one launch per block range) variants ----", "#define MK_BATCHED 1"]
        if len(s.kernels) == 1 and s.kernels[0].kind == "fe":
            kn = "MK_%s_K0" % s.name; E = s.kernels[0].E
            extra += [batchify(src, kn, "MkIn", ("time", "dtf")),
                      "#define MK_ELEM_ALL 1",
                      "static int mk_elem_wg(const MkIn& a) { return (a.ne + %d - 1) / %d; }" % (E, E),
                      "static void mk_elem_all(const MkIn* blocks, const int* wgoff, int nb, int nwg, dstype time, dstype dtf) {",
                      "    hipLaunchKernelGGL(%s_all, dim3(nwg), dim3(64), 0, mk_stream(), blocks, wgoff, nb, time, dtf); MK_HIPCHECK(\"elem_all\"); }" % kn]
        if s.face is not None and s.face.get("mode") == "split" and s.face.get("interp") in ("staged", "conn") and not s.face.get("fi"):
            conn = s.face.get("interp") == "conn"
            ki, k1, k2 = "MKF_%s_%s_int" % (s.name, "interpc" if conn else "interps"), "MKF_%s_flux1" % s.name, "MKF_%s_flux2" % s.name
            shm = "std::max((size_t)(64 / MK_NGF) * MK_NCU * MK_NGF, (size_t)0) * sizeof(dstype)"
            extra += [batchify(src, ki, "MkFaceIn"), batchify(src, k1, "MkFaceIn"), batchify(src, k2, "MkFaceIn"),
                      "#define MK_FACE_ALL 1",
                      "static int mk_fi_wg(const MkFaceIn& a) { return %s; }" % (("(a.nf + %d) / %d" % (int(s.face.get("sfb", 4)) - 1, int(s.face.get("sfb", 4)))) if conn else "(a.nf + 7) / 8"),
                      "static int mk_f1_wg(const MkFaceIn& a) { return (MK_NGF * a.nf + 63) / 64; }",
                      "static int mk_f2_wg(const MkFaceIn& a) { return (a.nf + (64 / MK_NGF) - 1) / (64 / MK_NGF); }",
                      "static void mk_face_interior_all(const MkFaceIn* bi, const int* oi, const int* o1, const int* o2, int nb, int ni, int n1, int n2, dstype time) {",
                      "    const int ncol = 2*MK_NC + 2*MK_NCW;",
                      ("    (void)oi; (void)ni; (void)ncol;   // interior face values written by the element kernel (producer)") if s.face.get("producer") else
                      ("    hipLaunchKernelGGL(%s_all, dim3(ni), dim3(%d), 0, mk_stream(), bi, oi, nb, time); (void)ncol;" % (ki, int(s.face.get("tpb", 256)))) if conn else
                      ("    hipLaunchKernelGGL(%s_all, dim3(ni, (ncol + 31) / 32), dim3(256), 0, mk_stream(), bi, oi, nb, time, ncol);" % ki),
                      "    hipLaunchKernelGGL(%s_all, dim3(n1), dim3(64), 0, mk_stream(), bi, o1, nb, time);" % k1,
                      "    hipLaunchKernelGGL(%s_all, dim3(n2), dim3(64), (size_t)(64 / MK_NGF) * MK_NCU * MK_NGF * sizeof(dstype), mk_stream(), bi, o2, nb, time);" % k2,
                      "    MK_HIPCHECK(\"face_all\"); }"]
            import re as _re
            bou = {int(m): "MKF_%s_bou%s" % (s.name, m) for m in _re.findall(r'MKF_%s_bou(\d+)\(' % s.name, src)}
            kib = "MKF_%s_interp_bou" % s.name
            if bou and (kib + "(") in src:
                extra += [batchify(src, kib, "MkFaceIn"), batch_switch(src, bou, "MkFaceIn", ("time",), "MKF_%s_bou_all" % s.name),
                          "#define MK_FACE_BOU_ALL 1",
                          "static int mk_fib_wg(const MkFaceIn& a) { return (int)(((size_t)MK_NGF * a.nf * (MK_NCU + MK_NC + MK_NCW) + 255) / 256); }",
                          "static int mk_fb_wg(const MkFaceIn& a) { return (a.nf + (64 / MK_NGF) - 1) / (64 / MK_NGF); }",
                          "static void mk_face_bou_all(const MkFaceIn* bb, const int* oi, const int* ob, int nb, int ni, int nbw, dstype time) {",
                          "    hipLaunchKernelGGL(%s_all, dim3(ni), dim3(256), 0, mk_stream(), bb, oi, nb, time, MK_NCU + MK_NC + MK_NCW);" % kib,
                          "    hipLaunchKernelGGL(MKF_%s_bou_all, dim3(nbw), dim3(64), (size_t)(64 / MK_NGF) * MK_NCU * MK_NGF * sizeof(dstype), mk_stream(), bb, ob, nb, time);" % s.name,
                          "    MK_HIPCHECK(\"face_bou_all\"); }"]
        if s.uhat:
            import re as _re
            ub = {int(m): "MKU_%s_bou%s" % (s.name, m) for m in _re.findall(r'MKU_%s_bou(\d+)\(' % s.name, src)}
            if ub:
                extra += [batch_switch(src, ub, "MkUhIn", (("int", "fsel"), "time"), "MKU_%s_bou_all" % s.name),
                          "#define MK_UHAT_BOU_ALL 1",
                          "static int mk_ub_wg(const MkUhIn& a) { return (MK_NPF * a.nf + 63) / 64; }",
                          "static void mk_uhat_bou_all(const MkUhIn* bb, const int* wo, int nb, int nwg, int fsel, dstype time) {",
                          "    hipLaunchKernelGGL(MKU_%s_bou_all, dim3(nwg), dim3(64), 0, mk_stream(), bb, wo, nb, fsel, time); MK_HIPCHECK(\"uhat_bou_all\"); }" % s.name]
        if s.q is not None:
            kn = "MKQ_%s_elem" % s.name
            extra += [batchify(src, kn, "MkQIn", ()).replace(", )", ")"),
                      ] + ([batchify(src, kn + "f", "MkQIn", ()).replace(", )", ")"),
                      "static void mk_qf_all(const MkQIn* blocks, const int* wgoff, int nb, int nwg) {",
                      "    hipLaunchKernelGGL(%sf_all, dim3(nwg), dim3(64), 0, mk_stream(), blocks, wgoff, nb); MK_HIPCHECK(\"qf_all\"); }" % kn]
                      if (kn + "f(") in src else []) + [
                      "#define MK_Q_ALL 1",
                      "static int mk_q_wg(const MkQIn& a) { return a.neb; }",
                      "static void mk_q_all(const MkQIn* blocks, const int* wgoff, int nb, int nwg) {",
                      "    hipLaunchKernelGGL(%s_all, dim3(nwg), dim3(64), 0, mk_stream(), blocks, wgoff, nb); MK_HIPCHECK(\"q_all\"); }" % kn]
        h += extra
    d = os.path.join(out, s.name); os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "schedule.hpp"), "w").write("\n".join(h) + "\n")
    for j, k in enumerate(s.kernels): print("  %-22s K%d %-60s %s" % (s.name, j, label(k), B[j]))

# ---- batched launches: one launch over all blocks of a range ------------------------------------------------------------
def _kernel_parts(src, kname):
    i = src.index(kname + "(")
    head = src.rfind('extern "C"', 0, i)
    j = src.index("{", i)
    depth, k = 0, j
    while True:
        if src[k] == "{": depth += 1
        elif src[k] == "}":
            depth -= 1
            if depth == 0: break
        k += 1
    return src[head:j], src[j + 1:k]


def _fields(fields):
    return [(f, "dstype") if isinstance(f, str) else (f[1], f[0]) for f in fields]


def batch_switch(src, knames, T, fields, outname):
    """One launch over blocks whose kernel depends on the block's boundary type: knames = {ib: kernel}. The kernels must
    share a signature (const T a, ...) and block size; each workgroup finds its block and runs that block's kernel body."""
    F = _fields(fields)
    sig0, _ = _kernel_parts(src, next(iter(knames.values())))
    first = next(iter(knames.values()))
    new_sig = sig0.replace(first + "(", outname + "(").replace("const %s a" % T,
        "const %s* mk_blocks, const int* mk_wgoff, const int mk_nb%s" % (T, "".join(", const %s mk_%s" % (t, f) for f, t in F)), 1)
    pro = ("    int mk_j = 0; while (mk_j + 1 < mk_nb && (int)blockIdx.x >= mk_wgoff[mk_j + 1]) mk_j++;\n"
           "    %s mk_ab = mk_blocks[mk_j]; %s const %s a = mk_ab; const int mk_bid = (int)blockIdx.x - mk_wgoff[mk_j];\n"
           % (T, " ".join("mk_ab.%s = mk_%s;" % (f, f) for f, _ in F), T))
    cases = []
    for ib, kn in sorted(knames.items()):
        _, body = _kernel_parts(src, kn)
        cases.append("    case %d: {\n%s\n    } break;" % (ib, body.replace("blockIdx.x", "mk_bid")))
    return new_sig + "{\n" + pro + "    switch (a.ib) {\n" + "\n".join(cases) + "\n    default: break; }\n}\n"


def batchify(src, kname, T, fields=("time",)):
    """Return a copy of kernel `kname` (in source text `src`) that covers several blocks in one launch: it takes a device
    table of the blocks' argument structs and each block's first workgroup, finds its block, overrides the per-call
    fields (`fields`, e.g. time, dtf) from kernel arguments, and uses the block-local workgroup index for blockIdx.x."""
    i = src.index(kname + "(")
    head = src.rfind('extern "C"', 0, i)
    j = src.index("{", i)
    depth, k = 0, j
    while True:
        if src[k] == "{": depth += 1
        elif src[k] == "}":
            depth -= 1
            if depth == 0: break
        k += 1
    sig, body = src[head:j], src[j + 1:k]
    F = _fields(fields)
    new_sig = sig.replace(kname + "(", kname + "_all(").replace("const %s a" % T,
        "const %s* mk_blocks, const int* mk_wgoff, const int mk_nb, %s" % (T, ", ".join("const %s mk_%s" % (t, f) for f, t in F)), 1)
    pro = ("    int mk_j = 0; while (mk_j + 1 < mk_nb && (int)blockIdx.x >= mk_wgoff[mk_j + 1]) mk_j++;\n"
           "    %s mk_ab = mk_blocks[mk_j]; %s const %s a = mk_ab; const int mk_bid = (int)blockIdx.x - mk_wgoff[mk_j];\n"
           % (T, " ".join("mk_ab.%s = mk_%s;" % (f, f) for f, _ in F), T))
    return new_sig + "{\n" + pro + body.replace("blockIdx.x", "mk_bid") + "}\n"


if __name__ == "__main__":
    spec = importlib.util.spec_from_file_location("schedules", sys.argv[1]); mod = importlib.util.module_from_spec(spec)
    sys.modules["gen"] = sys.modules[__name__]; spec.loader.exec_module(mod)
    out = sys.argv[2]; os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "schedules.txt"), "w") as f:
        for s in mod.SCHEDULES: emit(s, out); f.write("%s %s\n" % (s.name, s.sched))

