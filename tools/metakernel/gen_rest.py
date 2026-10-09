"""Generated uhat, RqFace and GetW stages (the rest of the end-to-end residual), plain HIP.

uhat    interior: one launch over every face node,   uh = 0.5*(u(side1) + u(side2))     (UhatInteriorFused arithmetic)
        boundary: per face block, uh = Ubou_ib(u(side1 at face nodes), nlg at face nodes)  (UhatBlock, ib>0)
        The face-node geometry is mesh-only; it is computed once with production's own primitives (as its cache does).
RqFace  per face block, team of FPW faces: ug(g,m) = sum_q shapfgt[g + ngf q] uh[q,m,f];
        Rh[q + npf (c + ncq f)] = sum_g shapfgw[q + npf g] ((ug(g,m) nl[g,d]) jac[g]),  c = m + ncu d  (RqFaceFused arithmetic)
GetW    per Newton iteration, one fused pass over all element blocks: F = EoS(w), dw = (0 + (1/EoSdw(w)) F);
        production's per-block norms (EosBlockNorms2), convergence test and masked update are then applied as-is.
"""
NB = 10

def _body(name, macros):
    return "\n".join(["{ // %s" % name,
        "const dstype* param = b.par; const dstype* uinf = b.uinf; const dstype* tau = b.tau; const dstype time = b.time;"
        " (void)param; (void)uinf; (void)tau; (void)time;"]
        + ["#define %s %s" % kv for kv in macros] + ['#include "body_%s.inc"' % name]
        + ["#undef %s" % kv[0].split("(")[0] for kv in macros] + ["}"])

def emit_rest(sname, uhat=False, rqface=False, w=False):
    o = []
    if uhat:
        o += ["#define MK_HAS_UHAT 1",
              "struct MkUhIn { dstype* uh; const dstype *udg, *geo; const int *find1, *facecon; const dstype *par, *uinf, *tau; dstype time; int f1, nf, ib; const int* ghost; int fsel; };",
              "#define MK_FSKIP(gh, fsel, f) ((fsel) != 0 && (gh) != nullptr && ((gh)[f] != 0) != ((fsel) == 2))",
              'extern "C" __global__ void __launch_bounds__(256) MKU_%s_int(dstype* uh, const dstype* udg, const int* facecon, const int nfn, const int* ghost, const int fsel) {' % sname,
              "    const size_t t = (size_t)blockIdx.x * 256 + threadIdx.x; if (t >= (size_t)nfn * MK_NCU) return;",
              "    const int i = (int)(t % nfn), j = (int)(t / nfn), q = i % MK_NPF, f = i / MK_NPF;",
              "    if (MK_FSKIP(ghost, fsel, f)) return;   // pass selection (identical values: only ghost-adjacent faces change between passes)",
              "    const int k1 = facecon[2*i], k2 = facecon[2*i + 1]; if (k1 == k2) return;   // boundary face: Ubou",
              "    const int m1 = k1 % MK_NPE, n1 = k1 / MK_NPE, m2 = k2 % MK_NPE, n2 = k2 / MK_NPE;",
              "    uh[q + MK_NPF*j + (size_t)MK_NPF*MK_NCU*f] = 0.5*(udg[m1 + MK_NPE*j + (size_t)MK_NPE*MK_NC*n1] + udg[m2 + MK_NPE*j + (size_t)MK_NPE*MK_NC*n2]);",
              "}"]
        for ib in range(1, NB + 1):
            o += ['extern "C" __global__ void __launch_bounds__(64) MKU_%s_bou%d(const MkUhIn a) {' % (sname, ib),
                  "    const int nn = MK_NPF * a.nf; const int i = blockIdx.x * 64 + threadIdx.x; if (i >= nn) return;",
                  "    if (MK_FSKIP(a.ghost, a.fsel, a.f1 + i / MK_NPF)) return;",
                  "    const MkUhIn& b = a; dstype mk_f[MK_NCU];",
                  _body("Ubou%d" % ib, [("MK_RD_u(k)", "b.udg[b.find1[i + (size_t)nn*(k)]]"), ("MK_RD_n(k)", "b.geo[(size_t)nn*MK_NCX + i + (size_t)nn*(k)]"),
                                        ("MK_RD_x(k)", "b.geo[i + (size_t)nn*(k)]"), ("MK_RD_o(k)", "((dstype)0)"), ("MK_RD_w(k)", "((dstype)0)"),
                                        ("MK_RD_uh(k)", "((dstype)0)"), ("MK_WR(k)", "mk_f[k]")]),
                  "#pragma unroll",
                  "    for (int m = 0; m < MK_NCU; m++) b.uh[(i % MK_NPF) + MK_NPF*m + (size_t)MK_NPF*MK_NCU*(b.f1 + i / MK_NPF)] = mk_f[m];",
                  "}"]
        o += ["static void mk_uhat_int(dstype* uh, const dstype* udg, const int* facecon, int nf, const int* ghost, int fsel) { const int nfn = MK_NPF * nf;",
              "    hipLaunchKernelGGL(MKU_%s_int, dim3(((size_t)nfn*MK_NCU + 255) / 256), dim3(256), 0, mk_stream(), uh, udg, facecon, nfn, ghost, fsel); MK_HIPCHECK(\"uhat_int\"); }" % sname,
              "static void mk_uhat_bou(const MkUhIn& a) { const int g = (MK_NPF * a.nf + 63) / 64; switch (a.ib) {"]
        o += ["  case %d: hipLaunchKernelGGL(MKU_%s_bou%d, dim3(g), dim3(64), 0, mk_stream(), a); break;" % (ib, sname, ib) for ib in range(1, NB + 1)]
        o += ['  default: printf("MKERR no Ubou kernel for ib=%d\\n", a.ib); std::abort(); } MK_HIPCHECK("uhat_bou"); }']
    if rqface:
        rqv = rqface if rqface in (2, 3) else 1
        rq2 = ['extern "C" __global__ void __launch_bounds__(64) MKR_%s_rqface(const MkFaceIn a, const int* ghost, const int fsel) {' % sname,
               "    // v2: ug of the ncu components in LDS once, then one pass per normal direction d through an ncu-column buffer",
               "    //     (13.6 KB -> 9 KB LDS, no fully unrolled 81-load prologue); same arithmetic and summation order as v1",
               "    constexpr int MK_FPW = 64 / MK_NGF;",
               "    const int lane = threadIdx.x, fq = lane / MK_NGF, g = lane % MK_NGF, fl = blockIdx.x * MK_FPW + fq;",
               "    const bool ok = fq < MK_FPW && fl < a.nf && !MK_FSKIP(ghost, fsel, a.f1 + fl); const int nga = MK_NGF * a.nf; const size_t i = g + (size_t)MK_NGF * fl;",
               "    __shared__ dstype ugs[(64 / MK_NGF) * MK_NCU * MK_NGF], sv[(64 / MK_NGF) * MK_NCU * MK_NGF]; const MkFaceIn& b = a;",
               "    if (ok) {",
               "#pragma unroll 1",
               "      for (int m = 0; m < MK_NCU; m++) { dstype ug = 0;",
               "#pragma unroll",
               "        for (int q = 0; q < MK_NPF; q++) ug += b.gt[g + MK_NGF*q] * b.uh[q + MK_NPF*m + (size_t)MK_NPF*MK_NCU*(b.f1 + fl)];",
               "        ugs[(fq*MK_NCU + m)*MK_NGF + g] = ug; } }",
               "#pragma unroll 1",
               "    for (int d = 0; d < MK_ND; d++) {",
               "      __syncthreads();",
               "      if (ok) { const dstype nj = b.nl[i + (size_t)nga*d], jc = b.jac[i];",
               "#pragma unroll",
               "        for (int m = 0; m < MK_NCU; m++) sv[(fq*MK_NCU + m)*MK_NGF + g] = (ugs[(fq*MK_NCU + m)*MK_NGF + g] * nj) * jc; }",
               "      __syncthreads();",
               "      if (ok) { const int q = g;",
               "#pragma unroll 1",
               "        for (int m = 0; m < MK_NCU; m++) { dstype s = 0;",
               "#pragma unroll",
               "          for (int gg = 0; gg < MK_NGF; gg++) s += b.gw[q + MK_NPF*gg] * sv[(fq*MK_NCU + m)*MK_NGF + gg];",
               "          b.Rh[q + MK_NPF*(m + MK_NCU*d + MK_NCQ*(size_t)(b.f1 + fl))] = s; } }",
               "    }",
               "}"]
        rq3 = ['extern "C" __global__ void __launch_bounds__(64) MKR_%s_rqface(const MkFaceIn a, const int* ghost, const int fsel) {' % sname,
               "    // v3: each face's 81 trace values staged once into LDS with contiguous loads (v1/v2 load each 9 times),",
               "    //     interpolation from LDS, then v2's per-direction passes (the staging buffer is reused as sv); same order",
               "    constexpr int MK_FPW = 64 / MK_NGF;",
               "    const int lane = threadIdx.x, fq = lane / MK_NGF, g = lane % MK_NGF, fl = blockIdx.x * MK_FPW + fq;",
               "    const bool ok = fq < MK_FPW && fl < a.nf && !MK_FSKIP(ghost, fsel, a.f1 + fl); const int nga = MK_NGF * a.nf; const size_t i = g + (size_t)MK_NGF * fl;",
               "    __shared__ dstype ugs[MK_FPW * MK_NCU * MK_NGF], buf[MK_FPW * MK_NPF * MK_NCU]; const MkFaceIn& b = a;",
               "    for (int t = lane; t < MK_FPW * MK_NPF * MK_NCU; t += 64) { const int fqq = t / (MK_NPF*MK_NCU), r = t % (MK_NPF*MK_NCU), f2 = blockIdx.x * MK_FPW + fqq;",
               "        buf[t] = f2 < a.nf ? b.uh[r + (size_t)MK_NPF*MK_NCU*(b.f1 + f2)] : 0.0; }",
               "    __syncthreads();",
               "    if (ok) {",
               "#pragma unroll 1",
               "      for (int m = 0; m < MK_NCU; m++) { dstype ug = 0;",
               "#pragma unroll",
               "        for (int q = 0; q < MK_NPF; q++) ug += b.gt[g + MK_NGF*q] * buf[fq*MK_NPF*MK_NCU + q + MK_NPF*m];",
               "        ugs[(fq*MK_NCU + m)*MK_NGF + g] = ug; } }",
               "    dstype* sv = buf;",
               "#pragma unroll 1",
               "    for (int d = 0; d < MK_ND; d++) {",
               "      __syncthreads();",
               "      if (ok) { const dstype nj = b.nl[i + (size_t)nga*d], jc = b.jac[i];",
               "#pragma unroll",
               "        for (int m = 0; m < MK_NCU; m++) sv[(fq*MK_NCU + m)*MK_NGF + g] = (ugs[(fq*MK_NCU + m)*MK_NGF + g] * nj) * jc; }",
               "      __syncthreads();",
               "      if (ok) { const int q = g;",
               "#pragma unroll 1",
               "        for (int m = 0; m < MK_NCU; m++) { dstype s = 0;",
               "#pragma unroll",
               "          for (int gg = 0; gg < MK_NGF; gg++) s += b.gw[q + MK_NPF*gg] * sv[(fq*MK_NCU + m)*MK_NGF + gg];",
               "          b.Rh[q + MK_NPF*(m + MK_NCU*d + MK_NCQ*(size_t)(b.f1 + fl))] = s; } }",
               "    }",
               "}"]
        if rqface == 4:   # v1 arithmetic, all face blocks in ONE launch per pass (block found from a workgroup-offset table)
            o += ["#define MK_HAS_RQFACE 1", "#define MK_RQFACE_ALL 1",
                  "#ifndef MK_FSKIP",
                  "#define MK_FSKIP(gh, fsel, f) ((fsel) != 0 && (gh) != nullptr && ((gh)[f] != 0) != ((fsel) == 2))",
                  "#endif",
                  'extern "C" __global__ void __launch_bounds__(64) MKR_%s_rqface_all(const MkFaceIn* blocks, const int* wgoff, const int nb, const int* ghost, const int fsel) {' % sname,
                  "    int j = 0; while (j + 1 < nb && (int)blockIdx.x >= wgoff[j + 1]) j++;",
                  "    const MkFaceIn& b = blocks[j]; const MkFaceIn& a = b; const int wg = (int)blockIdx.x - wgoff[j];",
                  "    constexpr int MK_FPW = 64 / MK_NGF;",
                  "    const int lane = threadIdx.x, fq = lane / MK_NGF, g = lane % MK_NGF, fl = wg * MK_FPW + fq;",
                  "    const bool ok = fq < MK_FPW && fl < a.nf && !MK_FSKIP(ghost, fsel, a.f1 + fl); const int nga = MK_NGF * a.nf; const size_t i = g + (size_t)MK_NGF * fl;",
                  "    __shared__ dstype sv[(64 / MK_NGF) * MK_NCQ * MK_NGF];",
                  "    if (ok) {",
                  "#pragma unroll",
                  "      for (int m = 0; m < MK_NCU; m++) { dstype ug = 0;",
                  "#pragma unroll",
                  "        for (int q = 0; q < MK_NPF; q++) ug += b.gt[g + MK_NGF*q] * b.uh[q + MK_NPF*m + (size_t)MK_NPF*MK_NCU*(b.f1 + fl)];",
                  "#pragma unroll",
                  "        for (int d = 0; d < MK_ND; d++) sv[(fq*MK_NCQ + m + MK_NCU*d)*MK_NGF + g] = (ug * b.nl[i + (size_t)nga*d]) * b.jac[i]; } }",
                  "    __syncthreads();",
                  "    if (ok) { const int q = g;",
                  "      for (int c = 0; c < MK_NCQ; c++) { dstype s = 0;",
                  "#pragma unroll",
                  "        for (int gg = 0; gg < MK_NGF; gg++) s += b.gw[q + MK_NPF*gg] * sv[(fq*MK_NCQ + c)*MK_NGF + gg];",
                  "        b.Rh[q + MK_NPF*(c + MK_NCQ*(size_t)(b.f1 + fl))] = s; } }",
                  "}",
                  "static void mk_rqface_all(const MkFaceIn* blocks, const int* wgoff, int nb, int nwg, const int* ghost, int fsel) {",
                  "    hipLaunchKernelGGL(MKR_%s_rqface_all, dim3(nwg), dim3(64), 0, mk_stream(), blocks, wgoff, nb, ghost, fsel); MK_HIPCHECK(\"rqface_all\"); }" % sname]
        else:
            pass
    if rqface and rqface != 4:
        o += ["#define MK_HAS_RQFACE 1",
              "#ifndef MK_FSKIP",
              "#define MK_FSKIP(gh, fsel, f) ((fsel) != 0 && (gh) != nullptr && ((gh)[f] != 0) != ((fsel) == 2))",
              "#endif",
              ] + (rq3 if rqv == 3 else rq2 if rqv == 2 else [              'extern "C" __global__ void __launch_bounds__(64) MKR_%s_rqface(const MkFaceIn a, const int* ghost, const int fsel) {' % sname,
              "    constexpr int MK_FPW = 64 / MK_NGF;",
              "    const int lane = threadIdx.x, fq = lane / MK_NGF, g = lane % MK_NGF, fl = blockIdx.x * MK_FPW + fq;",
              "    const bool ok = fq < MK_FPW && fl < a.nf && !MK_FSKIP(ghost, fsel, a.f1 + fl); const int nga = MK_NGF * a.nf; const size_t i = g + (size_t)MK_NGF * fl;",
              "    __shared__ dstype sv[(64 / MK_NGF) * MK_NCQ * MK_NGF]; const MkFaceIn& b = a;",
              "    if (ok) {",
              "#pragma unroll",
              "      for (int m = 0; m < MK_NCU; m++) { dstype ug = 0;",
              "#pragma unroll",
              "        for (int q = 0; q < MK_NPF; q++) ug += b.gt[g + MK_NGF*q] * b.uh[q + MK_NPF*m + (size_t)MK_NPF*MK_NCU*(b.f1 + fl)];",
              "#pragma unroll",
              "        for (int d = 0; d < MK_ND; d++) sv[(fq*MK_NCQ + m + MK_NCU*d)*MK_NGF + g] = (ug * b.nl[i + (size_t)nga*d]) * b.jac[i]; } }",
              "    __syncthreads();",
              "    if (ok) { const int q = g;",
              "      for (int c = 0; c < MK_NCQ; c++) { dstype s = 0;",
              "#pragma unroll",
              "        for (int gg = 0; gg < MK_NGF; gg++) s += b.gw[q + MK_NPF*gg] * sv[(fq*MK_NCQ + c)*MK_NGF + gg];",
              "        b.Rh[q + MK_NPF*(c + MK_NCQ*(size_t)(b.f1 + fl))] = s; } }",
              "}"]) + [
              "static void mk_rqface_block(const MkFaceIn& a, const int* ghost, int fsel) { const int fpw = 64 / MK_NGF;",
              "    hipLaunchKernelGGL(MKR_%s_rqface, dim3((a.nf + fpw - 1) / fpw), dim3(64), 0, mk_stream(), a, ghost, fsel); MK_HIPCHECK(\"rqface\"); }" % sname]
    if w:
        o += ["#define MK_HAS_W 1",
              "struct MkWIn { const dstype *x, *u, *o, *w; dstype *F, *dw; const dstype *par, *uinf, *tau; dstype time; int n; };",
              'extern "C" __global__ void __launch_bounds__(64) MKW_%s_eos(const MkWIn a) {' % sname,
              "    const int t = blockIdx.x * 64 + threadIdx.x; if (t >= a.n) return; const int j = t % MK_NPE, k = t / MK_NPE;",
              "    const MkWIn& b = a; dstype F, Fw;"]
        el = lambda arr, n: "b.%s[j + MK_NPE*(c) + (size_t)MK_NPE*%s*k]" % (arr, n)
        for nm, dst in (("EoS", "F"), ("EoSdw", "Fw")):
            o.append(_body(nm, [("MK_RD_u(c)", el("u", "MK_NC")), ("MK_RD_w(c)", el("w", "MK_NCW")), ("MK_RD_x(c)", el("x", "MK_NCX")),
                                ("MK_RD_o(c)", el("o", "MK_NCO")), ("MK_WR(c)", dst)]))
        o += ["    b.F[t] = F; const dstype inv = 1.0/Fw; dstype s = 0.0; s += inv*F; b.dw[t] = s;   // ArrayEosInverseMatrix11 + ArrayEosMatrixMultiplication",
              "}",
              "static void mk_w_fused(const MkWIn& a) { hipLaunchKernelGGL(MKW_%s_eos, dim3((a.n + 63) / 64), dim3(64), 0, mk_stream(), a); MK_HIPCHECK(\"w\"); }" % sname]
    return "\n".join(o)
