"""Face-residual generator for the meta-kernel (RuFace replacement), plain-HIP kernels.

One launch per production face block (fblks triple: f1, f2, ib). A team holds FPW = floor(64*waves / NGF) whole faces;
lane (fq, g) owns face fq's Gauss point g for the pointwise part and face node q for the integration.
  interior (ib==0):  interp u1,w1,u2,w2 at g  ->  F1 = Flux(side 1), F2 = Flux(side 2)
                     fh_m = ((a_m n0 + a_{m+9} n1) + a_{m+18} n2) + tau0 (u1_m - u2_m),  a = 0.5 (F1 + F2)   (FhatDriver order)
  boundary (ib>0):   interp u1,w1,uhg at g  ->  fh = Fbou_ib(u1, og1, w1, uhg, nl, tau)
  both:              v = fh*jac -> LDS -> Rh[q + npf (m + ncu fl)] = sum_g shapfgw[q + npf g] v_g        (RuFacePostFused order)
Interpolation sums run over face nodes in ascending order like RuFacePreFused. Options (dict `face=` on a schedule):
  mode: "fused" (one kernel per block, above) | "split" (interp kernel -> side-1 flux kernel -> side-2 flux + combine +
        integrate kernel; boundary: interp -> Fbou + integrate). The split interp is one thread per (point, column), as
        RuFacePreFused, writing [i + nga*col] columns (interior: u1 | w1 | u2 | w2, boundary: uh | u1 | w1).
  hand: (fused) where F1 lives while F2 is computed -- "reg" | "lds" | "glb" (per-lane L2 buffer)
  lb:   waves/SIMD target;  waves: waves per team
"""
NB = 10   # boundary kernels generated: KokkosFbou1..NB

def _flux_block(tag, u, w, o, out, rd_x="b.xg[i + (size_t)nga*(k)]"):
    return "\n".join([
        "{ // %s" % tag,
        "const dstype* param = b.par; const dstype* uinf = b.uinf; const dstype time = b.time; (void)param; (void)uinf; (void)time;",
        "#define MK_RD_u(k) %s" % u, "#define MK_RD_w(k) %s" % w, "#define MK_RD_o(k) %s" % o,
        "#define MK_RD_x(k) %s" % rd_x, "#define MK_WR(k) %s" % out,
        '#include "body_Flux.inc"',
        "#undef MK_RD_u", "#undef MK_RD_w", "#undef MK_RD_o", "#undef MK_RD_x", "#undef MK_WR", "}"])

def _interp(dst, comps, src_expr):
    # dst(c) = sum_q shapfgt[g + NGF q] * src(q, c), ascending q
    return ("#pragma unroll\nfor (int c = 0; c < %s; c++) { dstype mk_s = 0;\n#pragma unroll\n"
            "  for (int q = 0; q < MK_NPF; q++) mk_s += b.gt[g + MK_NGF*q] * %s;\n  %s = mk_s; }") % (comps, src_expr, dst)

def _head(nm, tp, lb):
    return ['extern "C" __global__ void __attribute__((amdgpu_flat_work_group_size(1, %d), amdgpu_waves_per_eu(%d)))' % (tp, lb),
            "%s(const MkFaceIn a) {" % nm, "    constexpr int MK_TP = %d, MK_FPW = MK_TP / MK_NGF;" % tp,
            "    const int lane = threadIdx.x, fq = lane / MK_NGF, g = lane % MK_NGF;",
            "    const int fl = blockIdx.x * MK_FPW + fq;             // face index local to the production block",
            "    const bool ok = fq < MK_FPW && fl < a.nf; const int nga = MK_NGF * a.nf; const size_t i = g + (size_t)MK_NGF * fl;",
            "    extern __shared__ dstype mk_shm[]; dstype* sv = mk_shm;   // [FPW][NCU][NGF] jac-scaled flux",
            "    const MkFaceIn& b = a;"]

def _tail():
    return ["    __syncthreads();",
            "    if (ok) { const int q = g;   // lane (fq, q): face node q of face fq",
            "#pragma unroll",
            "      for (int m = 0; m < MK_NCU; m++) { dstype mk_s = 0;",
            "#pragma unroll",
            "        for (int gg = 0; gg < MK_NGF; gg++) mk_s += b.gw[q + MK_NPF*gg] * sv[(fq*MK_NCU + m)*MK_NGF + gg];",
            "        b.Rh[q + MK_NPF*(m + MK_NCU*(size_t)fl)] = mk_s; } }", "}"]

def _side_gathers(side):
    """udg via find{1,2} (as RuFacePreFused), wdg via facecon side."""
    fi = "b.find%d" % side
    u = "b.udg[%s[q + MK_NPF*fl + (size_t)MK_NPF*b.nf*c]]" % fi
    w = ("b.wdg[(b.facecon[2*(MK_NPF*(b.f1 + fl) + q) + %d] %% MK_NPE) + MK_NPE*c + (size_t)MK_NPE*MK_NCW*"
         "(b.facecon[2*(MK_NPF*(b.f1 + fl) + q) + %d] / MK_NPE)]") % (side - 1, side - 1)
    return u, w

def emit_face(sname, opt):
    if opt.get("mode", "fused") == "split": return emit_face_split(sname, opt)
    hand, lb, waves = opt.get("hand", "reg"), opt.get("lb", 1), opt.get("waves", 1)
    tp = 64 * waves
    code = ["// ---- generated face residual (%s) ----" % opt, "#define MK_HAS_FACE 1"]
    shm_int = "(size_t)(%d / MK_NGF) * MK_NCU * MK_NGF * sizeof(dstype)" % tp
    if hand == "lds": shm_int += " + (size_t)MK_NF * %d * sizeof(dstype)" % tp
    u1, w1 = _side_gathers(1); u2, w2 = _side_gathers(2)
    # interior kernel
    nm = "MKF_%s_int" % sname
    o = _head(nm, tp, lb)
    o += ["    dstype mk_u1[MK_NC], mk_w1[MK_NCW], mk_u2[MK_NC], mk_w2[MK_NCW], mk_f2[MK_NF], mk_fh[MK_NCU];"]
    if hand == "reg": o.append("    dstype mk_f1[MK_NF];"); f1 = "mk_f1[k]"
    elif hand == "lds": o.append("    dstype* mk_L_f1 = mk_shm + (MK_TP / MK_NGF) * MK_NCU * MK_NGF;"); f1 = "mk_L_f1[(k)*MK_TP + lane]"
    else: f1 = "b.gf1[(size_t)(k)*nga + i]"
    o += ["    if (ok) {", _interp("mk_u1[c]", "MK_NC", u1), _interp("mk_w1[c]", "MK_NCW", w1),
          _interp("mk_u2[c]", "MK_NC", u2), _interp("mk_w2[c]", "MK_NCW", w2),
          _flux_block("F1 = Flux(side 1)", "mk_u1[k]", "mk_w1[k]", "b.og1[i + (size_t)nga*(k)]", f1)]
    if hand != "reg": o.append("    MK_CFENCE();")
    o += [_flux_block("F2 = Flux(side 2)", "mk_u2[k]", "mk_w2[k]", "b.og2[i + (size_t)nga*(k)]", "mk_f2[k]"),
          "#pragma unroll",
          "    for (int m = 0; m < MK_NCU; m++) {",
          "      dstype mk_s = (0.5 * (%s + mk_f2[m])) * b.nl[i];" % f1.replace("(k)", "(m)").replace("[k]", "[m]"),
          "      mk_s += (0.5 * (%s + mk_f2[m + MK_NCU])) * b.nl[i + nga];" % f1.replace("(k)", "(m + MK_NCU)").replace("[k]", "[m + MK_NCU]"),
          "      mk_s += (0.5 * (%s + mk_f2[m + 2*MK_NCU])) * b.nl[i + 2*nga];" % f1.replace("(k)", "(m + 2*MK_NCU)").replace("[k]", "[m + 2*MK_NCU]"),
          "      mk_fh[m] = mk_s + b.tau[0] * (mk_u1[m] - mk_u2[m]); }",
          "#pragma unroll",
          "    for (int m = 0; m < MK_NCU; m++) sv[(fq*MK_NCU + m)*MK_NGF + g] = mk_fh[m] * b.jac[i];",
          "    }"]
    o += _tail()
    code.append("\n".join(o))
    code.append("static size_t %s_SHM() { return %s; }" % (nm, shm_int))
    # boundary kernels, one per boundary type
    shm_b = "(size_t)(%d / MK_NGF) * MK_NCU * MK_NGF * sizeof(dstype)" % tp
    for ib in range(1, NB + 1):
        nmb = "MKF_%s_bou%d" % (sname, ib)
        o = _head(nmb, tp, lb)
        o += ["    dstype mk_u1[MK_NC], mk_w1[MK_NCW], mk_uh[MK_NCU], mk_fh[MK_NCU];",
              "    if (ok) {", _interp("mk_uh[c]", "MK_NCU", "b.uh[q + MK_NPF*c + (size_t)MK_NPF*MK_NCU*(b.f1 + fl)]"),
              _interp("mk_u1[c]", "MK_NC", u1), _interp("mk_w1[c]", "MK_NCW", w1),
              "{ // Fbou%d" % ib,
              "const dstype* param = b.par; const dstype* uinf = b.uinf; const dstype* tau = b.tau; const dstype time = b.time; (void)param; (void)uinf; (void)tau; (void)time;",
              "#define MK_RD_u(k) mk_u1[k]", "#define MK_RD_w(k) mk_w1[k]", "#define MK_RD_o(k) b.og1[i + (size_t)nga*(k)]",
              "#define MK_RD_uh(k) mk_uh[k]", "#define MK_RD_n(k) b.nl[i + (size_t)nga*(k)]", "#define MK_RD_x(k) b.xg[i + (size_t)nga*(k)]",
              "#define MK_WR(k) mk_fh[k]", '#include "body_Fbou%d.inc"' % ib,
              "#undef MK_RD_u", "#undef MK_RD_w", "#undef MK_RD_o", "#undef MK_RD_uh", "#undef MK_RD_n", "#undef MK_RD_x", "#undef MK_WR", "}",
              "#pragma unroll",
              "    for (int m = 0; m < MK_NCU; m++) sv[(fq*MK_NCU + m)*MK_NGF + g] = mk_fh[m] * b.jac[i];",
              "    }"]
        o += _tail()
        code.append("\n".join(o))
    # host launcher: one launch per production face block
    L = ["static void mk_face_block(const MkFaceIn& a) {",
         "  const int grid = (a.nf + (%d / MK_NGF) - 1) / (%d / MK_NGF);" % (tp, tp),
         "  if (a.ib == 0) { hipLaunchKernelGGL(%s, dim3(grid), dim3(%d), %s_SHM(), mk_stream(), a); MK_HIPCHECK(\"%s\"); return; }" % (nm, tp, nm, nm),
         "  switch (a.ib) {"]
    for ib in range(1, NB + 1):
        L.append("  case %d: hipLaunchKernelGGL(MKF_%s_bou%d, dim3(grid), dim3(%d), %s, mk_stream(), a); MK_HIPCHECK(\"bou%d\"); break;" % (ib, sname, ib, tp, shm_b, ib))
    L += ["  default: printf(\"MKERR no generated boundary kernel for ib=%d\\n\", a.ib); std::abort();", "  }", "}",
          'static const char* MK_FACE_LABEL = "face{hand=%s,lb=%d,waves=%d}";' % (hand, lb, waves)]
    code.append("\n".join(L))
    return "\n".join(code)


def emit_face_split(sname, opt):
    lb, lbi, waves = opt.get("lb", 2), opt.get("lbi", 1), opt.get("waves", 1)
    tp = 64 * waves
    code = ["// ---- generated face residual, split (%s) ----" % opt, "#define MK_HAS_FACE 1"]
    u1, w1 = _side_gathers(1); u2, w2 = _side_gathers(2)
    # interpolation kernel: thread per (i, col); columns follow the block kind
    def interp_kernel(nm, cols):
        o = ['extern "C" __global__ void __launch_bounds__(256) %s(const MkFaceIn a, const int ncol) {' % nm,
             "    const size_t t = (size_t)blockIdx.x * 256 + threadIdx.x; const int nga = MK_NGF * a.nf;",
             "    if (t >= (size_t)nga * ncol) return;",
             "    const int ii = (int)(t % nga), col = (int)(t / nga), g = ii % MK_NGF, fl = ii / MK_NGF; const MkFaceIn& b = a;",
             "    dstype mk_s = 0;"]
        first = True
        for (lo, hi, src) in cols:
            o.append("    %sif (col < %s) { const int c = col - (%s);" % ("" if first else "else ", hi, lo)); first = False
            o.append("#pragma unroll")
            o.append("      for (int q = 0; q < MK_NPF; q++) mk_s += b.gt[g + MK_NGF*q] * %s; }" % src)
        o += ["    b.gug[t] = mk_s;", "}"]
        return "\n".join(o)
    uh = "b.uh[q + MK_NPF*c + (size_t)MK_NPF*MK_NCU*(b.f1 + fl)]"
    def interp_staged_kernel(nm, cols):
        # each workgroup: MK_SFB faces x MK_SCW columns; every face-node value is gathered ONCE into LDS (the plain kernel
        # gathers it once per Gauss point, 9x), then the same ascending-q sums run against LDS -> bitwise identical
        o = ['extern "C" __global__ void __launch_bounds__(256) %s(const MkFaceIn a, const int ncol) {' % nm,
             "    constexpr int MK_SFB = 8, MK_SCW = 32;",
             "    __shared__ dstype L[MK_SFB * MK_NPF * MK_SCW];",
             "    const int fb0 = blockIdx.x * MK_SFB, c0 = blockIdx.y * MK_SCW, nga = MK_NGF * a.nf; const MkFaceIn& b = a;",
             "    for (int t = threadIdx.x; t < MK_SFB * MK_NPF * MK_SCW; t += 256) {",
             "        const int q = t % MK_NPF, r = t / MK_NPF, fl = fb0 + r % MK_SFB, col = c0 + r / MK_SFB; dstype v = 0;",
             "        if (fl < a.nf && col < ncol) {"]
        first = True
        for (lo, hi, src) in cols:
            o.append("            %sif (col < %s) { const int c = col - (%s); v = %s; }" % ("" if first else "else ", hi, lo, src)); first = False
        o += ["        }", "        L[t] = v; }",
              "    __syncthreads();",
              "    for (int t = threadIdx.x; t < MK_SFB * MK_NGF * MK_SCW; t += 256) {",
              "        const int g = t % MK_NGF, r = t / MK_NGF, fq = r % MK_SFB, cc = r / MK_SFB, fl = fb0 + fq, col = c0 + cc;",
              "        if (fl >= a.nf || col >= ncol) continue;",
              "        dstype mk_s = 0;",
              "#pragma unroll",
              "        for (int q = 0; q < MK_NPF; q++) mk_s += b.gt[g + MK_NGF*q] * L[q + MK_NPF*(fq + MK_SFB*cc)];",
              "        b.gug[(g + (size_t)MK_NGF*fl) + (size_t)nga*col] = mk_s; }",
              "}"]
        return "\n".join(o)
    sfb, tpb = int(opt.get("sfb", 4)), int(opt.get("tpb", 256))
    def interp_conn_kernel(nm):
        # staged interpolation with per-NODE connectivity: the absolute udg/wdg index of every face node comes from facecon
        # (2 ints per node) instead of findudg (one int per node per column, nc per node); setup verifies findudg equals
        # the facecon formula. Same values, same ascending-q sums from LDS -> bitwise identical to interps_int.
        return "\n".join([
            'extern "C" __global__ void __launch_bounds__(%d) %s(const MkFaceIn a) {' % (tpb, nm),
            "    constexpr int MK_SFB = %d, MK_TPB = %d, NCOL = 2*MK_NC + 2*MK_NCW;" % (sfb, tpb),
            "    __shared__ dstype L[MK_SFB * MK_NPF * NCOL]; __shared__ int K[2 * MK_SFB * MK_NPF];",
            "    const int fb0 = blockIdx.x * MK_SFB, nga = MK_NGF * a.nf; const MkFaceIn& b = a;",
            "    for (int t = threadIdx.x; t < 2 * MK_SFB * MK_NPF; t += MK_TPB) { const int fl = fb0 + t / (2*MK_NPF);",
            "        K[t] = fl < a.nf ? b.facecon[2*(MK_NPF*(b.f1 + fb0) ) + t] : 0; }",
            "    __syncthreads();",
            "    for (int t = threadIdx.x; t < MK_SFB * MK_NPF * NCOL; t += MK_TPB) {",
            "        const int q = t % MK_NPF, r = t / MK_NPF, fq = r % MK_SFB, col = r / MK_SFB, fl = fb0 + fq; dstype v = 0;",
            "        if (fl < a.nf) {",
            "            int side = 0, c = col;",
            "            if (c >= MK_NC + MK_NCW) { side = 1; c -= MK_NC + MK_NCW; }",
            "            const int k = K[2*(MK_NPF*fq + q) + side], m = k % MK_NPE, n = k / MK_NPE;",
            "            v = c < MK_NC ? b.udg[m + MK_NPE*c + (size_t)MK_NPE*MK_NC*n] : b.wdg[m + MK_NPE*(c - MK_NC) + (size_t)MK_NPE*MK_NCW*n]; }",
            "        L[q + MK_NPF*(fq + MK_SFB*col)] = v; }",
            "    __syncthreads();",
            "    for (int t = threadIdx.x; t < MK_SFB * MK_NGF * NCOL; t += MK_TPB) {",
            "        const int g = t % MK_NGF, r = t / MK_NGF, fq = r % MK_SFB, col = r / MK_SFB, fl = fb0 + fq;",
            "        if (fl >= a.nf) continue;",
            "        dstype mk_s = 0;",
            "#pragma unroll",
            "        for (int q = 0; q < MK_NPF; q++) mk_s += b.gt[g + MK_NGF*q] * L[q + MK_NPF*(fq + MK_SFB*col)];",
            "        b.gug[(g + (size_t)MK_NGF*fl) + (size_t)nga*col] = mk_s; }",
            "}"])
    def gather_kernel(nm, cols):
        # gat[i + nfn*col] = value at face node i = q + npf*fl, column col: the GetFaceNodes/GetArrayAtIndex gathers in one pass
        o = ['extern "C" __global__ void __launch_bounds__(256) %s(const MkFaceIn a, const int ncol) {' % nm,
             "    const size_t t = (size_t)blockIdx.x * 256 + threadIdx.x; const int nfn = MK_NPF * a.nf;",
             "    if (t >= (size_t)nfn * ncol) return;",
             "    const int ii = (int)(t % nfn), col = (int)(t / nfn), q = ii % MK_NPF, fl = ii / MK_NPF; const MkFaceIn& b = a;",
             "    dstype v = 0;"]
        first = True
        for (lo, hi, src) in cols:
            o.append("    %sif (col < %s) { const int c = col - (%s); v = %s; }" % ("" if first else "else ", hi, lo, src)); first = False
        o += ["    b.gat[t] = v;", "}"]
        return "\n".join(o)
    if opt.get("interp", "fused") == "gemm":
        code.append(gather_kernel("MKF_%s_gather_int" % sname, [("0", "MK_NC", u1), ("MK_NC", "MK_NC + MK_NCW", w1),
                    ("MK_NC + MK_NCW", "2*MK_NC + MK_NCW", u2), ("2*MK_NC + MK_NCW", "2*MK_NC + 2*MK_NCW", w2)]))
        code.append(gather_kernel("MKF_%s_gather_bou" % sname, [("0", "MK_NCU", uh), ("MK_NCU", "MK_NCU + MK_NC", u1),
                    ("MK_NCU + MK_NC", "MK_NCU + MK_NC + MK_NCW", w1)]))
    code.append(interp_kernel("MKF_%s_interp_int" % sname, [("0", "MK_NC", u1), ("MK_NC", "MK_NC + MK_NCW", w1),
                ("MK_NC + MK_NCW", "2*MK_NC + MK_NCW", u2), ("2*MK_NC + MK_NCW", "2*MK_NC + 2*MK_NCW", w2)]))
    code.append(interp_kernel("MKF_%s_interp_bou" % sname, [("0", "MK_NCU", uh), ("MK_NCU", "MK_NCU + MK_NC", u1),
                ("MK_NCU + MK_NC", "MK_NCU + MK_NC + MK_NCW", w1)]))
    conn = opt.get("interp", "fused") == "conn"
    staged = opt.get("interp", "fused") in ("staged", "conn")
    if conn:
        code.append("#define MK_FACE_CONN 1")
        code.append(interp_conn_kernel("MKF_%s_interpc_int" % sname))
    if staged:
        code.append(interp_staged_kernel("MKF_%s_interps_int" % sname, [("0", "MK_NC", u1), ("MK_NC", "MK_NC + MK_NCW", w1),
                    ("MK_NC + MK_NCW", "2*MK_NC + MK_NCW", u2), ("2*MK_NC + MK_NCW", "2*MK_NC + 2*MK_NCW", w2)]))
    G = lambda off: "b.gug[i + (size_t)nga*((%s) + (k))]" % off
    # side-1 flux, one lane per point
    nm1 = "MKF_%s_flux1" % sname
    o = ['extern "C" __global__ void __attribute__((amdgpu_flat_work_group_size(1, 64), amdgpu_waves_per_eu(%d)))' % lb,
         "%s(const MkFaceIn a) {" % nm1,
         "    const int nga = MK_NGF * a.nf; const size_t i = (size_t)blockIdx.x * 64 + threadIdx.x; if (i >= (size_t)nga) return;",
         "    const MkFaceIn& b = a;",
         _flux_block("F1 = Flux(side 1)", G("0"), G("MK_NC"), "b.og1[i + (size_t)nga*(k)]", "b.gf1[(size_t)(k)*nga + i]"), "}"]
    code.append("\n".join(o))
    # side-2 flux + combine + integrate, face map
    nm2 = "MKF_%s_flux2" % sname
    o = _head(nm2, tp, lb)
    o += ["    dstype mk_f2[MK_NF], mk_fh[MK_NCU];", "    if (ok) {",
          _flux_block("F2 = Flux(side 2)", G("MK_NC + MK_NCW"), G("2*MK_NC + MK_NCW"), "b.og2[i + (size_t)nga*(k)]", "mk_f2[k]"),
          "#pragma unroll",
          "    for (int m = 0; m < MK_NCU; m++) {",
          "      dstype mk_s = (0.5 * (b.gf1[(size_t)(m)*nga + i] + mk_f2[m])) * b.nl[i];",
          "      mk_s += (0.5 * (b.gf1[(size_t)(m + MK_NCU)*nga + i] + mk_f2[m + MK_NCU])) * b.nl[i + nga];",
          "      mk_s += (0.5 * (b.gf1[(size_t)(m + 2*MK_NCU)*nga + i] + mk_f2[m + 2*MK_NCU])) * b.nl[i + 2*nga];",
          "      mk_fh[m] = mk_s + b.tau[0] * (b.gug[i + (size_t)nga*m] - b.gug[i + (size_t)nga*(MK_NC + MK_NCW + m)]); }",
          "#pragma unroll",
          "    for (int m = 0; m < MK_NCU; m++) sv[(fq*MK_NCU + m)*MK_NGF + g] = mk_fh[m] * b.jac[i];", "    }"]
    o += _tail()
    code.append("\n".join(o))
    shm = "(size_t)(%d / MK_NGF) * MK_NCU * MK_NGF * sizeof(dstype)" % tp
    for ib in range(1, NB + 1):
        nmb = "MKF_%s_bou%d" % (sname, ib)
        o = _head(nmb, tp, lbi)
        o += ["    dstype mk_fh[MK_NCU];", "    if (ok) {",
              "{ // Fbou%d" % ib,
              "const dstype* param = b.par; const dstype* uinf = b.uinf; const dstype* tau = b.tau; const dstype time = b.time; (void)param; (void)uinf; (void)tau; (void)time;",
              "#define MK_RD_uh(k) %s" % G("0"), "#define MK_RD_u(k) %s" % G("MK_NCU"), "#define MK_RD_w(k) %s" % G("MK_NCU + MK_NC"),
              "#define MK_RD_o(k) b.og1[i + (size_t)nga*(k)]", "#define MK_RD_n(k) b.nl[i + (size_t)nga*(k)]", "#define MK_RD_x(k) b.xg[i + (size_t)nga*(k)]",
              "#define MK_WR(k) mk_fh[k]", '#include "body_Fbou%d.inc"' % ib,
              "#undef MK_RD_u", "#undef MK_RD_w", "#undef MK_RD_o", "#undef MK_RD_uh", "#undef MK_RD_n", "#undef MK_RD_x", "#undef MK_WR", "}",
              "#pragma unroll",
              "    for (int m = 0; m < MK_NCU; m++) sv[(fq*MK_NCU + m)*MK_NGF + g] = mk_fh[m] * b.jac[i];", "    }"]
        o += _tail()
        code.append("\n".join(o))
    # ---- fi=True: the interior interpolation fused into both flux kernels. Each wave stages its FPW faces' node values
    #      (side s: u_s | w_s, CH columns at a time) in LDS with all lanes' gathers in flight, then each lane interpolates
    #      its Gauss point into registers (ascending q, as the interp kernel) -> no gug round trip for u/w.
    #      flux1 also stores u1[0:ncu] at gug[i + nga m] (the tau term flux2 reads, at the same place as before).
    fi = opt.get("fi", False); CH = opt.get("ch", 19)
    if fi:
        def stage(side, ru, rw):
            us, ws = _side_gathers(side)
            return ["#pragma unroll",
                    "    for (int c0 = 0; c0 < MK_NC + MK_NCW; c0 += MK_CH) {",
                    "      for (int t = lane; t < MK_FPW*MK_NPF*MK_CH; t += 64) { const int fqq = t / (MK_NPF*MK_CH), r = t % (MK_NPF*MK_CH), cc = r / MK_NPF, q = r % MK_NPF;",
                    "        const int fl = blockIdx.x * MK_FPW + fqq, col = c0 + cc; dstype v = 0;",
                    "        if (fl < b.nf && col < MK_NC) { const int c = col; v = %s; }" % us,
                    "        else if (fl < b.nf && col < MK_NC + MK_NCW) { const int c = col - MK_NC; v = %s; }" % ws,
                    "        mk_L[t] = v; }",
                    "      __syncthreads();",
                    "      if (ok) {",
                    "#pragma unroll",
                    "        for (int cc = 0; cc < MK_CH; cc++) { const int col = c0 + cc; if (col >= MK_NC + MK_NCW) break; dstype mk_s = 0;",
                    "#pragma unroll",
                    "          for (int q = 0; q < MK_NPF; q++) mk_s += b.gt[g + MK_NGF*q] * mk_L[(fq*MK_CH + cc)*MK_NPF + q];",
                    "          if (col < MK_NC) %s[col] = mk_s; else %s[col - MK_NC] = mk_s; } }" % (ru, rw),
                    "      __syncthreads();",
                    "    }"]
        nmf1 = "MKF_%s_flux1f" % sname
        o = ['extern "C" __global__ void __attribute__((amdgpu_flat_work_group_size(1, 64), amdgpu_waves_per_eu(%d)))' % lb,
             "%s(const MkFaceIn a) {" % nmf1,
             "    constexpr int MK_FPW = 64 / MK_NGF, MK_CH = %d;" % CH,
             "    __shared__ dstype mk_L[MK_FPW*MK_NPF*MK_CH];",
             "    const int lane = threadIdx.x, fq = lane / MK_NGF, g = lane % MK_NGF, fl = blockIdx.x * MK_FPW + fq;",
             "    const bool ok = fq < MK_FPW && fl < a.nf; const int nga = MK_NGF * a.nf; const size_t i = g + (size_t)MK_NGF * fl;",
             "    const MkFaceIn& b = a;", "    dstype mk_u[MK_NC], mk_w[MK_NCW];"] + stage(1, "mk_u", "mk_w") + [
             "    if (!ok) return;",
             "#pragma unroll",
             "    for (int m = 0; m < MK_NCU; m++) b.gug[i + (size_t)nga*m] = mk_u[m];",
             _flux_block("F1 = Flux(side 1)", "mk_u[k]", "mk_w[k]", "b.og1[i + (size_t)nga*(k)]", "b.gf1[(size_t)(k)*nga + i]"), "}"]
        code.append("\n".join(o))
        nmf2 = "MKF_%s_flux2f" % sname
        o = _head(nmf2, 64, lb)
        o += ["    constexpr int MK_CH = %d; dstype* mk_L = mk_shm;   // staging, then (after the last barrier) sv" % CH,
              "    dstype mk_u[MK_NC], mk_w[MK_NCW], mk_f2[MK_NF], mk_fh[MK_NCU];"] + stage(2, "mk_u", "mk_w") + ["    if (ok) {",
              _flux_block("F2 = Flux(side 2)", "mk_u[k]", "mk_w[k]", "b.og2[i + (size_t)nga*(k)]", "mk_f2[k]"),
              "#pragma unroll",
              "    for (int m = 0; m < MK_NCU; m++) {",
              "      dstype mk_s = (0.5 * (b.gf1[(size_t)(m)*nga + i] + mk_f2[m])) * b.nl[i];",
              "      mk_s += (0.5 * (b.gf1[(size_t)(m + MK_NCU)*nga + i] + mk_f2[m + MK_NCU])) * b.nl[i + nga];",
              "      mk_s += (0.5 * (b.gf1[(size_t)(m + 2*MK_NCU)*nga + i] + mk_f2[m + 2*MK_NCU])) * b.nl[i + 2*nga];",
              "      mk_fh[m] = mk_s + b.tau[0] * (b.gug[i + (size_t)nga*m] - mk_u[m]); }",
              "#pragma unroll",
              "    for (int m = 0; m < MK_NCU; m++) sv[(fq*MK_NCU + m)*MK_NGF + g] = mk_fh[m] * b.jac[i];", "    }"]
        o += _tail()
        code.append("\n".join(o))
        shmf = "std::max((size_t)(64 / MK_NGF) * MK_NCU * MK_NGF, (size_t)(64 / MK_NGF) * MK_NPF * %d) * sizeof(dstype)" % CH
    gemm = opt.get("interp", "fused") == "gemm"
    def interp_launch(kind, ncol):
        if conn and kind == "int":
            return ["    hipLaunchKernelGGL(MKF_%s_interpc_int, dim3((a.nf + %d) / %d), dim3(%d), 0, mk_stream(), a);" % (sname, sfb - 1, sfb, tpb)]
        if staged and kind == "int":
            return ["    hipLaunchKernelGGL(MKF_%s_interps_int, dim3((a.nf + 7) / 8, ((%s) + 31) / 32), dim3(256), 0, mk_stream(), a, %s);" % (sname, ncol, ncol)]
        if gemm:   # gather once, then Node2Gauss on the matrix cores: gug[g + ngf*(fl + nf*col)] = sum_q shapfgt[g + ngf*q] gat[q + npf*(fl + nf*col)]
            return ["    hipLaunchKernelGGL(MKF_%s_gather_%s, dim3(((size_t)MK_NPF*a.nf*(%s) + 255) / 256), dim3(256), 0, mk_stream(), a, %s);" % (sname, kind, ncol, ncol),
                    "    mk_gemm_nn(a.gug, a.gt, a.gat, MK_NGF, MK_NPF, a.nf*(%s), MK_NGF);" % ncol]
        return ["    hipLaunchKernelGGL(MKF_%s_interp_%s, dim3(((size_t)nga*(%s) + 255) / 256), dim3(256), 0, mk_stream(), a, %s);" % (sname, kind, ncol, ncol)]
    L = ["static void mk_face_block(const MkFaceIn& a) {",
         "  const int nga = MK_NGF * a.nf, grid = (a.nf + (%d / MK_NGF) - 1) / (%d / MK_NGF); (void)nga;" % (tp, tp),
         "  if (a.ib == 0) { const int ncol = 2*MK_NC + 2*MK_NCW; (void)ncol;"] + ([
         "    const int gf = (a.nf + (64 / MK_NGF) - 1) / (64 / MK_NGF);",
         "    hipLaunchKernelGGL(%s, dim3(gf), dim3(64), 0, mk_stream(), a);" % nmf1,
         "    hipLaunchKernelGGL(%s, dim3(gf), dim3(64), %s, mk_stream(), a); MK_HIPCHECK(\"%s\"); return; }" % (nmf2, shmf, nmf2)] if fi else
         interp_launch("int", "2*MK_NC + 2*MK_NCW") + [
         "    hipLaunchKernelGGL(%s, dim3((nga + 63) / 64), dim3(64), 0, mk_stream(), a);" % nm1,
         "    hipLaunchKernelGGL(%s, dim3(grid), dim3(%d), %s, mk_stream(), a); MK_HIPCHECK(\"%s\"); return; }" % (nm2, tp, shm, nm2)]) + [
         "  {"] + interp_launch("bou", "MK_NCU + MK_NC + MK_NCW") + ["  }",
         "  switch (a.ib) {"]
    for ib in range(1, NB + 1):
        L.append("  case %d: hipLaunchKernelGGL(MKF_%s_bou%d, dim3(grid), dim3(%d), %s, mk_stream(), a); MK_HIPCHECK(\"bou%d\"); break;" % (ib, sname, ib, tp, shm, ib))
    L += ["  default: printf(\"MKERR no generated boundary kernel for ib=%d\\n\", a.ib); std::abort();", "  }", "}",
          'static const char* MK_FACE_LABEL = "face{split,interp=%s,lb=%d,lbi=%d,waves=%d}";' % ("fused-in-flux" if fi else opt.get("interp", "fused"), lb, lbi, waves)]
    code.append("\n".join(L))
    return "\n".join(code)
