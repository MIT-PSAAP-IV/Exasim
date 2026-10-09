"""q-path generator (GetQ replacement minus RqFace), plain HIP.

One team per element (64 lanes), launched per production element block:
  ug[g,k]  = sum_p shapegt[g + nge p] u[p,k]                                   (Node2Gauss, ncu comps)
  R[p,c]   = sum_{j<nd} sum_{g<nge} shapegw[p + npe g + npe nge (j+1)] (ug[g,k] Xx[g,m,j])   c = k + ncu m  (ApplyXx3 + Gauss2Node)
  R[p,c]  -= side-1 Rh / += side-2 Rh, contributions in (node, face point, side) order     (PutFaceNodesGather)
  q[i,c]   = 0 + sum_k Minv_e[i + npe k] R[k,c]                                  (ArrayGemmBatch1, curved mesh)
  udg[i + npe (ncu + c) + npe nc e] = q[i,c]                                     (ArrayInsert)
The face term Rh (RqFace) is computed before, by Exasim or a generated kernel.
"""
def emit_q(sname, opt):
    lb, v = opt.get("lb", 1), opt.get("v", 2)
    if v == 3: return emit_q_mfma(sname, opt)
    if v == 4: return emit_q_mfma(sname, opt, mfma_interp=True)    # v3 + u -> Gauss on the matrix cores
    if v == 5: return emit_q_mfma(sname, opt, mfma_interp=True, staged_gather=True)   # v4 + coalesced LDS-staged face gather
    if v == 10: return emit_q_v10(sname, opt)                                           # v6 + face term as extra MFMA k-steps
    if v == 9: return emit_q_v9(sname, opt)                                             # pass 1: RqFace + v6; pass 2: v8
    if v == 8: return emit_q_v8(sname, opt)                                             # v7 with all faces in parallel
    if v == 7: return emit_q_v7(sname, opt)                                             # v6 + face integrals in-kernel (no RqFace)
    if v == 6: return emit_q_v6(sname, opt)                                             # v4 with R in the MFMA accumulators
    nm = "MKQ_%s_elem" % sname
    o = ["// ---- generated q path (%s) ----" % opt, "#define MK_HAS_Q 1",
         'extern "C" __global__ void __attribute__((amdgpu_flat_work_group_size(1, 64), amdgpu_waves_per_eu(%d)))' % lb,
         "%s(const MkQIn a) {" % nm,
         "    __shared__ dstype ug[MK_NGE * MK_NCU], xs[MK_NGE * MK_ND * MK_ND], R[MK_NPE * MK_NCQ];",
         "    const int lane = threadIdx.x, el = blockIdx.x; const size_t e = (size_t)a.e1 + el; const int nga = MK_NGE * a.neb;",
         "    const MkQIn& b = a;",
         "    for (int t = lane; t < MK_NGE * MK_NCU; t += 64) { const int g = t % MK_NGE, k = t / MK_NGE; dstype s = 0;",
         "#pragma unroll",
         "        for (int p = 0; p < MK_NPE; p++) s += b.gt[g + MK_NGE*p] * b.udg[p + MK_NPE*k + (size_t)MK_NPE*MK_NC*e];",
         "        ug[t] = s; }"]
    if v == 1:
        o += ["    __syncthreads();",
              "    for (int t = lane; t < MK_NPE * MK_NCQ; t += 64) { const int p = t % MK_NPE, c = t / MK_NPE, k = c % MK_NCU, m = c / MK_NCU; dstype s = 0;",
              "        for (int j = 0; j < MK_ND; j++) {",
              "#pragma unroll",
              "            for (int g = 0; g < MK_NGE; g++)",
              "                s += b.gw[p + MK_NPE*g + MK_NPE*MK_NGE*(j + 1)] * (ug[g + MK_NGE*k] * b.Xx[g + MK_NGE*el + (size_t)nga*m + (size_t)nga*MK_ND*j]); }",
              "        const int node = p + MK_NPE * (int)e;",
              "        for (int r = b.nodeptr[node]; r < b.nodeptr[node + 1]; r++) { const int ci = b.contrib[r], ii = ci >> 1;",
              "            const dstype v = b.Rh[(ii % MK_NPF) + MK_NPF*(c + MK_NCQ*(ii / MK_NPF))]; if (ci & 1) s += v; else s -= v; }",
              "        R[t] = s; }",
              "    __syncthreads();",
              "    for (int t = lane; t < MK_NPE * MK_NCQ; t += 64) { const int i = t % MK_NPE, c = t / MK_NPE; dstype s = 0;",
              "#pragma unroll",
              "        for (int k = 0; k < MK_NPE; k++) s += b.Minv[i + MK_NPE*k + (size_t)MK_NPE*MK_NPE*e] * R[k + MK_NPE*c];",
              "        b.udg[i + MK_NPE*(MK_NCU + c) + (size_t)MK_NPE*MK_NC*e] = s; }"]
    else:
        # same per-output summation order; operands staged so each global load feeds several outputs
        o += ["    for (int t = lane; t < MK_NGE * MK_ND * MK_ND; t += 64) { const int g = t % MK_NGE, mj = t / MK_NGE;   // mj = m + nd j",
              "        xs[t] = b.Xx[g + MK_NGE*el + (size_t)nga*mj]; }",
              "    __syncthreads();",
              "    for (int t = lane; t < MK_NPE * MK_NCU; t += 64) { const int p = t % MK_NPE, k = t / MK_NPE; dstype s[MK_ND];",
              "#pragma unroll",
              "        for (int m = 0; m < MK_ND; m++) s[m] = 0;",
              "        for (int j = 0; j < MK_ND; j++) {",
              "#pragma unroll 9",
              "            for (int g = 0; g < MK_NGE; g++) { const dstype w = b.gw[p + MK_NPE*g + MK_NPE*MK_NGE*(j + 1)]; const dstype u = ug[g + MK_NGE*k];",
              "#pragma unroll",
              "                for (int m = 0; m < MK_ND; m++) s[m] += w * (u * xs[g + MK_NGE*(m + MK_ND*j)]); } }",
              "        const int node = p + MK_NPE * (int)e;",
              "#pragma unroll",
              "        for (int m = 0; m < MK_ND; m++) { const int c = k + MK_NCU*m; dstype sm = s[m];",
              "            for (int r = b.nodeptr[node]; r < b.nodeptr[node + 1]; r++) { const int ci = b.contrib[r], ii = ci >> 1;",
              "                const dstype v = b.Rh[(ii % MK_NPF) + MK_NPF*(c + MK_NCQ*(ii / MK_NPF))]; if (ci & 1) sm += v; else sm -= v; }",
              "            if (!b.curved) sm = sm/(b.Minv[MK_NPE*MK_NPE + e]);   // ApplyJacInv: jac of element e at Gauss point 0",
              "            R[p + MK_NPE*c] = sm; } }",
              "    __syncthreads();",
              "    for (int t = lane; t < MK_NPE * 3; t += 64) { const int i = t % MK_NPE, cg = t / MK_NPE; dstype s[MK_NCQ / 3];",
              "#pragma unroll",
              "        for (int cc = 0; cc < MK_NCQ / 3; cc++) s[cc] = 0;",
              "        const size_t mo = b.curved ? (size_t)MK_NPE*MK_NPE*e : 0;   // per-element inverse, or the master one",
              "        for (int k = 0; k < MK_NPE; k++) { const dstype mv = b.Minv[i + MK_NPE*k + mo];",
              "#pragma unroll",
              "            for (int cc = 0; cc < MK_NCQ / 3; cc++) s[cc] += mv * R[k + MK_NPE*(cg*(MK_NCQ/3) + cc)]; }",
              "#pragma unroll",
              "        for (int cc = 0; cc < MK_NCQ / 3; cc++) b.udg[i + MK_NPE*(MK_NCU + cg*(MK_NCQ/3) + cc) + (size_t)MK_NPE*MK_NC*e] = s[cc]; }"]
    o += ["}",
         "static void mk_q_block(const MkQIn& a) { hipLaunchKernelGGL(%s, dim3(a.neb), dim3(64), 0, mk_stream(), a); MK_HIPCHECK(\"MKQ\"); }" % nm,
         'static const char* MK_Q_LABEL = "q{elem+gather+Minv+insert,v%d,lb=%d}";' % (v, lb)]
    return "\n".join(o)


def emit_q_mfma(sname, opt, mfma_interp=False, staged_gather=False):
    nogather = opt.get("nogather", False)
    """v3: both per-element contractions on the fp64 matrix cores (v_mfma_f64_16x16x4f64), one wave per element.
    R (27x27) = Wg (27 x 81) * S (81 x 27), K index kk = g + nge*j (Gauss2Node's), S[kk][c] = ug[g,k] * Xx[g,m,j] (ApplyXx3's
    rounding), K accumulated in chunks of 4 in ascending order as production's MFMA GEMM; then the ordered face gather and
    (straight mesh) the jac0 division in LDS; then q (27x27) = Minv (27x27) * R, K = 27 in chunks of 4.
    MFMA f64 16x16x4 lane layout (measured on gfx942): A[row = l%16][k = l/16], B[k = l/16][col = l%16], D[row = 4*r + l/16][col = l%16]."""
    lb = opt.get("lb", 1)
    nm = "MKQ_%s_elem" % sname
    o = ["// ---- generated q path (%s), matrix cores ----" % opt, "#define MK_HAS_Q 1"] + (["#define MK_Q_NOGATHER 1"] if nogather else []) + [
         "typedef double mk_d4 __attribute__((ext_vector_type(4)));",
         'extern "C" __global__ void __attribute__((amdgpu_flat_work_group_size(1, 64), amdgpu_waves_per_eu(%d)))' % lb,
         "%s(const MkQIn a) {" % nm,
         "    __shared__ dstype ug[MK_NGE * MK_NCU], R[MK_NPE * MK_NCQ];",
         "    const int l = threadIdx.x, el = blockIdx.x; const size_t e = (size_t)a.e1 + el; const int nga = MK_NGE * a.neb;",
         "    const int lr = l % 16, lk = l / 16;   // MFMA lane coordinates",
         "    const MkQIn& b = a;",
         ] + ([
         "    {   // u -> Gauss on the matrix cores: ug(g, k) = sum_p gt[g + nge p] u(p, k); a = u column k = lr, b = gt row g (production's Node2Gauss chain)",
         "        dstype fr[7];",
         "#pragma unroll",
         "        for (int st = 0; st < 7; st++) { const int p = 4*st + lk; fr[st] = (lr < MK_NCU && p < MK_NPE) ? b.udg[p + MK_NPE*lr + (size_t)MK_NPE*MK_NC*e] : 0.0; }",
         "#pragma unroll",
         "        for (int m0 = 0; m0 < 32; m0 += 16) { mk_d4 acc = (mk_d4){0.0, 0.0, 0.0, 0.0}; const int g = m0 + lr;",
         "#pragma unroll",
         "            for (int st = 0; st < 7; st++) { const int p = 4*st + lk;",
         "                acc = __builtin_amdgcn_mfma_f64_16x16x4f64(fr[st], (g < MK_NGE && p < MK_NPE) ? b.gt[g + MK_NGE*p] : 0.0, acc, 0, 0, 0); }",
         "            if (g < MK_NGE) {",
         "#pragma unroll",
         "                for (int r = 0; r < 4; r++) { const int k = 4*r + lk; if (k < MK_NCU) ug[g + MK_NGE*k] = acc[r]; } } }",
         "    }"] if mfma_interp else [
         "    for (int t = l; t < MK_NGE * MK_NCU; t += 64) { const int g = t % MK_NGE, k = t / MK_NGE; dstype s = 0;",
         "#pragma unroll",
         "        for (int p = 0; p < MK_NPE; p++) s += b.gt[g + MK_NGE*p] * b.udg[p + MK_NPE*k + (size_t)MK_NPE*MK_NC*e];",
         "        ug[t] = s; }"]) + [ 
         "    __syncthreads();",
         "    // R = Wg * S: output tiles (rt, ct) in {0,1}^2 over p (rows) and c (cols), K = nge*nd = 81 padded to 84",
         "    mk_d4 acc[2][2];",
         "#pragma unroll",
         "    for (int rt = 0; rt < 2; rt++)",
         "#pragma unroll",
         "        for (int ct = 0; ct < 2; ct++) acc[rt][ct] = (mk_d4){0.0, 0.0, 0.0, 0.0};",
         "    for (int ks = 0; ks < (MK_NGE * MK_ND + 3) / 4; ks++) {",
         "        const int kk = 4*ks + lk; const bool kin = kk < MK_NGE * MK_ND; const int g = kk % MK_NGE, j = kk / MK_NGE;",
         "        dstype av[2], bv[2];",
         "#pragma unroll",
         "        for (int rt = 0; rt < 2; rt++) { const int p = 16*rt + lr;",
         "            av[rt] = (kin && p < MK_NPE) ? b.gw[p + MK_NPE*g + MK_NPE*MK_NGE*(j + 1)] : 0.0; }",
         "#pragma unroll",
         "        for (int ct = 0; ct < 2; ct++) { const int c = 16*ct + lr, k = c % MK_NCU, m = c / MK_NCU;",
         "            bv[ct] = (kin && c < MK_NCQ) ? ug[g + MK_NGE*k] * b.Xx[g + MK_NGE*el + (size_t)nga*m + (size_t)nga*MK_ND*j] : 0.0; }",
         "#pragma unroll",
         "        for (int rt = 0; rt < 2; rt++)",
         "#pragma unroll",
         "            for (int ct = 0; ct < 2; ct++) acc[rt][ct] = __builtin_amdgcn_mfma_f64_16x16x4f64(av[rt], bv[ct], acc[rt][ct], 0, 0, 0);",
         "    }",
         "#pragma unroll",
         "    for (int rt = 0; rt < 2; rt++)",
         "#pragma unroll",
         "        for (int ct = 0; ct < 2; ct++)",
         "#pragma unroll",
         "            for (int r = 0; r < 4; r++) { const int p = 16*rt + 4*r + lk, c = 16*ct + lr;",
         "                if (p < MK_NPE && c < MK_NCQ) R[p + MK_NPE*c] = acc[rt][ct][r]; }",
         "    __syncthreads();",
         ] + ([
         "    // ordered face gather, staged: the element's <= 6 face blocks of Rh are contiguous (npf*ncq values each), so they are",
         "    // loaded coalesced into LDS ncu components at a time and the same ordered sums run against LDS (bitwise identical)",
         "    __shared__ dstype FR[6 * MK_NPF * 9];",
         "    const int ef0 = b.efaces[6*e], ef1 = b.efaces[6*e+1], ef2 = b.efaces[6*e+2], ef3 = b.efaces[6*e+3], ef4 = b.efaces[6*e+4], ef5 = b.efaces[6*e+5];",
         "    for (int c0 = 0; c0 < MK_NCQ; c0 += 9) {",
         "        for (int t = l; t < 6 * MK_NPF * 9; t += 64) { const int sl = t / (MK_NPF*9), rr = t % (MK_NPF*9);",
         "            const int f = sl == 0 ? ef0 : sl == 1 ? ef1 : sl == 2 ? ef2 : sl == 3 ? ef3 : sl == 4 ? ef4 : ef5;",
         "            FR[t] = f >= 0 ? b.Rh[rr + (size_t)MK_NPF*(c0 + MK_NCQ*(size_t)f)] : 0.0; }",
         "        __syncthreads();",
         "        for (int t = l; t < MK_NPE * 9; t += 64) { const int p = t % MK_NPE, cc = t / MK_NPE, c = c0 + cc; dstype sm = R[p + MK_NPE*c];",
         "            const int node = p + MK_NPE * (int)e;",
         "            for (int r = b.nodeptr[node]; r < b.nodeptr[node + 1]; r++) { const int lc = b.lcontrib[r], li = lc >> 1;",
         "                const dstype v = FR[(li / MK_NPF) * (MK_NPF*9) + cc*MK_NPF + (li % MK_NPF)]; if (lc & 1) sm += v; else sm -= v; }",
         "            if (!b.curved) sm = sm/(b.Minv[MK_NPE*MK_NPE + e]);",
         "            R[p + MK_NPE*c] = sm; }",
         "        __syncthreads();",
         "    }"] if staged_gather else [         "    // ordered face gather (production order), then the straight-mesh jac0 division",
         "    for (int t = l; t < MK_NPE * MK_NCQ; t += 64) { const int p = t % MK_NPE, c = t / MK_NPE; dstype sm = R[t];",
         "        const int node = p + MK_NPE * (int)e;",
         "#ifndef MK_Q_NOGATHER   // timing ablation only: skips the face contributions (wrong result)",
         "        for (int r = b.nodeptr[node]; r < b.nodeptr[node + 1]; r++) { const int ci = b.contrib[r], ii = ci >> 1;",
         "            const dstype v = b.Rh[(ii % MK_NPF) + MK_NPF*(c + MK_NCQ*(ii / MK_NPF))]; if (ci & 1) sm += v; else sm -= v; }",
         "#endif",
         "        if (!b.curved) sm = sm/(b.Minv[MK_NPE*MK_NPE + e]);",
         "        R[t] = sm; }"]) + [
         "    __syncthreads();",
         "    // q = Minv * R: tiles over i (rows) and c (cols), K = npe = 27 padded to 28",
         "    const size_t mo = b.curved ? (size_t)MK_NPE*MK_NPE*e : 0;",
         "#pragma unroll",
         "    for (int rt = 0; rt < 2; rt++)",
         "#pragma unroll",
         "        for (int ct = 0; ct < 2; ct++) acc[rt][ct] = (mk_d4){0.0, 0.0, 0.0, 0.0};",
         "#pragma unroll",
         "    for (int ks = 0; ks < (MK_NPE + 3) / 4; ks++) {",
         "        const int k = 4*ks + lk; const bool kin = k < MK_NPE;",
         "        dstype av[2], bv[2];",
         "#pragma unroll",
         "        for (int rt = 0; rt < 2; rt++) { const int i = 16*rt + lr; av[rt] = (kin && i < MK_NPE) ? b.Minv[i + MK_NPE*k + mo] : 0.0; }",
         "#pragma unroll",
         "        for (int ct = 0; ct < 2; ct++) { const int c = 16*ct + lr; bv[ct] = (kin && c < MK_NCQ) ? R[k + MK_NPE*c] : 0.0; }",
         "#pragma unroll",
         "        for (int rt = 0; rt < 2; rt++)",
         "#pragma unroll",
         "            for (int ct = 0; ct < 2; ct++) acc[rt][ct] = __builtin_amdgcn_mfma_f64_16x16x4f64(av[rt], bv[ct], acc[rt][ct], 0, 0, 0);",
         "    }",
         "#pragma unroll",
         "    for (int rt = 0; rt < 2; rt++)",
         "#pragma unroll",
         "        for (int ct = 0; ct < 2; ct++)",
         "#pragma unroll",
         "            for (int r = 0; r < 4; r++) { const int i = 16*rt + 4*r + lk, c = 16*ct + lr;",
         "                if (i < MK_NPE && c < MK_NCQ) b.udg[i + MK_NPE*(MK_NCU + c) + (size_t)MK_NPE*MK_NC*e] = acc[rt][ct][r]; }",
         "}",
         "static void mk_q_block(const MkQIn& a) { hipLaunchKernelGGL(%s, dim3(a.neb), dim3(64), 0, mk_stream(), a); MK_HIPCHECK(\"MKQ\"); }" % nm,
         'static const char* MK_Q_LABEL = "q{mfma,lb=%d}";' % lb]
    return "\n".join(o)


def emit_q_v6(sname, opt):
    """v6 = v4 with R kept in the matrix-core accumulators. The MFMA output layout of R (row p = 16 rt + 4 r + l/16,
    col c = 16 ct + l%16) is exactly the B-operand layout of the next product q = Minv R (k = 4 ks + l/16, col = l%16,
    with ks = 4 rt + r), so each lane applies the ordered face contributions and the jac0 division to its own entries in
    registers and feeds them straight into the second MFMA: no R buffer in LDS, two fewer barriers, same arithmetic."""
    lb = opt.get("lb", 1)
    nm = "MKQ_%s_elem" % sname
    o = ["// ---- generated q path (%s), matrix cores, R in registers ----" % opt, "#define MK_HAS_Q 1",
         "typedef double mk_d4 __attribute__((ext_vector_type(4)));",
         'extern "C" __global__ void __attribute__((amdgpu_flat_work_group_size(1, 64), amdgpu_waves_per_eu(%d)))' % lb,
         "%s(const MkQIn a) {" % nm,
         "    __shared__ dstype ug[MK_NGE * MK_NCU];",
         "    const int l = threadIdx.x, el = blockIdx.x; const size_t e = (size_t)a.e1 + el; const int nga = MK_NGE * a.neb;",
         "    const int lr = l % 16, lk = l / 16;   // MFMA lane coordinates",
         "    const MkQIn& b = a;",
         "    {   // u -> Gauss on the matrix cores (production's Node2Gauss chain)",
         "        dstype fr[7];",
         "#pragma unroll",
         "        for (int st = 0; st < 7; st++) { const int p = 4*st + lk; fr[st] = (lr < MK_NCU && p < MK_NPE) ? b.udg[p + MK_NPE*lr + (size_t)MK_NPE*MK_NC*e] : 0.0; }",
         "#pragma unroll",
         "        for (int m0 = 0; m0 < 32; m0 += 16) { mk_d4 acc = (mk_d4){0.0, 0.0, 0.0, 0.0}; const int g = m0 + lr;",
         "#pragma unroll",
         "            for (int st = 0; st < 7; st++) { const int p = 4*st + lk;",
         "                acc = __builtin_amdgcn_mfma_f64_16x16x4f64(fr[st], (g < MK_NGE && p < MK_NPE) ? b.gt[g + MK_NGE*p] : 0.0, acc, 0, 0, 0); }",
         "            if (g < MK_NGE) {",
         "#pragma unroll",
         "                for (int r = 0; r < 4; r++) { const int k = 4*r + lk; if (k < MK_NCU) ug[g + MK_NGE*k] = acc[r]; } } }",
         "    }",
         "    __syncthreads();",
         "    // R = Wg * S on the matrix cores, K = nge*nd = 81 in chunks of 4 (as v3)",
         "    mk_d4 R[2][2];",
         "#pragma unroll",
         "    for (int rt = 0; rt < 2; rt++)",
         "#pragma unroll",
         "        for (int ct = 0; ct < 2; ct++) R[rt][ct] = (mk_d4){0.0, 0.0, 0.0, 0.0};",
         "    for (int ks = 0; ks < (MK_NGE * MK_ND + 3) / 4; ks++) {",
         "        const int kk = 4*ks + lk; const bool kin = kk < MK_NGE * MK_ND; const int g = kk % MK_NGE, j = kk / MK_NGE;",
         "        dstype av[2], bv[2];",
         "#pragma unroll",
         "        for (int rt = 0; rt < 2; rt++) { const int p = 16*rt + lr;",
         "            av[rt] = (kin && p < MK_NPE) ? b.gw[p + MK_NPE*g + MK_NPE*MK_NGE*(j + 1)] : 0.0; }",
         "#pragma unroll",
         "        for (int ct = 0; ct < 2; ct++) { const int c = 16*ct + lr, k = c % MK_NCU, m = c / MK_NCU;",
         "            bv[ct] = (kin && c < MK_NCQ) ? ug[g + MK_NGE*k] * b.Xx[g + MK_NGE*el + (size_t)nga*m + (size_t)nga*MK_ND*j] : 0.0; }",
         "#pragma unroll",
         "        for (int rt = 0; rt < 2; rt++)",
         "#pragma unroll",
         "            for (int ct = 0; ct < 2; ct++) R[rt][ct] = __builtin_amdgcn_mfma_f64_16x16x4f64(av[rt], bv[ct], R[rt][ct], 0, 0, 0);",
         "    }",
         "    // ordered face gather + straight-mesh jac0 division, each lane on its own R entries (same per-entry order)",
         "    const dstype jinv = b.Minv[MK_NPE*MK_NPE + e];",
         "#pragma unroll",
         "    for (int rt = 0; rt < 2; rt++)",
         "#pragma unroll",
         "        for (int ct = 0; ct < 2; ct++)",
         "#pragma unroll",
         "            for (int r = 0; r < 4; r++) { const int p = 16*rt + 4*r + lk, c = 16*ct + lr;",
         "                if (p < MK_NPE && c < MK_NCQ) { dstype sm = R[rt][ct][r]; const int node = p + MK_NPE * (int)e;",
         "                    for (int q = b.nodeptr[node]; q < b.nodeptr[node + 1]; q++) { const int ci = b.contrib[q], ii = ci >> 1;",
         "                        const dstype v = b.Rh[(ii % MK_NPF) + MK_NPF*(c + MK_NCQ*(ii / MK_NPF))]; if (ci & 1) sm += v; else sm -= v; }",
         "                    if (!b.curved) sm = sm/jinv;",
         "                    R[rt][ct][r] = sm; } }",
         "    // q = Minv * R: the B operand of k-step ks is this lane's R[ks/4][ct][ks%4] (k = 4 ks + l/16)",
         "    const size_t mo = b.curved ? (size_t)MK_NPE*MK_NPE*e : 0;",
         "    mk_d4 acc[2][2];",
         "#pragma unroll",
         "    for (int rt = 0; rt < 2; rt++)",
         "#pragma unroll",
         "        for (int ct = 0; ct < 2; ct++) acc[rt][ct] = (mk_d4){0.0, 0.0, 0.0, 0.0};",
         "#pragma unroll",
         "    for (int ks = 0; ks < (MK_NPE + 3) / 4; ks++) {",
         "        const int k = 4*ks + lk; const bool kin = k < MK_NPE;",
         "        dstype av[2], bv[2];",
         "#pragma unroll",
         "        for (int rt = 0; rt < 2; rt++) { const int i = 16*rt + lr; av[rt] = (kin && i < MK_NPE) ? b.Minv[i + MK_NPE*k + mo] : 0.0; }",
         "#pragma unroll",
         "        for (int ct = 0; ct < 2; ct++) { const int c = 16*ct + lr; bv[ct] = (kin && c < MK_NCQ) ? R[ks / 4][ct][ks % 4] : 0.0; }",
         "#pragma unroll",
         "        for (int rt = 0; rt < 2; rt++)",
         "#pragma unroll",
         "            for (int ct = 0; ct < 2; ct++) acc[rt][ct] = __builtin_amdgcn_mfma_f64_16x16x4f64(av[rt], bv[ct], acc[rt][ct], 0, 0, 0);",
         "    }",
         "#pragma unroll",
         "    for (int rt = 0; rt < 2; rt++)",
         "#pragma unroll",
         "        for (int ct = 0; ct < 2; ct++)",
         "#pragma unroll",
         "            for (int r = 0; r < 4; r++) { const int i = 16*rt + 4*r + lk, c = 16*ct + lr;",
         "                if (i < MK_NPE && c < MK_NCQ) b.udg[i + MK_NPE*(MK_NCU + c) + (size_t)MK_NPE*MK_NC*e] = acc[rt][ct][r]; }",
         "}",
         "static void mk_q_block(const MkQIn& a) { hipLaunchKernelGGL(%s, dim3(a.neb), dim3(64), 0, mk_stream(), a); MK_HIPCHECK(\"MKQ\"); }" % nm,
         'static const char* MK_Q_LABEL = "q{mfma,R in registers,lb=%d}";' % lb]
    return "\n".join(o)


def emit_q_v7(sname, opt):
    """v7 = v6 with the face integrals computed in the element kernel instead of read from Rh (no RqFace kernel). All the
    element's faces' uh (81 contiguous values each), normals and jacobians are staged into LDS at kernel start, overlapping
    the element loads and the volume MFMAs. Then for each face in ASCENDING face id: interpolate uh to Gauss points and
    integrate back with the n_d jac scaling applied per term -- RqFace's expressions in RqFace's order -- and every lane adds
    the face's contributions to its own R entries. A node's contributions are sorted by face-node index with distinct faces
    (checked at setup), so face-by-face accumulation in ascending id is production's gather order: bitwise identical.
    Interior faces are computed twice (once per side), but Rh's write + scattered re-read and the RqFace launches go."""
    src = emit_q_v6(sname, opt)
    old_sh = "    __shared__ dstype ug[MK_NGE * MK_NCU];"
    new_sh = "\n".join([
        "    __shared__ dstype ug[MK_NGE * MK_NCU];",
        "    __shared__ dstype fu[6 * MK_NPF * MK_NCU], fgeo[6 * MK_NGF * (MK_ND + 1)], fg[MK_NCU * MK_NGF], fh[MK_NPF * MK_NCQ];",
        "    __shared__ int fcode[6 * MK_NPE], fid[6];"])
    assert old_sh in src; src = src.replace(old_sh, new_sh, 1)
    # prefetch at kernel start (after the lane coordinates)
    anchor = "    const MkQIn& b = a;\n"
    pre = "\n".join([
        "    const MkQIn& b = a;",
        "    if (l < 6) fid[l] = b.qfs[6 * e + l];",
        "    for (int t = l; t < 6 * MK_NPE; t += 64) fcode[t] = b.qcode[6 * MK_NPE * e + t];",
        "    for (int t = l; t < 6 * MK_NPF * MK_NCU; t += 64) { const int s = t / (MK_NPF * MK_NCU), f = b.qfs[6 * e + s];",
        "        fu[t] = f >= 0 ? b.uh[(t % (MK_NPF * MK_NCU)) + (size_t)MK_NPF * MK_NCU * f] : 0.0; }",
        "    for (int t = l; t < 6 * MK_NGF * (MK_ND + 1); t += 64) { const int s = t / (MK_NGF * (MK_ND + 1)), r = t % (MK_NGF * (MK_ND + 1)), g = r % MK_NGF, d = r / MK_NGF;",
        "        const int f = b.qfs[6 * e + s];   // d < nd: normal component d; d == nd: jacobian",
        "        fgeo[t] = f >= 0 ? b.faceg[b.fofs[f] + g + (size_t)b.fnga[f] * (MK_NCX + d)] : 0.0; }",
        ""])
    assert anchor in src; src = src.replace(anchor, pre, 1)
    i0 = src.index("    // ordered face gather + straight-mesh jac0 division")
    i1 = src.index("    // q = Minv * R:")
    gather = "\n".join([
        "    // face integrals in-kernel, faces in ascending id (= production's per-node contribution order)",
        "    const dstype jinv = b.Minv[MK_NPE*MK_NPE + e];",
        "#pragma unroll 1",
        "    for (int s = 0; s < 6; s++) {",
        "        if (fid[s] < 0) break;   // uniform across the workgroup (fid written before the first barrier)",
        "        const dstype* u = fu + s * MK_NPF * MK_NCU; const dstype* gm = fgeo + s * MK_NGF * (MK_ND + 1);",
        "        for (int t = l; t < MK_NCU * MK_NGF; t += 64) { const int g = t % MK_NGF, m = t / MK_NGF; dstype v = 0;",
        "#pragma unroll",
        "            for (int q = 0; q < MK_NPF; q++) v += b.fgt[g + MK_NGF*q] * u[q + MK_NPF*m];",
        "            fg[t] = v; }",
        "        __syncthreads();",
        "        for (int t = l; t < MK_NPF * MK_NCQ; t += 64) { const int q = t % MK_NPF, c = t / MK_NPF, m = c % MK_NCU, d = c / MK_NCU; dstype sm = 0;",
        "#pragma unroll",
        "            for (int gg = 0; gg < MK_NGF; gg++) { const dstype sv = (fg[gg + MK_NGF*m] * gm[gg + MK_NGF*d]) * gm[gg + MK_NGF*MK_ND];",
        "                sm += b.fgw[q + MK_NPF*gg] * sv; }",
        "            fh[t] = sm; }",
        "        __syncthreads();",
        "#pragma unroll",
        "        for (int rt = 0; rt < 2; rt++)",
        "#pragma unroll",
        "            for (int ct = 0; ct < 2; ct++)",
        "#pragma unroll",
        "                for (int r = 0; r < 4; r++) { const int p = 16*rt + 4*r + lk, c = 16*ct + lr;",
        "                    if (p < MK_NPE && c < MK_NCQ) { const int cd = fcode[MK_NPE*s + p];",
        "                        if (cd >= 0) { const dstype v = fh[(cd >> 1) + MK_NPF*c]; if (cd & 1) R[rt][ct][r] += v; else R[rt][ct][r] -= v; } } }",
        "        // no barrier: the next face writes fg (last read before the barrier above) and writes fh only after its own barrier",
        "    }",
        "    if (!b.curved) {",
        "#pragma unroll",
        "        for (int rt = 0; rt < 2; rt++)",
        "#pragma unroll",
        "            for (int ct = 0; ct < 2; ct++)",
        "#pragma unroll",
        "                for (int r = 0; r < 4; r++) R[rt][ct][r] = R[rt][ct][r]/jinv; }",
        ""])
    src = src[:i0] + gather + src[i1:]
    src = src.replace("#define MK_HAS_Q 1", "#define MK_HAS_Q 1\n#define MK_Q_FACES 1", 1)
    src = src.replace('"q{mfma,R in registers,lb=', '"q{mfma,R in registers,faces in-kernel,lb=')
    return src


def emit_q_v8(sname, opt):
    """v8 = v7 with the six faces processed IN PARALLEL in RqFace's own lane layout: lane = (face slot, Gauss point) for
    the interpolation and scaling, lane = (face slot, face node) for the integration, so 54 of 64 lanes work in every
    phase and there are 4 barriers per element instead of 2 per face. uh / n / jac are prefetched into registers before
    the volume MFMAs. One LDS buffer (6 x 27 x 9) holds uh, then the scaled Gauss values, then the face integrals. Same
    expressions in the same order as RqFace, contributions added in ascending face id: bitwise identical."""
    src = emit_q_v6(sname, opt)
    old_sh = "    __shared__ dstype ug[MK_NGE * MK_NCU];"
    new_sh = "\n".join([
        "    constexpr int MK_FB = 6 * MK_NCQ * MK_NGF;   // >= 6 npf ncu (uh) and 6 npf ncq (integrals); also holds ug",
        "    __shared__ dstype mk_fb[MK_FB > MK_NGE * MK_NCU ? MK_FB : MK_NGE * MK_NCU]; dstype* ug = mk_fb;",
        "    __shared__ int fcode[6 * MK_NPE];"])
    assert old_sh in src; src = src.replace(old_sh, new_sh, 1)
    anchor = "    const MkQIn& b = a;\n"
    pre = "\n".join([
        "    const MkQIn& b = a;",
        "    for (int t = l; t < 6 * MK_NPE; t += 64) fcode[t] = b.qcode[6 * MK_NPE * e + t];",
        "    constexpr int MK_NU = (6 * MK_NPF * MK_NCU + 63) / 64;",
        "    dstype mk_pu[MK_NU];   // this lane's share of the six faces' uh, held in registers across the volume part",
        "#pragma unroll",
        "    for (int k = 0; k < MK_NU; k++) { const int t = l + 64 * k, s = t / (MK_NPF * MK_NCU); const int f = (t < 6 * MK_NPF * MK_NCU) ? b.qfs[6 * e + s] : -1;",
        "        mk_pu[k] = f >= 0 ? b.uh[(t % (MK_NPF * MK_NCU)) + (size_t)MK_NPF * MK_NCU * f] : 0.0; }",
        "    const int fs_ = l / MK_NGF, fgp = l % MK_NGF; const int ff = (fs_ < 6) ? b.qfs[6 * e + fs_] : -1;   // lane = (slot, Gauss point)",
        "    dstype mk_geo[MK_ND + 1];",
        "#pragma unroll",
        "    for (int d = 0; d <= MK_ND; d++) mk_geo[d] = ff >= 0 ? b.faceg[b.fofs[ff] + fgp + (size_t)b.fnga[ff] * (MK_NCX + d)] : 0.0;",
        ""])
    assert anchor in src; src = src.replace(anchor, pre, 1)
    i0 = src.index("    // ordered face gather + straight-mesh jac0 division")
    i1 = src.index("    // q = Minv * R:")
    gather = "\n".join([
        "    // ---- face integrals, all faces in parallel ----",
        "    const dstype jinv = b.Minv[MK_NPE*MK_NPE + e];",
        "    __syncthreads();   // everyone is done with ug (the volume part)",
        "#pragma unroll",
        "    for (int k = 0; k < MK_NU; k++) { const int t = l + 64 * k; if (t < 6 * MK_NPF * MK_NCU) mk_fb[t] = mk_pu[k]; }",
        "    __syncthreads();",
        "    dstype mk_sv[MK_NCQ];   // lane (slot, g): RqFace's (ug n_d) jac for every component",
        "    if (ff >= 0) {",
        "#pragma unroll",
        "        for (int m = 0; m < MK_NCU; m++) { dstype ugv = 0;",
        "#pragma unroll",
        "            for (int q = 0; q < MK_NPF; q++) ugv += b.fgt[fgp + MK_NGF*q] * mk_fb[fs_ * MK_NPF * MK_NCU + q + MK_NPF*m];",
        "#pragma unroll",
        "            for (int d = 0; d < MK_ND; d++) mk_sv[m + MK_NCU*d] = (ugv * mk_geo[d]) * mk_geo[MK_ND]; } }",
        "    __syncthreads();   // uh consumed",
        "    if (ff >= 0) {",
        "#pragma unroll",
        "        for (int c = 0; c < MK_NCQ; c++) mk_fb[(fs_ * MK_NCQ + c) * MK_NGF + fgp] = mk_sv[c]; }",
        "    __syncthreads();",
        "    dstype mk_fv[MK_NCQ];   // lane (slot, face node q = fgp): the face integrals for every component",
        "    if (ff >= 0) {",
        "#pragma unroll",
        "        for (int c = 0; c < MK_NCQ; c++) { dstype sm = 0;",
        "#pragma unroll",
        "            for (int gg = 0; gg < MK_NGF; gg++) sm += b.fgw[fgp + MK_NPF*gg] * mk_fb[(fs_ * MK_NCQ + c) * MK_NGF + gg];",
        "            mk_fv[c] = sm; } }",
        "    __syncthreads();   // scaled values consumed",
        "    if (ff >= 0) {",
        "#pragma unroll",
        "        for (int c = 0; c < MK_NCQ; c++) mk_fb[fgp + MK_NPF * (c + MK_NCQ * fs_)] = mk_fv[c]; }",
        "    __syncthreads();",
        "#pragma unroll",
        "    for (int rt = 0; rt < 2; rt++)",
        "#pragma unroll",
        "        for (int ct = 0; ct < 2; ct++)",
        "#pragma unroll",
        "            for (int r = 0; r < 4; r++) { const int p = 16*rt + 4*r + lk, c = 16*ct + lr;",
        "                if (p < MK_NPE && c < MK_NCQ) { dstype sm = R[rt][ct][r];",
        "                    for (int s = 0; s < 6; s++) { const int cd = fcode[MK_NPE*s + p];",
        "                        if (cd >= 0) { const dstype v = mk_fb[(cd >> 1) + MK_NPF * (c + MK_NCQ * s)]; if (cd & 1) sm += v; else sm -= v; } }",
        "                    if (!b.curved) sm = sm/jinv;",
        "                    R[rt][ct][r] = sm; } }",
        ""])
    src = src[:i0] + gather + src[i1:]
    src = src.replace("#define MK_HAS_Q 1", "#define MK_HAS_Q 1\n#define MK_Q_FACES 1", 1)
    src = src.replace('"q{mfma,R in registers,lb=', '"q{mfma,R in registers,faces in-kernel parallel,lb=')
    return src


def emit_q_v9(sname, opt):
    """Hybrid: pass 1 (most elements) keeps RqFace + v6 (the face integrals are computed once per face there); pass 2
    (the few elements next to the ghosts) uses v8, so RqFace's second full-grid launch disappears. Both bitwise."""
    v6 = emit_q_v6(sname, opt)
    v8 = emit_q_v8(sname, opt).replace("MKQ_%s_elem" % sname, "MKQ_%s_elemf" % sname)
    v8 = v8.replace("#define MK_HAS_Q 1\n#define MK_Q_FACES 1", "#define MK_Q_HYBRID 1", 1)
    v8 = v8.replace("typedef double mk_d4 __attribute__((ext_vector_type(4)));\n", "", 1)
    v8 = v8.replace("static void mk_q_block(", "static void mk_qf_block(").replace("static const char* MK_Q_LABEL", "static const char* MK_QF_LABEL")
    return v6 + "\n" + v8


def emit_q_v10(sname, opt):
    """v10 = v6 with the face term folded into the same MFMA chain that builds R (rounding-level, not bitwise): the face
    contribution to element node p is sum over (face slot, Gauss point) of sign * wf(q(p), g) * (ug n_d) jac, i.e. more
    k-steps of R += A B with A = signed face weights (from qcode) and B = the scaled face values, staged two faces at a
    time in LDS. No RqFace kernel and no Rh traffic in either pass."""
    src = emit_q_v6(sname, opt)
    old_sh = "    __shared__ dstype ug[MK_NGE * MK_NCU];"
    new_sh = "    __shared__ dstype ug[MK_NGE * MK_NCU];\n    __shared__ dstype qsv[2 * MK_NGF * MK_NCQ], quh[2 * MK_NPF * MK_NCU];"
    assert old_sh in src; src = src.replace(old_sh, new_sh, 1)
    i0 = src.index("    // ordered face gather + straight-mesh jac0 division")
    i1 = src.index("    // q = Minv * R:")
    body = "\n".join([
        "    // ---- face term: more k-steps of the R chain, two face slots per round ----",
        "    const dstype jinv = b.Minv[MK_NPE*MK_NPE + e];",
        "#pragma unroll 1",
        "    for (int s0 = 0; s0 < 6; s0 += 2) {",
        "        if (b.qfs[6 * e + s0] < 0) break;   // slots are filled in order",
        "        __syncthreads();   // previous round's qsv / quh consumed",
        "        for (int t = l; t < 2 * MK_NPF * MK_NCU; t += 64) { const int f = b.qfs[6 * e + s0 + t / (MK_NPF * MK_NCU)];",
        "            quh[t] = f >= 0 ? b.uh[(t % (MK_NPF * MK_NCU)) + (size_t)MK_NPF * MK_NCU * f] : 0.0; }",
        "        __syncthreads();",
        "        for (int t = l; t < 2 * MK_NGF * MK_NCQ; t += 64) { const int k = t % (2 * MK_NGF), c = t / (2 * MK_NGF);",
        "            const int sl = k / MK_NGF, gg = k % MK_NGF, m = c % MK_NCU, d = c / MK_NCU, f = b.qfs[6 * e + s0 + sl]; dstype v = 0;",
        "            if (f >= 0) { dstype u = 0;",
        "#pragma unroll",
        "                for (int q = 0; q < MK_NPF; q++) u += b.fgt[gg + MK_NGF*q] * quh[sl * MK_NPF * MK_NCU + q + MK_NPF*m];",
        "                const size_t fo = b.fofs[f]; const int fn = b.fnga[f];",
        "                v = (u * b.faceg[fo + gg + (size_t)fn*(MK_NCX + d)]) * b.faceg[fo + gg + (size_t)fn*(MK_NCX + MK_ND)]; }",
        "            qsv[k + 2 * MK_NGF * c] = v; }",
        "        __syncthreads();",
        "#pragma unroll",
        "        for (int ks = 0; ks < (2 * MK_NGF + 3) / 4; ks++) {",
        "            const int kk = 4*ks + lk; const bool kin = kk < 2 * MK_NGF; const int sl = s0 + (kin ? kk / MK_NGF : 0), gg = kk % MK_NGF;",
        "            dstype av[2], bv[2];",
        "#pragma unroll",
        "            for (int rt = 0; rt < 2; rt++) { const int p = 16*rt + lr; const int cd = (kin && p < MK_NPE) ? b.qcode[(size_t)MK_NPE * (6 * e + sl) + p] : -1;",
        "                av[rt] = cd >= 0 ? ((cd & 1) ? b.fgw[(cd >> 1) + MK_NPF*gg] : -b.fgw[(cd >> 1) + MK_NPF*gg]) : 0.0; }",
        "#pragma unroll",
        "            for (int ct = 0; ct < 2; ct++) { const int c = 16*ct + lr; bv[ct] = (kin && c < MK_NCQ) ? qsv[kk + 2 * MK_NGF * c] : 0.0; }",
        "#pragma unroll",
        "            for (int rt = 0; rt < 2; rt++)",
        "#pragma unroll",
        "                for (int ct = 0; ct < 2; ct++) R[rt][ct] = __builtin_amdgcn_mfma_f64_16x16x4f64(av[rt], bv[ct], R[rt][ct], 0, 0, 0);",
        "        }",
        "    }",
        "    if (!b.curved) {",
        "#pragma unroll",
        "        for (int rt = 0; rt < 2; rt++)",
        "#pragma unroll",
        "            for (int ct = 0; ct < 2; ct++)",
        "#pragma unroll",
        "                for (int r = 0; r < 4; r++) R[rt][ct][r] = R[rt][ct][r]/jinv; }",
        ""])
    src = src[:i0] + body + src[i1:]
    src = src.replace("#define MK_HAS_Q 1", "#define MK_HAS_Q 1\n#define MK_Q_FACES 1", 1)
    src = src.replace('"q{mfma,R in registers,lb=', '"q{mfma,R in registers,face term in the MFMA chain,lb=')
    return src
