# Meta-kernel schedules (v0 sweep). Each S(...) becomes one executable. See DESIGN.md for the axes.
from gen import S, gemm, pt, el

MO = "iterative-maxocc"
SCHEDULES = [
    # --- baselines / framework validation ---
    S("ref_staged", [gemm("interp"), pt("source"), pt("time"), pt("flux"), pt("scale"), gemm("integrate")],
      note="production path in the DSL: must match the reference"),
    S("ref_staged_kk", [gemm("interp"), pt("source"), pt("time"), pt("flux"), pt("scale"), gemm("integrate")], backend="kokkos",
      note="same as ref_staged, Kokkos TeamPolicy backend"),
    S("props_split", [gemm("interp"), pt("source"), pt("time"), pt("props", lb=2), pt("fluxp", lb=2), pt("scale"), gemm("integrate")],
      note="staged props split (measured ~1.10x by hand)"),
    # --- light fusion around the physics (GEMMs kept) ---
    S("light_mono", [gemm("interp"), pt("source", "time"), pt("flux", "scale"), gemm("integrate")]),
    S("light_split", [gemm("interp"), pt("source", "time"), pt("props", lb=2), pt("fluxp", "scale", lb=2), gemm("integrate")]),
    S("light_split_mo", [gemm("interp"), pt("source", "time"), pt("props", lb=2), pt("fluxp", "scale", lb=2), gemm("integrate")], sched=MO),
    S("light_split3", [gemm("interp"), pt("props", lb=2), pt("source", "time", "fluxp", "scale", lb=2, launder=True), gemm("integrate")],
      note="source+time folded into the flux kernel"),
    # --- lane interpolation feeding the physics (no gather, no interp GEMM, no u round trip) ---
    S("lane_split", [pt("interp", "source", "time", "props", lb=2, launder=True, hand={"u": "glb"}), pt("fluxp", "scale", lb=2), gemm("integrate")],
      note="interp in the props kernel; u still written once for the flux kernel"),
    # --- full pointwise fusion (all but integrate) ---
    S("pt_all_reg", [pt("interp", "source", "time", "props", "fluxp", "scale"), gemm("integrate")]),
    S("pt_all_reg_kk", [pt("interp", "source", "time", "props", "fluxp", "scale"), gemm("integrate")], backend="kokkos"),
    S("light_split_kk", [gemm("interp"), pt("source", "time"), pt("props", lb=2), pt("fluxp", "scale", lb=2), gemm("integrate")], backend="kokkos"),
    S("pt_all_phased", [pt("interp", "source", "time", "props", "fluxp", "scale", lb=2, phased=True), gemm("integrate")],
      note="stages as branches of a nounroll loop, hand-offs through L2 buffers"),
    # --- element maps, integrate in LDS ---
    S("el1_all", [el(1, "interp", "source", "time", "flux", "scale", "integrate")], note="one element per wave (27/64 lanes)"),
    S("el2_all", [el(2, "interp", "source", "time", "flux", "scale", "integrate")], note="two elements per wave (54/64)"),
    S("el7w3_all", [el(7, "interp", "source", "time", "flux", "scale", "integrate", waves=3)], note="7 elements on 3 waves (189/192)"),
    S("el2_split", [gemm("interp"), pt("source", "time"), pt("props", lb=2), el(2, "fluxp", "scale", "integrate", lb=2)],
      note="integrate folded into the flux kernel"),
]
