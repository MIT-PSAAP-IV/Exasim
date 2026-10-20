// [nonmatching] Phase 13 T0/T1 baseline emitter.
//
// This translation unit intentionally compiles only Phase-12 APIs.  A reviewer
// can copy it beside nonmatching_3d.cpp in a clean 6d2aba5a worktree and build
// it there.  The embedded accepted test prints deterministic setup hashes and
// all same-family operator diagnostics without depending on any Phase-13 type
// or field.

#define main phase13_embedded_nonmatching_3d_main
#include "nonmatching_3d.cpp"
#undef main

int main(int, char**)
{
    char name[] = "unit_nonmatching_fingerprint";
    char operators[] = "--operators";
    char rebuild[] = "--rebuild";
    char* arguments[] = {name, operators, rebuild, nullptr};
    std::printf("PHASE13 BASELINE EMIT: Phase-12 same-family operators and setup hashes\n");
    return phase13_embedded_nonmatching_3d_main(3, arguments);
}
