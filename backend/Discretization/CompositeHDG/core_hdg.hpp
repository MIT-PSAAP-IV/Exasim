// Included inside fieldsplit after Kernels<ND>. The policy replaces complete
// core operations; generic scalar, geometry and execution paths are unchanged.
template<int ND, class ExecutionSpace=Kokkos::DefaultExecutionSpace>
struct CoreHDG : Kernels<ND> {};

// An application can specialize CoreHDG<geometry, execution space> here. The
// header is included inside namespace fieldsplit, with all layouts available.
#ifdef EXASIM_COMPOSITE_HDG_OVERRIDE_HEADER
#include EXASIM_COMPOSITE_HDG_OVERRIDE_HEADER
#endif
