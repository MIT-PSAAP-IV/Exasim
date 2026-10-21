# Opening of the generated Kokkos::parallel_for over the quadrature points, shared by every Kokkos generator.
#
# HIP (MI300A / CDNA3): the default ~256-thread workgroups launch too few workgroups to fill the GPU for these
# per-point physics kernels. LaunchBounds<64,1> gives 1-wavefront workgroups, so ~4x more workgroups and all CUs
# busy (the flux kernel ran 1.70x faster, same results). The MATLAB frontend applies the same rewrite to every
# generated kernel (kklaunchbounds.m). Guarded to HIP: CUDA and host backends keep the default policy.
def parallel_for_open(filename, n="ng"):
    return ("\tKokkos::parallel_for(\"" + filename + "\",\n"
            "#if defined(KOKKOS_ENABLE_HIP)\n"
            "\t\tKokkos::RangePolicy<Kokkos::LaunchBounds<64,1>>(0, (size_t)" + n + "),\n"
            "#else\n"
            "\t\t" + n + ",\n"
            "#endif\n"
            "\t\tKOKKOS_LAMBDA(const size_t i) {\n")
