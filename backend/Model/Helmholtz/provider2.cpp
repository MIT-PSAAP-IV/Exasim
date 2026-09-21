#include "modeldefaults.hpp"
#ifdef EXASIM_HELMHOLTZ_MODEL_2_HEADER
#define PdeModel HelmholtzGeneratedModel2
#include EXASIM_HELMHOLTZ_MODEL_2_HEADER
#undef PdeModel
using PdeModel = HelmholtzGeneratedModel2;
#else
#include "helmholtz_model.hpp"
using PdeModel = HelmholtzModel<2>;
#endif
#define kokkos_kernel_source helmholtz_kernel_source_2
#define getKokkosKernelExasimDriverABI getHelmholtzModel2ABIInternal
#include "modelprovider.hpp"

const ExasimDriverABI& getHelmholtzModel2ABI()
{
    return getHelmholtzModel2ABIInternal();
}
