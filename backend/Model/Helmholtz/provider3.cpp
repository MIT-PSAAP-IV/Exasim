#include "modeldefaults.hpp"
#ifdef EXASIM_HELMHOLTZ_MODEL_3_HEADER
#define PdeModel HelmholtzGeneratedModel3
#include EXASIM_HELMHOLTZ_MODEL_3_HEADER
#undef PdeModel
using PdeModel = HelmholtzGeneratedModel3;
#else
#include "helmholtz_model.hpp"
using PdeModel = HelmholtzModel<3>;
#endif
#define kokkos_kernel_source helmholtz_kernel_source_3
#define getKokkosKernelExasimDriverABI getHelmholtzModel3ABIInternal
#include "modelprovider.hpp"

const ExasimDriverABI& getHelmholtzModel3ABI()
{
    return getHelmholtzModel3ABIInternal();
}
