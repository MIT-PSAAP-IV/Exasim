#include "modeldefaults.hpp"
#ifdef EXASIM_HELMHOLTZ_MODEL_1_HEADER
#define PdeModel HelmholtzGeneratedModel1
#include EXASIM_HELMHOLTZ_MODEL_1_HEADER
#undef PdeModel
using PdeModel = HelmholtzGeneratedModel1;
#else
#include "helmholtz_model.hpp"
using PdeModel = HelmholtzModel<1>;
#endif
#define kokkos_kernel_source helmholtz_kernel_source_1
#define getKokkosKernelExasimDriverABI getHelmholtzModel1ABIInternal
#include "modelprovider.hpp"

const ExasimDriverABI& getHelmholtzModel1ABI()
{
    return getHelmholtzModel1ABIInternal();
}
