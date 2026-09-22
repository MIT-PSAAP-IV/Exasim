#include "modeldefaults.hpp"
#ifdef EXASIM_LINEAR_ELASTICITY_MODEL_3_HEADER
#define PdeModel LinearElasticityGeneratedModel3
#include EXASIM_LINEAR_ELASTICITY_MODEL_3_HEADER
#undef PdeModel
using PdeModel = LinearElasticityGeneratedModel3;
#else
#include "linear_elasticity_model.hpp"
using PdeModel = LinearElasticityModel<3>;
#endif
#define kokkos_kernel_source linear_elasticity_kernel_source_3
#define getKokkosKernelExasimDriverABI getLinearElasticityModel3ABIInternal
#include "modelprovider.hpp"

const ExasimDriverABI& getLinearElasticityModel3ABI()
{
    return getLinearElasticityModel3ABIInternal();
}
