#include "modeldefaults.hpp"
#ifdef EXASIM_LINEAR_ELASTICITY_MODEL_2_HEADER
#define PdeModel LinearElasticityGeneratedModel2
#include EXASIM_LINEAR_ELASTICITY_MODEL_2_HEADER
#undef PdeModel
using PdeModel = LinearElasticityGeneratedModel2;
#else
#include "linear_elasticity_model.hpp"
using PdeModel = LinearElasticityModel<2>;
#endif
#define kokkos_kernel_source linear_elasticity_kernel_source_2
#define getKokkosKernelExasimDriverABI getLinearElasticityModel2ABIInternal
#include "modelprovider.hpp"

const ExasimDriverABI& getLinearElasticityModel2ABI()
{
    return getLinearElasticityModel2ABIInternal();
}
