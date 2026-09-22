#include "linear_elasticity_provider.hpp"
#include <stdexcept>

const ExasimDriverABI& getLinearElasticityModel2ABI();
const ExasimDriverABI& getLinearElasticityModel3ABI();

const ExasimDriverABI& GetLinearElasticityModelABI(int nd)
{
    if (nd == 2) return getLinearElasticityModel2ABI();
    if (nd == 3) return getLinearElasticityModel3ABI();
    throw std::invalid_argument("Backend mesh elasticity supports only 2D and 3D meshes.");
}
