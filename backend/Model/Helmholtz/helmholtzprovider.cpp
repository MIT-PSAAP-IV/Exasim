#include "helmholtzprovider.hpp"
#include <stdexcept>

const ExasimDriverABI& getHelmholtzModel1ABI();
const ExasimDriverABI& getHelmholtzModel2ABI();
const ExasimDriverABI& getHelmholtzModel3ABI();

const ExasimDriverABI& GetHelmholtzModelABI(int nd)
{
    if (nd == 1) return getHelmholtzModel1ABI();
    if (nd == 2) return getHelmholtzModel2ABI();
    if (nd == 3) return getHelmholtzModel3ABI();
    throw std::invalid_argument("The internal Helmholtz AV filter supports only 1D, 2D, and 3D meshes.");
}
