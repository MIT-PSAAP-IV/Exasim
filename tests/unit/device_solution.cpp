#include <Kokkos_Core.hpp>
#include <iostream>
#include <fstream>
#include <numeric>
#include <algorithm>
using std::partial_sum;
using std::max;
using std::string;
#include "../../backend/Common/common.h"
#include "../../backend/Common/cpuimpl.h"
#include "../../backend/Common/kokkosimpl.h"
#include "../../backend/Common/pblas.h"
#include <exasim/drivers.hpp>
#include <exasim/execution_mode.hpp>
// Execute the real device setup on host allocations, so missing host metadata
// is covered even on CI machines without a CUDA/HIP device.
#define HAVE_GPU
#ifdef TEST_LEGACY_SETUP
#include "../../backend/Discretization/setstructs.cpp"
#else
#include "../../backend/Discretization/setstructs.hpp"
#endif

int main(int argc,char** argv) {
    Kokkos::initialize(argc,argv);
    int failures=0;
    {
        solstruct source{}, device{};
        commonstruct common{};
        common.backend=0;
        Int lengths[1]={5}, sizes[5]={1,0,0,0,7}, dims[1]={0};
        double w[7]={1,2,3,4,5,6,7};
        source.lsize=lengths; source.nsize=sizes; source.ndims=dims;
        source.wdg=w; source.szwdg=7;
        devsolstruct(device,source,common);
        failures+=device.szwdg!=7;
        {
            AuxiliaryStateSnapshot<double> snapshot(device.wdg,device.szwdg);
            ArraySetNaN(device.wdg,7);
        }
        for(int i=0;i<7;++i) failures+=device.wdg[i]!=w[i];
        device.freememory(0);
        solstruct empty{};
        sizes[4]=0; source.wdg=nullptr; source.szwdg=0;
        devsolstruct(empty,source,common);
        failures+=empty.szwdg!=0 || empty.wdg!=nullptr;
        AuxiliaryStateSnapshot<double> emptySnapshot(empty.wdg,empty.szwdg);
        empty.freememory(0);
    }
    Kokkos::finalize();
    std::printf("device solution extent/rollback: %d failures\n",failures);
    return failures ? 1:0;
}
