#include "composite_finite.hpp"
#include <iostream>
#include <stdexcept>
#include <vector>

struct Case { std::uint64_t bits; bool invalidDouble, invalidFloat; };
int main(int argc,char** argv) {
    Kokkos::initialize(argc,argv);
    int result=0;
    try {
        const std::vector<Case> cases{
            {UINT64_C(0x7ff8000000000001),true,true}, // positive quiet NaN
            {UINT64_C(0xfff8000000000001),true,true}, // negative quiet NaN
            {UINT64_C(0x7ff0000000000001),true,true}, // signaling NaN
            {UINT64_C(0xfff0000000000001),true,true},
            {UINT64_C(0x7ff0000000000000),true,true}, // infinities
            {UINT64_C(0xfff0000000000000),true,true},
            {UINT64_C(0x0000000000000000),false,false},
            {UINT64_C(0x8000000000000000),false,false},
            {UINT64_C(0x3ff0000000000000),false,false}, // +/-1
            {UINT64_C(0xbff0000000000000),false,false},
            {UINT64_C(0x0000000000000001),false,false}, // +/- smallest subnormal
            {UINT64_C(0x8000000000000001),false,false},
            {UINT64_C(0x0010000000000000),false,false}, // smallest normal
            {UINT64_C(0x7fefffffffffffff),true,true},   // double overflow bound
            {UINT64_C(0xffefffffffffffff),true,true},
            {UINT64_C(0x47efffffe0000000),false,false}, // exactly largest float
            {UINT64_C(0xc7efffffe0000000),false,false},
            {UINT64_C(0x47efffffe0000001),false,true},  // first double above float bound
            {UINT64_C(0xc7efffffe0000001),false,true},
            {Kokkos::bit_cast<std::uint64_t>(1.e300),false,true},
            {Kokkos::bit_cast<std::uint64_t>(-1.e300),false,true},
            {Kokkos::bit_cast<std::uint64_t>(1.e301),true,true},
            {Kokkos::bit_cast<std::uint64_t>(-1.e301),true,true}
        };
        Kokkos::View<std::uint64_t*> bits("finite-test bits",cases.size());
        Kokkos::View<int*> flags("finite-test flags",3*cases.size());
        auto hostBits=Kokkos::create_mirror_view(bits);
        for(size_t i=0;i<cases.size();++i) {
            hostBits[i]=cases[i].bits;
            const double value=Kokkos::bit_cast<double>(cases[i].bits);
            const bool finite=(cases[i].bits & UINT64_C(0x7ff0000000000000))!=UINT64_C(0x7ff0000000000000);
            if(exasim::composite::finiteValue(value)!=finite ||
               exasim::composite::invalidInverseValue(value,false)!=cases[i].invalidDouble ||
               exasim::composite::invalidInverseValue(value,true)!=cases[i].invalidFloat)
                throw std::runtime_error("Host inverse guard failed at case "+std::to_string(i));
        }
        Kokkos::deep_copy(bits,hostBits);
        Kokkos::parallel_for("finite guard exact production helper",int(cases.size()),KOKKOS_LAMBDA(int i) {
            const double value=Kokkos::bit_cast<double>(bits[i]);
            flags[3*i]=exasim::composite::invalidInverseValue(value,false);
            flags[3*i+1]=exasim::composite::invalidInverseValue(value,true);
            flags[3*i+2]=exasim::composite::finiteValue(value);
        });
        auto hostFlags=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),flags);
        for(size_t i=0;i<cases.size();++i) {
            const bool finite=(cases[i].bits & UINT64_C(0x7ff0000000000000))!=UINT64_C(0x7ff0000000000000);
            if(hostFlags[3*i]!=cases[i].invalidDouble || hostFlags[3*i+1]!=cases[i].invalidFloat || hostFlags[3*i+2]!=finite)
                throw std::runtime_error("Kokkos inverse guard failed at case "+std::to_string(i));
        }
        std::cout << "COMPOSITE_FINITE_FASTMATH_PASS cases=" << cases.size() << " host=1 kokkos=1\n";
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';result=1;}
    Kokkos::finalize();
    return result;
}
