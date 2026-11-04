"""Check per-model mass overloads and column-major derivatives."""

import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import sympy as sp

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "frontends" / "Python"))
from exasim.Gencode.gencodeelem import gencodeelem
from exasim.Gencode.hdggencodeelem import hdggencodeelem


class MassDerivativeTests(unittest.TestCase):
    def test_generated_overload_and_column_major_partials(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("C++ compiler unavailable")
        u = np.array(sp.symbols("udg1:4"), dtype=object)
        w = np.array(sp.symbols("wdg1:3"), dtype=object)
        empty = np.array([], dtype=object)
        cases = {
            "Constant": np.array([2, 3], dtype=object),
            "State": np.array([1 + u[0]**2 + u[1]*w[0], u[2]**2 + w[1]**2], dtype=object),
            "Steady": np.zeros(2, dtype=object),
        }
        with tempfile.TemporaryDirectory(prefix="exasim-mass-generator-") as directory:
            generated = []
            for name, f in cases.items():
                args = (name, f, empty, u, empty, w, empty, empty, sp.Symbol("time"), directory)
                gencodeelem(*args)
                hdggencodeelem(*args, "Kokkos" + name)
                generated.extend([f'#include "Kokkos{name}.cpp"', f'#include "Hdg{name}.cpp"'])
            # Sequential shim exercises the generated arithmetic and layouts;
            # actual backend/device execution is covered by integration tests.
            source = r'''
#include <cmath>
#include <cstddef>
#include <algorithm>
using dstype = double;
#define KOKKOS_LAMBDA [=]
namespace Kokkos {
template<class F> void parallel_for(const char*, int n, F f) {
    for (int i=0;i<n;++i) f(i);
}}
''' + "\n".join(generated) + r'''
int main() {
    constexpr int ng=2, nc=3, ncu=2, nw=2;
    double u[]={.7,.8,.4,.5,.2,.3}, w[]={.1,.2,.6,.7};
    double f[4], g[4], du[12], dw[8], plus[4], minus[4];
    auto check=[&](auto value, auto jac) {
        jac(f,du,dw,nullptr,u,nullptr,w,nullptr,nullptr,0.,0,ng,nc,ncu,1,0,0,nw);
        value(g,nullptr,u,nullptr,w,nullptr,nullptr,0.,0,ng,nc,ncu,1,0,0,nw);
        for(int j=0;j<4;++j) if(f[j]!=g[j]) return false;
        for(int kind=0;kind<2;++kind) {
            double* state=kind?w:u; double* derivative=kind?dw:du;
            const int count=kind?nw:nc;
            for(int k=0;k<count;++k) for(int i=0;i<ng;++i) {
                double& entry=state[i+ng*k]; const double old=entry, h=1e-6;
                entry=old+h; value(plus,nullptr,u,nullptr,w,nullptr,nullptr,0.,0,ng,nc,ncu,1,0,0,nw);
                entry=old-h; value(minus,nullptr,u,nullptr,w,nullptr,nullptr,0.,0,ng,nc,ncu,1,0,0,nw);
                entry=old;
                for(int m=0;m<ncu;++m) {
                    const double fd=(plus[i+ng*m]-minus[i+ng*m])/(2*h);
                    if(std::abs(fd-derivative[i+ng*(m+ncu*k)])>1e-8) return false;
                }
            }
        }
        return true;
    };
    using Value=void(*)(double*,const double*,const double*,const double*,const double*,const double*,const double*,double,int,int,int,int,int,int,int,int);
    using Jac=void(*)(double*,double*,double*,const double*,const double*,const double*,const double*,const double*,const double*,double,int,int,int,int,int,int,int,int);
    if(!check(static_cast<Value>(KokkosConstant),static_cast<Jac>(KokkosConstant))) return 1;
    if(!check(static_cast<Value>(KokkosState),static_cast<Jac>(KokkosState))) return 2;
    if(!check(static_cast<Value>(KokkosSteady),static_cast<Jac>(KokkosSteady))) return 3;
}
'''
            path = pathlib.Path(directory)
            (path / "probe.cpp").write_text(source)
            subprocess.run([compiler, "-std=c++17", "-O2", str(path / "probe.cpp"),
                            "-o", str(path / "probe")], check=True, capture_output=True, text=True)
            subprocess.run([str(path / "probe")], check=True)


if __name__ == "__main__":
    unittest.main()

