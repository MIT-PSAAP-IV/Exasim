#include "text2code/text2code/TextParser.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

#ifndef GENERATED_MODEL_HEADER

int main()
{
    ParsedSpec legacy;
    if (legacy.hasinteractiongraph || legacy.nfint != 0 || legacy.maxinteractioncondition != 0)
        return 1;
    ParsedSpec graph;
    graph.hasinteractiongraph = true;
    graph.nfint = 1;
    graph.maxinteractioncondition = 2;
    if (graph.nfint <= 0 || 2 % graph.nfint != 0 ||
        graph.maxinteractioncondition > 2 / graph.nfint)
        return 2;
    std::cout << "PASS: Text2Code interaction condition metadata contract\n";
    return 0;
}

#else

using dstype = double;
template <class Derived> struct ModelDefaults {};
#include GENERATED_MODEL_HEADER

static double value(int ib, double uq0, double w0, double uh0)
{
    double f[1] = {-777.0};
    const double x[2] = {0.2,0.3}, uq[3] = {uq0,0.4,-0.6}, v[1] = {0.0};
    const double w[1] = {w0}, uh[1] = {uh0}, n[2] = {0.8,-0.6};
    const double tau[1] = {1.3}, mu[1] = {0.9}, uinf[1] = {0.0};
    PdeModel::fint(f,ib,x,uq,v,w,uh,n,tau,mu,uinf,0.1);
    return f[0];
}

static void require_close(double actual, double expected, double tolerance, const char* label)
{
    if (std::abs(actual-expected) > tolerance*std::max({1.0,std::abs(actual),std::abs(expected)}))
        throw std::runtime_error(std::string("Text2Code condition check failed: ")+label);
}

int main()
{
    static_assert(PdeModel::nfint == 1, "generated graph model must expose per-condition nfint");
    const double uq0=0.7, w0=-0.2, uh0=0.35;
    require_close(value(1,uq0,w0,uh0),2*uq0+3*w0+5*uh0,1e-13,"ib=1 value");
    require_close(value(2,uq0,w0,uh0),7*uq0*uq0+11*w0*w0+13*uh0*uh0,1e-13,"ib=2 value");

    double juq[3]={0,0,0}, jw[1]={0}, juh[1]={0};
    const double x[2]={0.2,0.3}, uq[3]={uq0,0.4,-0.6}, v[1]={0}, w[1]={w0};
    const double uh[1]={uh0}, n[2]={0.8,-0.6}, tau[1]={1.3}, mu[1]={0.9}, uinf[1]={0};
    for (int ib=1; ib<=2; ++ib) {
        PdeModel::fint_jac_uq(juq,ib,x,uq,v,w,uh,n,tau,mu,uinf,0.1);
        PdeModel::fint_jac_w(jw,ib,x,uq,v,w,uh,n,tau,mu,uinf,0.1);
        PdeModel::fint_jac_uh(juh,ib,x,uq,v,w,uh,n,tau,mu,uinf,0.1);
        const double scale=std::max({1.0,std::abs(uq0),std::abs(w0),std::abs(uh0)});
        const double h=std::sqrt(std::numeric_limits<double>::epsilon())*scale;
        for (double multiplier : {0.1,1.0,10.0}) {
            const double step=h*multiplier;
            const double du=(value(ib,uq0+step,w0,uh0)-value(ib,uq0-step,w0,uh0))/(2*step);
            const double dw=(value(ib,uq0,w0+step,uh0)-value(ib,uq0,w0-step,uh0))/(2*step);
            const double dh=(value(ib,uq0,w0,uh0+step)-value(ib,uq0,w0,uh0-step))/(2*step);
            require_close(juq[0],du,2e-7,"uq finite difference");
            require_close(jw[0],dw,2e-7,"w finite difference");
            require_close(juh[0],dh,2e-7,"uh finite difference");
        }
        require_close(juq[1],0.0,1e-13,"unused uq[1] derivative");
        require_close(juq[2],0.0,1e-13,"unused uq[2] derivative");
    }
    std::cout << "PASS: generated two-condition Fint values and finite-difference Jacobians\n";
    return 0;
}

#endif
