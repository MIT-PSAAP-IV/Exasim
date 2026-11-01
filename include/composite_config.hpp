#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace exasim::composite {
struct Config { bool enabled=false; int batch=1024; };
template<int D> struct Shape {
    static_assert(D == 2 || D == 3);
    static constexpr int nd=D, P=D==2?9:27, PF=D==2?3:9, faces=2*D;
    static constexpr int Q=PF*faces, NC=6+D, state=NC*(D+1), other=11+D;
    static constexpr int flow=D+2, N=P*NC, M=Q*NC;
    static constexpr int AF=P*flow, BF=Q*flow, AS=P*4, BS=Q*4;
    static constexpr int HS=BF*BF+4*Q*Q, FS=AF*BF+4*P*Q;
    static constexpr int thermal=nd+2*state+other+3;
};
struct Dimensions {
    int nd,P,PF,Q,NC,state,other,flow,N,M,AF,BF,AS,BS,HS,FS,thermal;
};
template<int D> constexpr Dimensions dimensions() {
    using S=Shape<D>;
    return {D,S::P,S::PF,S::Q,S::NC,S::state,S::other,S::flow,S::N,S::M,
            S::AF,S::BF,S::AS,S::BS,S::HS,S::FS,S::thermal};
}
inline Dimensions dimensions(int nd) {
    if(nd==2)return dimensions<2>();
    if(nd==3)return dimensions<3>();
    throw std::invalid_argument("compositehdg requires two or three dimensions");
}
struct Layout {
    static constexpr int H=84564, F=42282, fullH=236196, fullF=118098;
    static constexpr int asmSlots=H/2;
};
inline int integer(double value,const char* name,int minimum,int maximum) {
    // The solver may be compiled with -ffast-math; std::isfinite can then be
    // folded to true. Inspect IEEE-754 exponent bits before any arithmetic.
    static_assert(sizeof(double)==sizeof(uint64_t) && std::numeric_limits<double>::is_iec559);
    uint64_t bits;
    std::memcpy(&bits,&value,sizeof(bits));
    const bool finite=(bits&UINT64_C(0x7ff0000000000000))!=UINT64_C(0x7ff0000000000000);
    if(!finite || value<minimum || value>maximum || std::floor(value)!=value)
        throw std::invalid_argument(std::string(name)+" must be an integer in ["+
                                    std::to_string(minimum)+","+std::to_string(maximum)+"]");
    return static_cast<int>(value);
}
// Parse these settings before the generic text reader can truncate to int.
inline int parseInteger(const std::string& text,const char* name,int minimum,int maximum) {
    size_t consumed = 0;
    double value;
    try {
        value = std::stod(text, &consumed);
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string(name)+" must be a finite integer");
    }
    if (text.find_first_not_of(" \t\r\n", consumed) != std::string::npos)
        throw std::invalid_argument(std::string(name)+" must be a scalar integer");
    return integer(value,name,minimum,maximum);
}
inline int count(uint64_t value,const char* name) {
    if(value>uint64_t(std::numeric_limits<int>::max()))
        throw std::overflow_error(std::string(name)+" exceeds a signed 32-bit kernel interface");
    return static_cast<int>(value);
}
inline Config decodeValues(double mode,double cap) {
    Config config;
    config.enabled=integer(mode,"compositehdg",0,1)!=0;
    // Binary metadata compatibility: zero means an unset, formerly reserved
    // app.bin slot, including a missing slot supplied by decode(). Explicit
    // frontend/text settings and new binary writers require a positive cap;
    // zero is not a valid user-requested batch size.
    const int checkedCap=integer(cap,"compositebatchsize",0,2147483647);
    config.batch=checkedCap==0?1024:checkedCap;
    return config;
}
template<class App> Config decode(const App& app) {
    // Lightweight native callers may not populate the optional app.bin header.
    if(!app.nsize || !app.ndims)return {};
    return decodeValues(app.nsize[0]>23?app.ndims[23]:0,
                        app.nsize[0]>24?app.ndims[24]:0);
}
// Call only on host-resident app metadata (before GPU staging/allocation).
// Auxiliary scalar/elasticity models reread the same app.bin, so this initial
// composite implementation rejects options that would construct those models.
template<class App> void validateOptions(const Config& config,const App& app) {
    if(!config.enabled)return;
    if(app.szavfilterparam>0) {
        if(!app.avfilterparam)
            throw std::invalid_argument("compositehdg: missing AV filter parameters");
        if(integer(app.avfilterparam[0],"AVsmoothingMethod",0,1)!=0)
            throw std::invalid_argument("compositehdg does not yet support Helmholtz AV smoothing; use AVsmoothingMethod=0");
    }
    if(app.szmeshadaptparam>0) {
        if(!app.meshadaptparam)
            throw std::invalid_argument("compositehdg: missing mesh-adaptation parameters");
        if(integer(app.meshadaptparam[0],"meshadapt",0,1)!=0)
            throw std::invalid_argument("compositehdg does not yet support mesh adaptation");
    }
}
template<class Scalar,class App,class Master>
void validateShape(const Config& config,const App& app,const Master& master) {
    if(!config.enabled)return;
    validateOptions(config,app);
    if(app.nsize[0]<14 || app.nsize[1]<7 || app.nsize[2]<29)
        throw std::invalid_argument("compositehdg requires complete app dimensions, flags and problem settings");
    const auto d=dimensions(master.ndims[0]);
    if(!std::is_same_v<Scalar,double> || app.problem[0]!=1 ||
       master.ndims[1]!=1 || master.ndims[3]!=2 || master.ndims[5]!=d.P || master.ndims[6]!=d.PF ||
       app.ndims[5]!=d.state || app.ndims[6]!=d.NC || app.ndims[7]!=d.NC*d.nd || app.ndims[9]!=d.other ||
       app.ndims[10]!=d.NC || app.ndims[11]!=d.nd || app.ndims[13]!=1 ||
       app.flag[6]!=1 || app.problem[28]!=0 || app.flag[3]!=0)
        throw std::invalid_argument("compositehdg requires double P2 quad/hex CNS5air HDG, elemental ASM, no coupled interfaces/debug dumps");
    if(app.nsize[2]>34 && app.problem[34]!=0)
        throw std::invalid_argument("compositehdg does not support nonmatching interfaces");
    if(app.szinteractions>0)
        throw std::invalid_argument("compositehdg does not support generalized coupled interactions");
    // Cap bound is a supported temporary-count limit, not a power-of-two rule.
    // Validate requested cap as well as effective block: an unsafe request must
    // not pass only because today's partition happens to be small.
    count(uint64_t(config.batch)*309960,"compositebatchsize temporary count");
}
inline void checkTemporaryCounts(int block,int elements,int faceblock,int cg,int nge,int ngf) {
    if(block<0 || elements<0 || faceblock<0 || cg<0 || nge<0 || ngf<0)
        throw std::invalid_argument("negative composite allocation dimension");
    count(uint64_t(block)*Layout::fullH,"full H workspace");
    count(uint64_t(block)*201015,"field workspace");
    count(uint64_t(block)*nge*4*9*27,"volume derivative workspace");
    count(uint64_t(block)*ngf*6*2296,"face derivative workspace");
    count(uint64_t(elements)*27*14*2,"nodal workspace");
    count(uint64_t(elements)*27*36,"state extent");
    count(uint64_t(faceblock)*9*9*9*9,"face factor workspace");
    count(uint64_t(cg)*2,"CG workspace");
}
} // namespace exasim::composite
