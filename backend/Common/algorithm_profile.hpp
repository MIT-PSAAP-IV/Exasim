#pragma once
#include <Kokkos_Core.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>

// Kokkos Tools regions are exported as rocTX by rocprofv3 --kokkos-trace.
namespace exasim::profile {
inline int mode() {
    static int value=[] { const char* p=std::getenv("EXASIM_ALGORITHM_PROFILE");
        return !p || !std::strcmp(p,"off") ? 0 : !std::strcmp(p,"sync") ? 2 : 1; }();
    return value;
}
struct Entry { unsigned long long calls=0; double inclusive=0,exclusive=0; };
inline auto& entries() { static std::map<std::string,Entry> table; return table; }
class Scope;
inline thread_local Scope* current=nullptr;
class Scope {
    using Clock=std::chrono::steady_clock;
    const char* name_; Scope* parent_=nullptr; Clock::time_point begin_;
    double children_=0; bool active_=false;
public:
    explicit Scope(const char* name):name_(name),active_(mode()!=0) {
        if(!active_) return;
        if(mode()==2) Kokkos::fence();
        parent_=current; current=this;
        Kokkos::Profiling::pushRegion(name_);
        begin_=Clock::now();
    }
    Scope(const Scope&)=delete;
    ~Scope() {
        if(!active_) return;
        if(mode()==2) Kokkos::fence();
        double seconds=std::chrono::duration<double>(Clock::now()-begin_).count();
        Kokkos::Profiling::popRegion();
        auto& e=entries()[name_]; ++e.calls; e.inclusive+=seconds;
        e.exclusive+=seconds-children_;
        current=parent_; if(parent_) parent_->children_+=seconds;
    }
};
// Each rank writes one small aggregate CSV per window. No collective/fence
// here; the caller owns the observation boundary and solver synchronization.
inline void report(int rank,int window) {
    if(!mode() || entries().empty()) return;
    const char* dir=std::getenv("EXASIM_PROFILE_DIR");
    std::string path=std::string(dir?dir:".")+"/algorithm-rank"+
        std::to_string(rank)+"-window"+std::to_string(window)+".csv";
    FILE* f=std::fopen(path.c_str(),"w");
    if(!f) { std::perror(path.c_str()); return; }
    std::fprintf(f,"rank,window,mode,region,calls,inclusive_s,exclusive_s\n");
    for(const auto& [name,e]:entries())
        std::fprintf(f,"%d,%d,%s,%s,%llu,%.12g,%.12g\n",rank,window,
            mode()==2?"synchronized":"host",name.c_str(),e.calls,e.inclusive,e.exclusive);
    std::fclose(f); entries().clear();
}
}
#define EXASIM_PROFILE_JOIN_(a,b) a##b
#define EXASIM_PROFILE_JOIN(a,b) EXASIM_PROFILE_JOIN_(a,b)
#define EXASIM_PROFILE_SCOPE(name) ::exasim::profile::Scope EXASIM_PROFILE_JOIN(exasim_profile_,__LINE__)(name)
