#pragma once
#include <map>
#include <mutex>
#include <cstddef>
#include <cstdio>
// High-water application allocations, not total hardware memory: covers
// TemplateMalloc and Kokkos allocations, excludes BLAS/runtime private heaps
// and native raw host malloc. All builds use the same hooks.
#ifdef EXASIM_COMPOSITE_TRACK_ALLOCATIONS
void numerical3dMemorySample();
#else
inline void numerical3dMemorySample(){}
#endif
namespace fsmemory {
inline std::mutex mutex;
inline std::map<const void*,size_t> live;
inline size_t current=0,peak=0;
inline void allocated(const void* p,size_t n){
#ifndef EXASIM_COMPOSITE_TRACK_ALLOCATIONS
 (void)p;(void)n;return;
#endif
if(!p||!n)return;std::lock_guard<std::mutex> g(mutex);auto it=live.find(p);if(it!=live.end())current-=it->second;live[p]=n;current+=n;if(current>peak)peak=current;numerical3dMemorySample();}
inline void released(const void* p){
#ifndef EXASIM_COMPOSITE_TRACK_ALLOCATIONS
 (void)p;return;
#endif
if(!p)return;std::lock_guard<std::mutex> g(mutex);auto it=live.find(p);if(it!=live.end()){current-=it->second;live.erase(it);}}
}
