#include "composite_blocks.hpp"
#include <cassert>
#include <climits>
#include <cstdlib>
#include <iostream>

using namespace exasim::composite;
struct Mesh {
    int lengths[1] = {50}, sizes[50] = {20}, dimensions[20] = {};
    int* lsize = lengths;
    int* nsize = sizes;
    int* ndims = dimensions;
    int* eblks = nullptr;
    int szeblks = 0;
    int untouchedConnectivity[4] = {42,17,8,99};
    ~Mesh() { std::free(eblks); }
    void set(const std::vector<int>& blocks, int elements, int maximum) {
        std::free(eblks); eblks = nullptr;
        szeblks = nsize[2] = int(blocks.size());
        ndims[1] = elements; ndims[5] = int(blocks.size()/3); ndims[6] = maximum;
        if (szeblks) { eblks = static_cast<int*>(std::malloc(blocks.size()*sizeof(int))); std::copy(blocks.begin(),blocks.end(),eblks); }
    }
};
template<class Function> void rejects(Function operation) {
    bool rejected = false;
    try { operation(); } catch (const std::exception&) { rejected = true; }
    assert(rejected);
}
int main() {
    const Config on{true,4}, off{false,4};
    const std::vector<int> original{1,6,0, 7,13,1, 14,18,2};
    const std::vector<int> expected{1,4,0, 5,6,0, 7,10,1, 11,13,1, 14,17,2, 18,18,2};
    Mesh mesh;mesh.set(original,18,7);
    int allocations=0,releases=0;
    auto alloc=[&](int n){++allocations;return static_cast<int*>(std::malloc(size_t(n)*sizeof(int)));};
    auto release=[&](int*& p){++releases;std::free(p);p=nullptr;};
    int* initial=mesh.eblks;
    limitMeshBlocks(mesh,off,alloc,release);
    assert(mesh.eblks==initial && mesh.ndims[5]==3 && mesh.ndims[6]==7 && !allocations && !releases);
    limitMeshBlocks(mesh,on,alloc,release);
    assert(mesh.ndims[5]==6 && mesh.ndims[6]==4 && mesh.nsize[2]==18 && mesh.szeblks==18);
    assert(std::vector<int>(mesh.eblks,mesh.eblks+18)==expected && allocations==1 && releases==1);
    assert(mesh.untouchedConnectivity[0]==42 && mesh.untouchedConnectivity[3]==99);
    int* split=mesh.eblks;limitMeshBlocks(mesh,on,alloc,release);
    assert(mesh.eblks==split && allocations==1 && releases==1); // generated/already-limited blocks are untouched
    // Tags and even same-tag interval boundaries are preserved; arbitrary caps work.
    const std::vector<int> sameTags{1,2,7,3,7,7};
    auto p=limitBlockIntervals(sameTags.data(),6,7,2,5,Config{true,3});
    assert((p.entries==std::vector<int>{1,2,7,3,5,7,6,7,7}));
    // An overly conservative serialized maximum must not defeat the cap.
    p=limitBlockIntervals(sameTags.data(),6,7,2,1024,Config{true,5});
    assert(p.changed && p.entries==sameTags && p.maximum==5);
    assert(!limitBlockIntervals(nullptr,0,0,0,0,on).changed);
    assert(!limitBlockIntervals(nullptr,-1,-1,-1,-1,off).changed); // no feature-off validation/regression
    rejects([&]{limitBlockIntervals(nullptr,0,1,0,0,on);});
    rejects([&]{limitBlockIntervals(nullptr,3,1,1,1,on);});
    rejects([&]{limitBlockIntervals(original.data(),8,18,3,7,on);});
    rejects([&]{limitBlockIntervals(original.data(),9,18,3,6,on);});
    rejects([&]{limitBlockIntervals(original.data(),9,19,3,7,on);});
    rejects([&]{limitBlockIntervals(original.data(),9,18,3,7,Config{true,0});});
    for (const std::vector<int>& invalid : {std::vector<int>{0,1,0}, {1,0,0}, {2,2,0}, {1,3,0},
                                          {1,1,0,1,2,1}, {1,1,0,3,3,1}})
        rejects([&]{limitBlockIntervals(invalid.data(),int(invalid.size()),2,int(invalid.size()/3),3,on);});
    const int huge[]={1,INT_MAX,0};
    rejects([&]{limitBlockIntervals(huge,3,INT_MAX,1,INT_MAX,Config{true,1});});
    p=limitBlockIntervals(huge,3,INT_MAX,1,INT_MAX,Config{true,INT_MAX-1});
    assert((p.entries==std::vector<int>{1,INT_MAX-1,0,INT_MAX,INT_MAX,0}));
    // Invalid metadata and failed allocation must preserve the original table.
    mesh.nsize[2]=17;rejects([&]{limitMeshBlocks(mesh,on,alloc,release);});
    assert(mesh.eblks==split);mesh.nsize[2]=18;
    mesh.set(original,18,7);initial=mesh.eblks;
    rejects([&]{limitMeshBlocks(mesh,on,[](int){return static_cast<int*>(nullptr);},release);});
    assert(mesh.eblks==initial && mesh.ndims[5]==3 && mesh.nsize[2]==9);
    std::cout << "COMPOSITE_BLOCK_LIMIT_PASS\n";
}
