#include "../backend/Common/storage_index.hpp"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <iostream>

template<class F> void rejects(F f) {
    bool caught=false;
    try { f(); } catch(const std::overflow_error&) { caught=true; }
    assert(caught);
}
int main() {
    using namespace exasim::index;
    constexpr Storage limit=std::numeric_limits<std::int32_t>::max();
    static_assert(uses32BitLoop(0) && uses32BitLoop(limit));
    static_assert(!uses32BitLoop(limit+1));
    assert(loopCount<std::int32_t>(limit)==2147483647);
    assert(loopCount<std::int64_t>(limit+1)==2147483648LL);
    rejects([&]{loopCount<std::int32_t>(limit+1);});
    rejects([&]{loopCount<std::int64_t>(std::numeric_limits<Storage>::max());});
    assert(storageProduct<double>(84564,25394,"H")<=limit);
    assert(storageProduct<double>(84564,25395,"H")>limit);
    assert(storageProduct<double>(84564,40000,"H")==3382560000ULL);
    // A legal int vector length can still require wide basis-column offsets.
    const std::int32_t rows=30000000, column=85;
    assert(Storage(column)*rows==2550000000ULL);
    assert(storageProduct<double>(0,std::numeric_limits<std::uint64_t>::max(),"zero")==0);
    const auto bytes=std::uint64_t(std::numeric_limits<std::ptrdiff_t>::max())/8;
    assert(storageCount<double>(bytes,"last")==bytes);
    rejects([&]{storageCount<double>(bytes+1,"overflow");});
    rejects([&]{storageProduct<double>(1ULL<<63,8,"multiply before checking");});
    rejects([&]{storageProduct<double>(84564,std::numeric_limits<std::uint64_t>::max(),"product");});
    std::cout<<"STORAGE_INDEX_HOST_PASS\n";
}
