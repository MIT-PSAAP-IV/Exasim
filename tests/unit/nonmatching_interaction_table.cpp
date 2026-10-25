#ifdef EXASIM_APP_ROUNDTRIP
#include <Kokkos_Core.hpp>
// common.h contains a few legacy non-inline helpers. Rename those test-local
// definitions so this fixture can use the real appstruct layout while linking
// the production reader/writer from cpuEXASIM_static without ODR collisions.
#define trimToSubstringAtLastOccurence phase1_test_trim_last
#define trimToSubstringAtFirstOccurence phase1_test_trim_first
#define make_path phase1_test_make_path
#define trim_dir phase1_test_trim_dir
#define ensure_dir phase1_test_ensure_dir
#define is_nan_bitwise phase1_test_is_nan_bitwise
#define cublasMinusone phase1_test_cublasMinusone
#define cublasOne phase1_test_cublasOne
#define cublasZero phase1_test_cublasZero
#include "backend/Common/common.h"
#undef trimToSubstringAtLastOccurence
#undef trimToSubstringAtFirstOccurence
#undef make_path
#undef trim_dir
#undef ensure_dir
#undef is_nan_bitwise
#undef cublasMinusone
#undef cublasOne
#undef cublasZero
#endif
#include "backend/Discretization/nonmatchinginteraction.hpp"

#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef EXASIM_APP_ROUNDTRIP
void readappstruct(std::string filename, appstruct& app);
void writeappstruct(std::string filename, appstruct& app);
#endif

static std::vector<char> file_bytes(const std::string& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open round-trip input");
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

int main(int argc, char** argv)
{
    const std::vector<double> chain = {
        1,3,2,
        11,0,1,1,2,1,1,0,4,1,1,0,
        11,1,2,1,2,1,1,0,4,1,1,0};
    const auto records = exasim::nonmatching::parse_interaction_table<int>(chain.data(), chain.size());
    if (chain.size() != 27 || records.size() != 2 || records[0].ordinal != 0 ||
        records[1].endpoint[1].domain != 2 || records[0].endpoint[0].interfaceSelector != 2)
        return 1;
    if (!exasim::nonmatching::parse_interaction_table<int>(nullptr, 0).empty()) return 2;

    auto mustReject = [](std::vector<double> table) {
        try {
            (void) exasim::nonmatching::parse_interaction_table<int>(table.data(), table.size());
        } catch (const std::exception&) {
            return;
        }
        throw std::runtime_error("malformed interaction table was accepted");
    };
    auto bad = chain; bad[0] = 2; mustReject(bad);
    bad = chain; bad[3] = 10; mustReject(bad);
    bad = chain; bad[6] = 0; mustReject(bad);
    bad = chain; bad[9] = 0; mustReject(bad);
    bad = chain; bad.push_back(0); mustReject(bad);
    bad = chain; bad[4] = 0.5; mustReject(bad);

    if (argc == 3) {
#ifdef EXASIM_APP_ROUNDTRIP
        appstruct app;
        readappstruct(argv[1], app);
        if (app.szinteractions <= 0 || app.interactions == nullptr)
            throw std::runtime_error("round-trip input has no interaction table");
        std::vector<double> raw(static_cast<std::size_t>(app.szinteractions));
        for (std::size_t i = 0; i < raw.size(); ++i) raw[i] = app.interactions[i];
        if (exasim::nonmatching::parse_interaction_table<Int>(raw.data(),raw.size()).empty())
            throw std::runtime_error("round-trip input has no interaction record");
        writeappstruct(argv[2], app);
        if (file_bytes(argv[1]) != file_bytes(argv[2]))
            throw std::runtime_error("readappstruct/writeappstruct changed app.bin bytes");
#else
        throw std::runtime_error("this build does not provide the production app reader/writer");
#endif
    } else if (argc != 1) {
        std::cerr << "usage: " << argv[0] << " [input-app.bin output-app.bin]\n";
        return 3;
    }
    std::cout << "PASS: generalized interaction table structural parser\n";
    return 0;
}
