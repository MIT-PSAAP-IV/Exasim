#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using dstype = double;
using Int = int;

template <class T, class I>
struct appstructT {
    I materialdb_nstate = 0;
    I materialdb_nprop = 0;
    I materialdb_porder = 0;
    I materialdb_elemtype = 0;
    I materialdb_npe = 0;
    I materialdb_ne = 0;

    I* materialdb_elementcounts = nullptr;
    I* materialdb_ncgi = nullptr;
    I* materialdb_gridoffset = nullptr;
    I* materialdb_elemoffset = nullptr;

    T* materialdb_statecoords = nullptr;
    T* materialdb_propvalues = nullptr;
    T* materialdb_gridcoords = nullptr;
    T* materialdb_elemcoords = nullptr;

    I szmaterialdb_elementcounts = 0;
    I szmaterialdb_ncgi = 0;
    I szmaterialdb_gridoffset = 0;
    I szmaterialdb_elemoffset = 0;
    I szmaterialdb_statecoords = 0;
    I szmaterialdb_propvalues = 0;
    I szmaterialdb_gridcoords = 0;
    I szmaterialdb_elemcoords = 0;
};

using appstruct = appstructT<dstype, Int>;

template <typename T>
static void TemplateMalloc(T** data, Int n, Int)
{
    *data = nullptr;
    if (n > 0) {
        *data = static_cast<T*>(std::malloc(static_cast<std::size_t>(n) * sizeof(T)));
        if (*data == nullptr) {
            throw std::bad_alloc();
        }
    }
}

template <typename T>
static void TemplateFree(T*& data, Int)
{
    std::free(data);
    data = nullptr;
}

#include <backend/Discretization/appstruct_materialdatabase.hpp>

namespace {

void require(bool ok, const std::string& message)
{
    if (!ok) {
        throw std::runtime_error(message);
    }
}

void requireClose(double a, double b, const std::string& message)
{
    if (std::abs(a - b) > 1.0e-14) {
        throw std::runtime_error(message);
    }
}

void writeBin(const std::string& filename, const std::vector<double>& values)
{
    std::ofstream out(filename, std::ios::binary);
    require(static_cast<bool>(out), "could not open test binary for writing");
    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(double)));
    require(static_cast<bool>(out), "failed to write test binary");
}

std::size_t linearGridIndex(const std::vector<int>& idx, const std::vector<int>& dims)
{
    std::size_t out = 0;
    std::size_t stride = 1;
    for (std::size_t is = 0; is < idx.size(); ++is) {
        out += static_cast<std::size_t>(idx[is]) * stride;
        stride *= static_cast<std::size_t>(dims[is]);
    }
    return out;
}

std::vector<double> makeDatabase(int nstate,
                                 int nprop,
                                 const std::vector<std::vector<double>>& axes,
                                 bool shuffled = true)
{
    std::vector<int> dims{1, 1, 1};
    for (int is = 0; is < nstate; ++is) {
        dims[static_cast<std::size_t>(is)] =
            static_cast<int>(axes[static_cast<std::size_t>(is)].size());
    }
    std::vector<double> values{
        static_cast<double>(nstate),
        static_cast<double>(nprop),
        static_cast<double>(dims[0]),
        static_cast<double>(dims[1]),
        static_cast<double>(dims[2])};
    std::size_t nrows = 1;
    for (int is = 0; is < nstate; ++is) {
        nrows *= static_cast<std::size_t>(dims[static_cast<std::size_t>(is)]);
    }
    std::vector<std::size_t> order(nrows);
    for (std::size_t i = 0; i < nrows; ++i) {
        order[i] = i;
    }
    if (shuffled) {
        std::reverse(order.begin(), order.end());
    }
    for (std::size_t rlin : order) {
        int rem = static_cast<int>(rlin);
        std::vector<int> idx(static_cast<std::size_t>(nstate), 0);
        for (int is = 0; is < nstate; ++is) {
            idx[static_cast<std::size_t>(is)] =
                rem % dims[static_cast<std::size_t>(is)];
            rem /= dims[static_cast<std::size_t>(is)];
            values.push_back(axes[static_cast<std::size_t>(is)]
                                  [static_cast<std::size_t>(idx[static_cast<std::size_t>(is)])]);
        }
        double sum = 0.0;
        for (int is = 0; is < nstate; ++is) {
            sum += (is + 1) * axes[static_cast<std::size_t>(is)]
                                  [static_cast<std::size_t>(idx[static_cast<std::size_t>(is)])];
        }
        values.push_back(sum);
        if (nprop > 1) {
            values.push_back(1.0 - sum);
        }
    }
    return values;
}

void verifyBasicMesh(const appstruct& app, int nstate, int nprop, int porder)
{
    require(app.materialdb_nstate == nstate, "unexpected nstate");
    require(app.materialdb_nprop == nprop, "unexpected nprop");
    require(app.materialdb_porder == porder, "unexpected selected porder");
    require(app.materialdb_elemtype == 1, "unexpected elemtype");
    int npe = 1;
    for (int is = 0; is < nstate; ++is) {
        npe *= porder + 1;
    }
    require(app.materialdb_npe == npe, "unexpected npe");
    require(app.szmaterialdb_elementcounts == nstate, "element count size mismatch");
    require(app.szmaterialdb_ncgi == nstate, "ncgi size mismatch");
    require(app.szmaterialdb_gridoffset == nstate + 1, "grid offset size mismatch");
    require(app.szmaterialdb_elemoffset == nstate + 1, "element offset size mismatch");
}

void testOneDimensionalPorderSelection(const std::string& tmp)
{
    {
        appstruct app;
        const std::string file = tmp + "/n11.bin";
        writeBin(file, makeDatabase(1, 1, {{0.0, 0.01, 0.04, 0.10, 0.30, 0.55, 0.70, 0.78, 0.86, 0.93, 1.0}}));
        exasim::materials::detail::readMaterialDatabaseIntoAppStruct(file, app);
        verifyBasicMesh(app, 1, 1, 5);
        require(app.materialdb_elementcounts[0] == 2, "N=11 should give two p=5 elements");
        requireClose(app.materialdb_elemcoords[0], 0.0, "bad first element coordinate");
        requireClose(app.materialdb_elemcoords[1], 0.55, "bad middle element coordinate");
        requireClose(app.materialdb_elemcoords[2], 1.0, "bad last element coordinate");
        exasim::materials::detail::releaseAppMaterialDatabase(app);
    }
    {
        appstruct app;
        const std::string file = tmp + "/n13.bin";
        writeBin(file, makeDatabase(1, 1, {{0.0, 0.1, 0.2, 0.4, 0.5, 0.55, 0.6, 0.8, 0.9, 1.3, 1.4, 1.7, 2.0}}));
        exasim::materials::detail::readMaterialDatabaseIntoAppStruct(file, app);
        verifyBasicMesh(app, 1, 1, 4);
        require(app.materialdb_elementcounts[0] == 3, "N=13 should give three p=4 elements");
        exasim::materials::detail::releaseAppMaterialDatabase(app);
    }
    {
        appstruct app;
        const std::string file = tmp + "/n10.bin";
        writeBin(file, makeDatabase(1, 1, {{0.0, 0.02, 0.10, 0.25, 0.60, 0.77, 1.00, 1.40, 2.00, 3.00}}));
        exasim::materials::detail::readMaterialDatabaseIntoAppStruct(file, app);
        verifyBasicMesh(app, 1, 1, 3);
        require(app.materialdb_elementcounts[0] == 3, "N=10 should give three p=3 elements");
        exasim::materials::detail::releaseAppMaterialDatabase(app);
    }
}

void testMultidimensionalAndNonuniform(const std::string& tmp)
{
    appstruct app;
    const std::vector<std::vector<double>> axes{
        {0.0, 0.02, 0.10, 0.25, 0.60, 0.77, 1.00},
        {-2.0, -1.8, -1.7, -0.2, 0.4, 1.1, 3.0}};
    const std::string file = tmp + "/two_dimensional.bin";
    writeBin(file, makeDatabase(2, 2, axes));
    exasim::materials::detail::readMaterialDatabaseIntoAppStruct(file, app);

    verifyBasicMesh(app, 2, 2, 3);
    require(app.materialdb_elementcounts[0] == 2, "bad x element count");
    require(app.materialdb_elementcounts[1] == 2, "bad y element count");
    require(app.materialdb_ne == 4, "bad total element count");
    require(app.materialdb_gridoffset[0] == 0, "bad gridoffset[0]");
    require(app.materialdb_gridoffset[1] == 7, "bad gridoffset[1]");
    require(app.materialdb_gridoffset[2] == 14, "bad gridoffset[2]");
    require(app.materialdb_elemoffset[0] == 0, "bad elemoffset[0]");
    require(app.materialdb_elemoffset[1] == 3, "bad elemoffset[1]");
    require(app.materialdb_elemoffset[2] == 6, "bad elemoffset[2]");
    requireClose(app.materialdb_elemcoords[0], 0.0, "bad x elem 0");
    requireClose(app.materialdb_elemcoords[1], 0.25, "bad x elem 1");
    requireClose(app.materialdb_elemcoords[2], 1.0, "bad x elem 2");
    requireClose(app.materialdb_elemcoords[3], -2.0, "bad y elem 0");
    requireClose(app.materialdb_elemcoords[4], -0.2, "bad y elem 1");
    requireClose(app.materialdb_elemcoords[5], 3.0, "bad y elem 2");

    const int e = 1;
    const int a = 4;
    const int i0 = a % (app.materialdb_porder + 1);
    const int i1 = a / (app.materialdb_porder + 1);
    const double x = axes[0][static_cast<std::size_t>(app.materialdb_porder + i0)];
    const double y = axes[1][static_cast<std::size_t>(i1)];
    requireClose(app.materialdb_statecoords[a + app.materialdb_npe * (0 + app.materialdb_nstate * e)],
                 x,
                 "bad dgnodes/statecoords x");
    requireClose(app.materialdb_statecoords[a + app.materialdb_npe * (1 + app.materialdb_nstate * e)],
                 y,
                 "bad dgnodes/statecoords y");
    requireClose(app.materialdb_propvalues[a + app.materialdb_npe * (0 + app.materialdb_nprop * e)],
                 x + 2.0 * y,
                 "bad propvalue");
    requireClose(app.materialdb_propvalues[a + app.materialdb_npe * (1 + app.materialdb_nprop * e)],
                 1.0 - (x + 2.0 * y),
                 "bad second propvalue");
    exasim::materials::detail::releaseAppMaterialDatabase(app);
}

void testThreeDimensional(const std::string& tmp)
{
    appstruct app;
    const std::string file = tmp + "/three_dimensional.bin";
    writeBin(file, makeDatabase(3, 1, {
        {0.0, 0.3, 1.0},
        {2.0, 2.5, 4.0},
        {-1.0, -0.1, 0.0}}));
    exasim::materials::detail::readMaterialDatabaseIntoAppStruct(file, app);
    verifyBasicMesh(app, 3, 1, 2);
    require(app.materialdb_ne == 1, "3D p=2 test should have one element");
    require(app.materialdb_npe == 27, "3D p=2 should have 27 local nodes");
    exasim::materials::detail::releaseAppMaterialDatabase(app);
}

void testMalformedDuplicate(const std::string& tmp)
{
    const std::string file = tmp + "/duplicate.bin";
    std::vector<double> values = makeDatabase(1, 1, {{0.0, 0.5, 1.0}}, false);
    values[5 + 2] = 0.0; // second row state duplicates first row.
    writeBin(file, values);
    appstruct app;
    bool failed = false;
    try {
        exasim::materials::detail::readMaterialDatabaseIntoAppStruct(file, app);
    } catch (const std::exception&) {
        failed = true;
    }
    require(failed, "duplicate state point should be rejected");
    exasim::materials::detail::releaseAppMaterialDatabase(app);
}

void testReleaseDefaults()
{
    appstruct app;
    exasim::materials::detail::releaseAppMaterialDatabase(app);
    require(app.materialdb_nstate == 0, "release should reset nstate");
    require(app.materialdb_elementcounts == nullptr, "release should null elementcounts");
    require(app.materialdb_statecoords == nullptr, "release should null statecoords");
}

} // namespace

int main(int argc, char** argv)
{
    try {
        require(argc == 2, "usage: appstruct_materialdatabase_test <tmpdir>");
        const std::string tmp = argv[1];
        testReleaseDefaults();
        testOneDimensionalPorderSelection(tmp);
        testMultidimensionalAndNonuniform(tmp);
        testThreeDimensional(tmp);
        testMalformedDuplicate(tmp);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
