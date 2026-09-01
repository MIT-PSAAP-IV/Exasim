#ifndef __APPSTRUCT_MATERIALDATABASE_HPP__
#define __APPSTRUCT_MATERIALDATABASE_HPP__

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <exception>
#include <fstream>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace exasim::materials {
namespace detail {

inline std::runtime_error materialDatabaseError(const std::string& message) {
    return std::runtime_error("materialdatabase.bin: " + message);
}

inline int checkedMaterialDatabaseInt(double value, const std::string& name) {
    if (!std::isfinite(value) || std::nearbyint(value) != value) {
        throw materialDatabaseError(name + " must be finite and integer-valued");
    }
    if (value < static_cast<double>(std::numeric_limits<int>::min()) ||
        value > static_cast<double>(std::numeric_limits<int>::max())) {
        throw materialDatabaseError(name + " is outside int range");
    }
    return static_cast<int>(std::llround(value));
}

inline std::vector<double> readMaterialDatabaseDoubles(const std::string& filename) {
    std::ifstream in(filename.c_str(), std::ios::in | std::ios::binary);
    if (!in) {
        throw materialDatabaseError("could not open " + filename);
    }
    in.seekg(0, std::ios::end);
    const std::streamoff bytes = in.tellg();
    if (bytes < 0 || bytes % static_cast<std::streamoff>(sizeof(double)) != 0) {
        throw materialDatabaseError(filename + " does not contain a whole number of Float64 values");
    }
    in.seekg(0, std::ios::beg);
    std::vector<double> data(static_cast<std::size_t>(bytes / static_cast<std::streamoff>(sizeof(double))));
    if (!data.empty()) {
        in.read(reinterpret_cast<char*>(data.data()), bytes);
    }
    if (!in) {
        throw materialDatabaseError("failed while reading " + filename);
    }
    return data;
}

template <class T, class I>
inline void releaseAppMaterialDatabase(appstructT<T,I>& app, I backend = 1) {
    TemplateFree(app.materialdb_elementcounts, backend);
    TemplateFree(app.materialdb_ncgi, backend);
    TemplateFree(app.materialdb_gridoffset, backend);
    TemplateFree(app.materialdb_elemoffset, backend);
    TemplateFree(app.materialdb_statecoords, backend);
    TemplateFree(app.materialdb_propvalues, backend);
    TemplateFree(app.materialdb_gridcoords, backend);
    TemplateFree(app.materialdb_elemcoords, backend);

    app.materialdb_nstate = 0;
    app.materialdb_nprop = 0;
    app.materialdb_porder = 0;
    app.materialdb_elemtype = 0;
    app.materialdb_npe = 0;
    app.materialdb_ne = 0;

    app.szmaterialdb_elementcounts = 0;
    app.szmaterialdb_ncgi = 0;
    app.szmaterialdb_gridoffset = 0;
    app.szmaterialdb_elemoffset = 0;
    app.szmaterialdb_statecoords = 0;
    app.szmaterialdb_propvalues = 0;
    app.szmaterialdb_gridcoords = 0;
    app.szmaterialdb_elemcoords = 0;
}

inline int tensorNodeCount(int porder, int nstate) {
    int npe = 1;
    for (int is = 0; is < nstate; ++is) {
        npe *= porder + 1;
    }
    return npe;
}

inline std::size_t tensorGridSize(const std::array<int, 3>& dims, int nstate) {
    std::size_t n = 1;
    for (int is = 0; is < nstate; ++is) {
        n *= static_cast<std::size_t>(dims[static_cast<std::size_t>(is)]);
    }
    return n;
}

inline std::size_t gridLinearIndex(const int* multi, const std::array<int, 3>& dims, int nstate) {
    std::size_t index = 0;
    std::size_t stride = 1;
    for (int is = 0; is < nstate; ++is) {
        index += static_cast<std::size_t>(multi[is]) * stride;
        stride *= static_cast<std::size_t>(dims[static_cast<std::size_t>(is)]);
    }
    return index;
}

inline int chooseMaterialDatabasePorder(const std::array<int, 3>& dims, int nstate) {
    for (int is = 0; is < nstate; ++is) {
        if (dims[static_cast<std::size_t>(is)] < 2) {
            throw materialDatabaseError("active state dimensions require at least two grid points");
        }
    }
    for (int p = 5; p >= 1; --p) {
        bool valid = true;
        for (int is = 0; is < nstate; ++is) {
            if ((dims[static_cast<std::size_t>(is)] - 1) % p != 0) {
                valid = false;
                break;
            }
        }
        if (valid) {
            return p;
        }
    }
    throw materialDatabaseError("could not determine a valid polynomial order");
}

template <class T>
inline std::vector<std::vector<T>> sortedMaterialDatabaseAxes(
        const std::vector<T>& rows,
        int nstate,
        int nprop,
        const std::array<int, 3>& dims) {
    const int ncols = nstate + nprop;
    const std::size_t nrows = tensorGridSize(dims, nstate);
    std::vector<std::vector<T>> axes(static_cast<std::size_t>(nstate));
    for (int is = 0; is < nstate; ++is) {
        auto& axis = axes[static_cast<std::size_t>(is)];
        axis.reserve(nrows);
        for (std::size_t r = 0; r < nrows; ++r) {
            axis.push_back(rows[r * static_cast<std::size_t>(ncols) + static_cast<std::size_t>(is)]);
        }
        std::sort(axis.begin(), axis.end());
        axis.erase(std::unique(axis.begin(), axis.end()), axis.end());
        if (axis.size() != static_cast<std::size_t>(dims[static_cast<std::size_t>(is)])) {
            std::ostringstream msg;
            msg << "state dimension " << (is + 1)
                << " has " << axis.size()
                << " unique coordinates, expected "
                << dims[static_cast<std::size_t>(is)];
            throw materialDatabaseError(msg.str());
        }
        for (std::size_t i = 1; i < axis.size(); ++i) {
            if (!(axis[i] > axis[i - 1])) {
                throw materialDatabaseError("state coordinates must be strictly increasing after sorting");
            }
        }
    }
    return axes;
}

template <class T>
inline int axisIndex(const std::vector<T>& axis, T value) {
    const auto it = std::lower_bound(axis.begin(), axis.end(), value);
    if (it == axis.end() || *it != value) {
        throw materialDatabaseError("sample coordinate does not belong to its sorted state axis");
    }
    return static_cast<int>(it - axis.begin());
}

template <class T, class I>
inline void readMaterialDatabaseIntoAppStruct(const std::string& filename, appstructT<T,I>& app) {
    releaseAppMaterialDatabase(app, static_cast<I>(1));

    const std::vector<double> raw = readMaterialDatabaseDoubles(filename);
    if (raw.size() < 5) {
        throw materialDatabaseError("file is too short; expected five-value header");
    }

    const int nstate = checkedMaterialDatabaseInt(raw[0], "nstate");
    const int nprop = checkedMaterialDatabaseInt(raw[1], "nprop");
    const std::array<int, 3> dims{
        checkedMaterialDatabaseInt(raw[2], "n1"),
        checkedMaterialDatabaseInt(raw[3], "n2"),
        checkedMaterialDatabaseInt(raw[4], "n3")};

    if (nstate < 1 || nstate > 3) {
        throw materialDatabaseError("requires 1 <= nstate <= 3");
    }
    if (nprop < 1) {
        throw materialDatabaseError("requires nprop >= 1");
    }
    if (dims[0] <= 0 || dims[1] <= 0 || dims[2] <= 0) {
        throw materialDatabaseError("requires n1,n2,n3 > 0");
    }
    if (nstate == 1 && (dims[1] != 1 || dims[2] != 1)) {
        throw materialDatabaseError("inactive dimensions for nstate=1 require n2=1 and n3=1");
    }
    if (nstate == 2 && dims[2] != 1) {
        throw materialDatabaseError("inactive dimension for nstate=2 requires n3=1");
    }

    const std::size_t nrows = tensorGridSize(dims, nstate);
    const int ncols = nstate + nprop;
    const std::size_t expected = 5 + nrows * static_cast<std::size_t>(ncols);
    if (raw.size() != expected) {
        std::ostringstream msg;
        msg << "contains " << raw.size() << " Float64 values, expected " << expected;
        throw materialDatabaseError(msg.str());
    }

    std::vector<T> rows(raw.size() - 5);
    for (std::size_t i = 5; i < raw.size(); ++i) {
        if (!std::isfinite(raw[i])) {
            throw materialDatabaseError("contains NaN or Inf");
        }
        rows[i - 5] = static_cast<T>(raw[i]);
    }

    const auto axes = sortedMaterialDatabaseAxes(rows, nstate, nprop, dims);
    std::vector<T> gridprops(nrows * static_cast<std::size_t>(nprop));
    std::vector<unsigned char> occupied(nrows, 0);
    int multi[3] = {0, 0, 0};
    for (std::size_t r = 0; r < nrows; ++r) {
        for (int is = 0; is < nstate; ++is) {
            multi[is] = axisIndex(axes[static_cast<std::size_t>(is)],
                rows[r * static_cast<std::size_t>(ncols) + static_cast<std::size_t>(is)]);
        }
        const std::size_t gridIndex = gridLinearIndex(multi, dims, nstate);
        if (occupied[gridIndex] != 0) {
            throw materialDatabaseError("contains duplicated state points");
        }
        occupied[gridIndex] = 1;
        for (int ip = 0; ip < nprop; ++ip) {
            gridprops[gridIndex + nrows * static_cast<std::size_t>(ip)] =
                rows[r * static_cast<std::size_t>(ncols) + static_cast<std::size_t>(nstate + ip)];
        }
    }
    if (std::any_of(occupied.begin(), occupied.end(), [](unsigned char v) { return v == 0; })) {
        throw materialDatabaseError("is missing tensor-product state points");
    }

    const int porder = chooseMaterialDatabasePorder(dims, nstate);
    std::vector<I> elementcounts(static_cast<std::size_t>(nstate));
    int ne = 1;
    for (int is = 0; is < nstate; ++is) {
        elementcounts[static_cast<std::size_t>(is)] =
            static_cast<I>((dims[static_cast<std::size_t>(is)] - 1) / porder);
        ne *= static_cast<int>(elementcounts[static_cast<std::size_t>(is)]);
    }
    const int npe = tensorNodeCount(porder, nstate);

    app.materialdb_nstate = static_cast<I>(nstate);
    app.materialdb_nprop = static_cast<I>(nprop);
    app.materialdb_porder = static_cast<I>(porder);
    app.materialdb_elemtype = static_cast<I>(1);
    app.materialdb_npe = static_cast<I>(npe);
    app.materialdb_ne = static_cast<I>(ne);

    app.szmaterialdb_elementcounts = static_cast<I>(nstate);
    app.szmaterialdb_ncgi = static_cast<I>(nstate);
    app.szmaterialdb_gridoffset = static_cast<I>(nstate + 1);
    app.szmaterialdb_elemoffset = static_cast<I>(nstate + 1);

    I gridcoordsize = 0;
    I elemcoordsize = 0;
    for (int is = 0; is < nstate; ++is) {
        gridcoordsize += static_cast<I>(dims[static_cast<std::size_t>(is)]);
        elemcoordsize += elementcounts[static_cast<std::size_t>(is)] + static_cast<I>(1);
    }
    app.szmaterialdb_gridcoords = gridcoordsize;
    app.szmaterialdb_elemcoords = elemcoordsize;
    app.szmaterialdb_statecoords = static_cast<I>(npe * nstate * ne);
    app.szmaterialdb_propvalues = static_cast<I>(npe * nprop * ne);

    try {
        TemplateMalloc(&app.materialdb_elementcounts, app.szmaterialdb_elementcounts, 1);
        TemplateMalloc(&app.materialdb_ncgi, app.szmaterialdb_ncgi, 1);
        TemplateMalloc(&app.materialdb_gridoffset, app.szmaterialdb_gridoffset, 1);
        TemplateMalloc(&app.materialdb_elemoffset, app.szmaterialdb_elemoffset, 1);
        TemplateMalloc(&app.materialdb_gridcoords, app.szmaterialdb_gridcoords, 1);
        TemplateMalloc(&app.materialdb_elemcoords, app.szmaterialdb_elemcoords, 1);
        TemplateMalloc(&app.materialdb_statecoords, app.szmaterialdb_statecoords, 1);
        TemplateMalloc(&app.materialdb_propvalues, app.szmaterialdb_propvalues, 1);

        app.materialdb_gridoffset[0] = 0;
        app.materialdb_elemoffset[0] = 0;
        for (int is = 0; is < nstate; ++is) {
            app.materialdb_elementcounts[is] = elementcounts[static_cast<std::size_t>(is)];
            app.materialdb_ncgi[is] = static_cast<I>(dims[static_cast<std::size_t>(is)]);
            app.materialdb_gridoffset[is + 1] =
                app.materialdb_gridoffset[is] + app.materialdb_ncgi[is];
            app.materialdb_elemoffset[is + 1] =
                app.materialdb_elemoffset[is] + app.materialdb_elementcounts[is] + static_cast<I>(1);
        }

        for (int is = 0; is < nstate; ++is) {
            const auto& axis = axes[static_cast<std::size_t>(is)];
            for (int j = 0; j < dims[static_cast<std::size_t>(is)]; ++j) {
                app.materialdb_gridcoords[app.materialdb_gridoffset[is] + j] =
                    axis[static_cast<std::size_t>(j)];
            }
            for (I ie = 0; ie <= app.materialdb_elementcounts[is]; ++ie) {
                app.materialdb_elemcoords[app.materialdb_elemoffset[is] + ie] =
                    axis[static_cast<std::size_t>(ie * app.materialdb_porder)];
            }
        }

        std::fill_n(app.materialdb_statecoords,
            static_cast<std::size_t>(app.szmaterialdb_statecoords),
            static_cast<T>(0));
        std::fill_n(app.materialdb_propvalues,
            static_cast<std::size_t>(app.szmaterialdb_propvalues),
            static_cast<T>(0));

        int elemMulti[3] = {0, 0, 0};
        int localMulti[3] = {0, 0, 0};
        int gridMulti[3] = {0, 0, 0};
        for (int e = 0; e < ne; ++e) {
            int rem = e;
            for (int is = 0; is < nstate; ++is) {
                elemMulti[is] = rem % static_cast<int>(elementcounts[static_cast<std::size_t>(is)]);
                rem /= static_cast<int>(elementcounts[static_cast<std::size_t>(is)]);
            }
            for (int a = 0; a < npe; ++a) {
                rem = a;
                for (int is = 0; is < nstate; ++is) {
                    localMulti[is] = rem % (porder + 1);
                    rem /= porder + 1;
                    gridMulti[is] = elemMulti[is] * porder + localMulti[is];
                    app.materialdb_statecoords[
                        a + npe * (is + nstate * e)] =
                        axes[static_cast<std::size_t>(is)][static_cast<std::size_t>(gridMulti[is])];
                }
                const std::size_t gridIndex = gridLinearIndex(gridMulti, dims, nstate);
                for (int ip = 0; ip < nprop; ++ip) {
                    app.materialdb_propvalues[
                        a + npe * (ip + nprop * e)] =
                        gridprops[gridIndex + nrows * static_cast<std::size_t>(ip)];
                }
            }
        }
    } catch (...) {
        releaseAppMaterialDatabase(app, static_cast<I>(1));
        throw;
    }
}

} // namespace detail
} // namespace exasim::materials

#endif
