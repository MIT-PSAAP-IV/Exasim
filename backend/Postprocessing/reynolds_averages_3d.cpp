#include "reynolds_averages_3d.hpp"

#include "solution_io_lib.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {

constexpr int kNumAverages = 30;

struct SnapshotMetadata {
    int npe = 0;
    int nc = 0;
    int ne = 0;
    std::size_t snapshotSize = 0;
    std::int64_t nsnapshots = 0;
};

std::size_t checkedProduct(std::size_t a, std::size_t b, const std::string& label)
{
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
        throw std::runtime_error("Overflow while computing " + label + ".");
    }
    return a * b;
}

int checkedPositiveIntegerHeader(double value,
                                 const std::string& name,
                                 const std::string& fname)
{
    if (!std::isfinite(value)) {
        throw std::runtime_error("Header " + name + " is not finite in file: " + fname);
    }

    double intpart = 0.0;
    if (std::modf(value, &intpart) != 0.0) {
        throw std::runtime_error("Header " + name + " is not an integer in file: " + fname);
    }
    if (intpart > static_cast<double>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("Header " + name + " exceeds int range in file: " + fname);
    }

    const int ivalue = static_cast<int>(intpart);
    if (ivalue <= 0) {
        throw std::runtime_error("Header " + name + " must be positive in file: " + fname);
    }
    return ivalue;
}

std::size_t stateIndex(int i, int c, int e, int npe, int nc)
{
    return static_cast<std::size_t>(i) +
           static_cast<std::size_t>(npe) *
               (static_cast<std::size_t>(c) +
                static_cast<std::size_t>(nc) * static_cast<std::size_t>(e));
}

std::size_t averageIndex(int i, int c, int e, int npe)
{
    return static_cast<std::size_t>(i) +
           static_cast<std::size_t>(npe) *
               (static_cast<std::size_t>(c) +
                static_cast<std::size_t>(kNumAverages) * static_cast<std::size_t>(e));
}

[[noreturn]] void invalidDensity(const std::string& fname,
                                 int snapshot,
                                 int point,
                                 int elem,
                                 double rho)
{
    std::ostringstream oss;
    oss << "Invalid density rho=" << rho
        << " at snapshot " << snapshot
        << ", point " << point
        << ", element " << elem
        << " in file: " << fname;
    throw std::runtime_error(oss.str());
}

void validateDistinctFiles(const std::string& fileout, const std::string& filein)
{
    const std::filesystem::path outPath = std::filesystem::absolute(fileout).lexically_normal();
    const std::filesystem::path inPath = std::filesystem::absolute(filein).lexically_normal();
    if (outPath == inPath) {
        throw std::runtime_error("Output file must differ from input file: " + filein);
    }
}

SnapshotMetadata readSnapshotMetadata(std::ifstream& in, const std::string& filein)
{
    const std::int64_t bytes = fileSizeBytes(filein);
    if (bytes < static_cast<std::int64_t>(3 * sizeof(double))) {
        throw std::runtime_error("File too small for 3-value header: " + filein);
    }
    if (bytes % static_cast<std::int64_t>(sizeof(double)) != 0) {
        throw std::runtime_error("File size not multiple of 8 in: " + filein);
    }

    double header[3] = {0.0, 0.0, 0.0};
    readDoubles(in, header, 3, filein);

    SnapshotMetadata meta;
    meta.npe = checkedPositiveIntegerHeader(header[0], "npe", filein);
    meta.nc = checkedPositiveIntegerHeader(header[1], "nc", filein);
    meta.ne = checkedPositiveIntegerHeader(header[2], "ne", filein);
    if (meta.nc < 5) {
        throw std::runtime_error("Header nc must satisfy nc >= 5 in file: " + filein);
    }

    meta.snapshotSize =
        checkedProduct(checkedProduct(static_cast<std::size_t>(meta.npe),
                                      static_cast<std::size_t>(meta.nc),
                                      "npe*nc"),
                       static_cast<std::size_t>(meta.ne),
                       "npe*nc*ne");
    const std::int64_t ndoubles = bytes / static_cast<std::int64_t>(sizeof(double));
    const std::int64_t payloadDoubles = ndoubles - 3;
    if (payloadDoubles < 0) {
        throw std::runtime_error("Negative payload size in file: " + filein);
    }
    if (meta.snapshotSize == 0) {
        throw std::runtime_error("Snapshot size is zero in file: " + filein);
    }
    if (payloadDoubles % static_cast<std::int64_t>(meta.snapshotSize) != 0) {
        throw std::runtime_error("Payload does not contain complete snapshots in file: " + filein);
    }

    meta.nsnapshots = payloadDoubles / static_cast<std::int64_t>(meta.snapshotSize);
    return meta;
}

SnapshotMetadata readSnapshotMetadata(const std::string& filein)
{
    std::ifstream in(filein, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Cannot open file: " + filein);
    }
    return readSnapshotMetadata(in, filein);
}

} // namespace

void ReynoldsAverages3D(const std::string& fileout,
                        const std::string& filein,
                        int nsteps,
                        int stepoffsets,
                        double gamma)
{
    validateDistinctFiles(fileout, filein);

    if (gamma <= 1.0) {
        throw std::runtime_error("gamma must be greater than 1.");
    }
    if (nsteps <= 0) {
        throw std::runtime_error("nsteps must be positive.");
    }
    if (stepoffsets < 0) {
        throw std::runtime_error("stepoffsets must be nonnegative.");
    }

    std::ifstream in(filein, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Cannot open file: " + filein);
    }

    const SnapshotMetadata meta = readSnapshotMetadata(in, filein);
    const int npe = meta.npe;
    const int nc = meta.nc;
    const int ne = meta.ne;
    const std::size_t snapshotSize = meta.snapshotSize;
    const std::int64_t nsnapshots = meta.nsnapshots;
    if (static_cast<std::int64_t>(stepoffsets) >= nsnapshots) {
        std::ostringstream oss;
        oss << "Requested starting snapshot " << stepoffsets
            << " is outside the available snapshot range [0, "
            << (nsnapshots - 1) << "] in file: " << filein;
        throw std::runtime_error(oss.str());
    }

    const std::int64_t requestedEnd =
        static_cast<std::int64_t>(stepoffsets) + static_cast<std::int64_t>(nsteps);
    const int effectiveNsteps =
        static_cast<int>(std::min<std::int64_t>(static_cast<std::int64_t>(nsteps),
                                               nsnapshots - static_cast<std::int64_t>(stepoffsets)));
    if (effectiveNsteps <= 0) {
        std::ostringstream oss;
        oss << "No snapshots available for requested interval [" << stepoffsets << ", "
            << (requestedEnd - 1) << "] with available snapshot count "
            << nsnapshots << " in file: " << filein;
        throw std::runtime_error(oss.str());
    }

    const std::streamoff startOffset =
        static_cast<std::streamoff>(3 * sizeof(double)) +
        static_cast<std::streamoff>(stepoffsets) *
            static_cast<std::streamoff>(snapshotSize * sizeof(double));
    in.seekg(startOffset, std::ios::beg);
    if (!in) {
        throw std::runtime_error("seekg failed in: " + filein);
    }

    std::vector<double> snapshot(snapshotSize);
    const std::size_t avgSize =
        checkedProduct(checkedProduct(static_cast<std::size_t>(npe),
                                      static_cast<std::size_t>(kNumAverages),
                                      "npe*30"),
                       static_cast<std::size_t>(ne),
                       "npe*30*ne");
    std::vector<double> averages(avgSize, 0.0);

    for (int s = 0; s < effectiveNsteps; ++s) {
        const int snapshotIndex = stepoffsets + s;
        readDoubles(in, snapshot.data(), snapshotSize, filein);

        for (int e = 0; e < ne; ++e) {
            for (int i = 0; i < npe; ++i) {
                const double rho = snapshot[stateIndex(i, 0, e, npe, nc)];
                if (rho <= 0.0) {
                    invalidDensity(filein, snapshotIndex, i, e, rho);
                }

                const double rhou = snapshot[stateIndex(i, 1, e, npe, nc)];
                const double rhov = snapshot[stateIndex(i, 2, e, npe, nc)];
                const double rhow = snapshot[stateIndex(i, 3, e, npe, nc)];
                const double rhoE = snapshot[stateIndex(i, 4, e, npe, nc)];

                const double u = rhou / rho;
                const double v = rhov / rho;
                const double w = rhow / rho;
                const double kinetic = 0.5 * (rhou * u + rhov * v + rhow * w);
                const double p = (gamma - 1.0) * (rhoE - kinetic);
                const double T = p / ((gamma - 1.0) * rho);

                const double values[kNumAverages] = {
                    rho,
                    rhou,
                    rhov,
                    rhow,
                    rhoE,
                    u,
                    v,
                    w,
                    p,
                    T,
                    rhou * rhou / rho,
                    rhov * rhov / rho,
                    rhow * rhow / rho,
                    rhou * rhov / rho,
                    rhou * rhow / rho,
                    rhov * rhow / rho,
                    u * u,
                    v * v,
                    w * w,
                    u * v,
                    u * w,
                    v * w,
                    rho * rho,
                    p * p,
                    T * T,
                    rho * T,
                    rho * T * T,
                    rhou * T,
                    rhov * T,
                    rhow * T
                };

                for (int c = 0; c < kNumAverages; ++c) {
                    averages[averageIndex(i, c, e, npe)] += values[c];
                }
            }
        }
    }

    const double scale = 1.0 / static_cast<double>(effectiveNsteps);
    for (double& value : averages) {
        value *= scale;
    }

    std::ofstream out(fileout, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Unable to open file " + fileout);
    }

    const double outHeader[3] = {
        static_cast<double>(npe),
        static_cast<double>(kNumAverages),
        static_cast<double>(ne)
    };
    writeDoubles(out, outHeader, 3, fileout);
    writeDoubles(out, averages.data(), averages.size(), fileout);
}

void ReynoldsAverages3D(const std::string& fileout,
                        const std::string& filein,
                        int nsteps,
                        double gamma)
{
    if (nsteps <= 0) {
        throw std::runtime_error("nsteps must be positive.");
    }

    const SnapshotMetadata meta = readSnapshotMetadata(filein);
    const std::int64_t stepoffsets64 = meta.nsnapshots - static_cast<std::int64_t>(nsteps);
    if (stepoffsets64 < 0) {
        std::ostringstream oss;
        oss << "nsteps (" << nsteps << ") exceeds available snapshots ("
            << meta.nsnapshots << ") in file: " << filein;
        throw std::runtime_error(oss.str());
    }
    if (stepoffsets64 > static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("Computed stepoffsets exceeds int range for file: " + filein);
    }

    ReynoldsAverages3D(fileout, filein, nsteps, static_cast<int>(stepoffsets64), gamma);
}
