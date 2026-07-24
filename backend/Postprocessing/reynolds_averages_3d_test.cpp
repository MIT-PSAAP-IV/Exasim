#include "reynolds_averages_3d.hpp"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kNumAverages = 30;

std::size_t stateIndex(int i, int c, int e, int npe, int nc)
{
    return static_cast<std::size_t>(i) +
           static_cast<std::size_t>(npe) *
               (static_cast<std::size_t>(c) +
                static_cast<std::size_t>(nc) * static_cast<std::size_t>(e));
}

std::size_t avgIndex(int i, int c, int e, int npe)
{
    return static_cast<std::size_t>(i) +
           static_cast<std::size_t>(npe) *
               (static_cast<std::size_t>(c) +
                static_cast<std::size_t>(kNumAverages) * static_cast<std::size_t>(e));
}

void writeBinary(const std::filesystem::path& path, const std::vector<double>& values)
{
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Unable to open file " + path.string());
    }
    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(double)));
    if (!out) {
        throw std::runtime_error("Failed to write file " + path.string());
    }
}

std::vector<double> readBinary(const std::filesystem::path& path)
{
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        throw std::runtime_error("Unable to open file " + path.string());
    }
    const std::streamoff size = in.tellg();
    if (size < 0 || size % static_cast<std::streamoff>(sizeof(double)) != 0) {
        throw std::runtime_error("Invalid file size for " + path.string());
    }
    in.seekg(0, std::ios::beg);
    std::vector<double> values(static_cast<std::size_t>(size / sizeof(double)));
    in.read(reinterpret_cast<char*>(values.data()), size);
    if (!in) {
        throw std::runtime_error("Failed to read file " + path.string());
    }
    return values;
}

bool nearlyEqual(double a, double b, double tol = 1.0e-12)
{
    return std::abs(a - b) <= tol * std::max(1.0, std::max(std::abs(a), std::abs(b)));
}

void expect(bool cond, const std::string& message)
{
    if (!cond) {
        throw std::runtime_error(message);
    }
}

void expectClose(double actual, double expected, const std::string& label)
{
    if (!nearlyEqual(actual, expected)) {
        throw std::runtime_error(label + ": expected " + std::to_string(expected) +
                                 ", got " + std::to_string(actual));
    }
}

void expectThrowsContains(const std::string& label,
                          const std::string& needle,
                          const std::function<void()>& fn)
{
    try {
        fn();
    } catch (const std::runtime_error& e) {
        const std::string what = e.what();
        if (what.find(needle) == std::string::npos) {
            throw std::runtime_error(label + ": wrong error message: " + what);
        }
        return;
    }
    throw std::runtime_error(label + ": expected runtime_error");
}

struct SnapshotCase {
    int npe;
    int nc;
    int ne;
    std::vector<std::vector<double>> snapshots;
};

SnapshotCase makeCase()
{
    SnapshotCase c;
    c.npe = 2;
    c.nc = 7;
    c.ne = 2;
    c.snapshots.resize(4);
    for (std::vector<double>& snap : c.snapshots) {
        snap.assign(static_cast<std::size_t>(c.npe * c.nc * c.ne), 0.0);
    }

    for (int s = 0; s < 4; ++s) {
        for (int e = 0; e < c.ne; ++e) {
            for (int i = 0; i < c.npe; ++i) {
                const double rho = 2.0 + 0.5 * static_cast<double>(s) +
                                   0.25 * static_cast<double>(e) +
                                   0.1 * static_cast<double>(i);
                const double u = 1.1 + 0.2 * static_cast<double>(s) +
                                 0.05 * static_cast<double>(e) -
                                 0.03 * static_cast<double>(i);
                const double v = -0.3 + 0.1 * static_cast<double>(s) +
                                 0.04 * static_cast<double>(e) +
                                 0.02 * static_cast<double>(i);
                const double w = 0.7 - 0.05 * static_cast<double>(s) +
                                 0.03 * static_cast<double>(e) +
                                 0.01 * static_cast<double>(i);
                const double p = 1.5 + 0.3 * static_cast<double>(s) +
                                 0.1 * static_cast<double>(e) +
                                 0.05 * static_cast<double>(i);
                const double gamma = 1.4;
                const double rhoE = p / (gamma - 1.0) +
                                    0.5 * rho * (u * u + v * v + w * w);

                c.snapshots[s][stateIndex(i, 0, e, c.npe, c.nc)] = rho;
                c.snapshots[s][stateIndex(i, 1, e, c.npe, c.nc)] = rho * u;
                c.snapshots[s][stateIndex(i, 2, e, c.npe, c.nc)] = rho * v;
                c.snapshots[s][stateIndex(i, 3, e, c.npe, c.nc)] = rho * w;
                c.snapshots[s][stateIndex(i, 4, e, c.npe, c.nc)] = rhoE;
                c.snapshots[s][stateIndex(i, 5, e, c.npe, c.nc)] = 1000.0 + 17.0 * s + 3.0 * e + i;
                c.snapshots[s][stateIndex(i, 6, e, c.npe, c.nc)] = -500.0 - 11.0 * s - 5.0 * e - i;
            }
        }
    }
    return c;
}

std::vector<double> serializeCase(const SnapshotCase& c)
{
    std::vector<double> data;
    data.reserve(3 + c.snapshots.size() * c.snapshots[0].size());
    data.push_back(static_cast<double>(c.npe));
    data.push_back(static_cast<double>(c.nc));
    data.push_back(static_cast<double>(c.ne));
    for (const std::vector<double>& snap : c.snapshots) {
        data.insert(data.end(), snap.begin(), snap.end());
    }
    return data;
}

std::vector<double> expectedAverages(const SnapshotCase& c,
                                     int stepoffsets,
                                     int nsteps,
                                     double gamma)
{
    std::vector<double> averages(static_cast<std::size_t>(c.npe * kNumAverages * c.ne), 0.0);
    for (int s = stepoffsets; s < stepoffsets + nsteps; ++s) {
        const std::vector<double>& snap = c.snapshots[static_cast<std::size_t>(s)];
        for (int e = 0; e < c.ne; ++e) {
            for (int i = 0; i < c.npe; ++i) {
                const double rho = snap[stateIndex(i, 0, e, c.npe, c.nc)];
                const double rhou = snap[stateIndex(i, 1, e, c.npe, c.nc)];
                const double rhov = snap[stateIndex(i, 2, e, c.npe, c.nc)];
                const double rhow = snap[stateIndex(i, 3, e, c.npe, c.nc)];
                const double rhoE = snap[stateIndex(i, 4, e, c.npe, c.nc)];
                const double u = rhou / rho;
                const double v = rhov / rho;
                const double w = rhow / rho;
                const double kinetic = 0.5 * (rhou * u + rhov * v + rhow * w);
                const double p = (gamma - 1.0) * (rhoE - kinetic);
                const double T = p / ((gamma - 1.0) * rho);

                const double vals[kNumAverages] = {
                    rho, rhou, rhov, rhow, rhoE, u, v, w, p, T,
                    rhou * rhou / rho,
                    rhov * rhov / rho,
                    rhow * rhow / rho,
                    rhou * rhov / rho,
                    rhou * rhow / rho,
                    rhov * rhow / rho,
                    u * u, v * v, w * w,
                    u * v, u * w, v * w,
                    rho * rho, p * p, T * T,
                    rho * T, rho * T * T,
                    rhou * T, rhov * T, rhow * T
                };

                for (int cidx = 0; cidx < kNumAverages; ++cidx) {
                    averages[avgIndex(i, cidx, e, c.npe)] += vals[cidx];
                }
            }
        }
    }

    const double scale = 1.0 / static_cast<double>(nsteps);
    for (double& value : averages) {
        value *= scale;
    }
    return averages;
}

void verifyOutput(const std::filesystem::path& output,
                  const std::vector<double>& expected,
                  int npe,
                  int ne)
{
    const std::vector<double> values = readBinary(output);
    expect(values.size() == static_cast<std::size_t>(3 + npe * kNumAverages * ne),
           "Unexpected output size.");
    expectClose(values[0], static_cast<double>(npe), "output header npe");
    expectClose(values[1], static_cast<double>(kNumAverages), "output header nc");
    expectClose(values[2], static_cast<double>(ne), "output header ne");
    for (std::size_t i = 0; i < expected.size(); ++i) {
        expectClose(values[3 + i], expected[i], "output payload " + std::to_string(i));
    }
}

void testFileFormatAndVaryingState(const std::filesystem::path& dir)
{
    const double gamma = 1.4;
    const SnapshotCase c = makeCase();
    const std::filesystem::path input = dir / "varying_input.bin";
    const std::filesystem::path output = dir / "varying_output.bin";
    writeBinary(input, serializeCase(c));

    ReynoldsAverages3D(output.string(), input.string(), 4, 0, gamma);

    verifyOutput(output, expectedAverages(c, 0, 4, gamma), c.npe, c.ne);

    const std::uintmax_t expectedBytes =
        static_cast<std::uintmax_t>(3 + c.npe * kNumAverages * c.ne) * sizeof(double);
    expect(std::filesystem::file_size(output) == expectedBytes, "Unexpected output file size.");
}

void testConstantState(const std::filesystem::path& dir)
{
    const double gamma = 1.4;
    SnapshotCase c;
    c.npe = 2;
    c.nc = 7;
    c.ne = 2;
    c.snapshots.resize(3);
    const double rho = 3.0;
    const double u = 2.0;
    const double v = -1.0;
    const double w = 0.5;
    const double p = 2.4;
    const double rhoE = p / (gamma - 1.0) + 0.5 * rho * (u * u + v * v + w * w);
    for (std::vector<double>& snap : c.snapshots) {
        snap.assign(static_cast<std::size_t>(c.npe * c.nc * c.ne), 0.0);
        for (int e = 0; e < c.ne; ++e) {
            for (int i = 0; i < c.npe; ++i) {
                snap[stateIndex(i, 0, e, c.npe, c.nc)] = rho;
                snap[stateIndex(i, 1, e, c.npe, c.nc)] = rho * u;
                snap[stateIndex(i, 2, e, c.npe, c.nc)] = rho * v;
                snap[stateIndex(i, 3, e, c.npe, c.nc)] = rho * w;
                snap[stateIndex(i, 4, e, c.npe, c.nc)] = rhoE;
                snap[stateIndex(i, 5, e, c.npe, c.nc)] = 9999.0;
                snap[stateIndex(i, 6, e, c.npe, c.nc)] = -9999.0;
            }
        }
    }

    const std::filesystem::path input = dir / "constant_input.bin";
    const std::filesystem::path output = dir / "constant_output.bin";
    writeBinary(input, serializeCase(c));
    ReynoldsAverages3D(output.string(), input.string(), 3, 0, gamma);
    verifyOutput(output, expectedAverages(c, 0, 3, gamma), c.npe, c.ne);
}

void testOffsetAndExtraComponentIgnore(const std::filesystem::path& dir)
{
    const double gamma = 1.4;
    SnapshotCase c = makeCase();
    const std::filesystem::path inputA = dir / "offset_input_a.bin";
    const std::filesystem::path inputB = dir / "offset_input_b.bin";
    const std::filesystem::path outputA = dir / "offset_output_a.bin";
    const std::filesystem::path outputB = dir / "offset_output_b.bin";

    writeBinary(inputA, serializeCase(c));

    SnapshotCase changed = c;
    for (std::vector<double>& snap : changed.snapshots) {
        for (int e = 0; e < changed.ne; ++e) {
            for (int i = 0; i < changed.npe; ++i) {
                snap[stateIndex(i, 5, e, changed.npe, changed.nc)] *= -123.0;
                snap[stateIndex(i, 6, e, changed.npe, changed.nc)] += 4567.0;
            }
        }
    }
    writeBinary(inputB, serializeCase(changed));

    ReynoldsAverages3D(outputA.string(), inputA.string(), 2, 1, gamma);
    ReynoldsAverages3D(outputB.string(), inputB.string(), 2, 1, gamma);

    const std::vector<double> expected = expectedAverages(c, 1, 2, gamma);
    verifyOutput(outputA, expected, c.npe, c.ne);
    verifyOutput(outputB, expected, c.npe, c.ne);
}

void testValidationErrors(const std::filesystem::path& dir)
{
    const double gamma = 1.4;
    const SnapshotCase c = makeCase();
    const std::filesystem::path goodInput = dir / "good_input.bin";
    writeBinary(goodInput, serializeCase(c));

    expectThrowsContains("missing input", "Cannot open file", [&]() {
        ReynoldsAverages3D((dir / "out.bin").string(), (dir / "missing.bin").string(), 1, 0, gamma);
    });

    writeBinary(dir / "empty.bin", {});
    expectThrowsContains("empty file", "File too small", [&]() {
        ReynoldsAverages3D((dir / "out_empty.bin").string(), (dir / "empty.bin").string(), 1, 0, gamma);
    });

    writeBinary(dir / "short.bin", {1.0, 2.0});
    expectThrowsContains("short header", "File too small", [&]() {
        ReynoldsAverages3D((dir / "out_short.bin").string(), (dir / "short.bin").string(), 1, 0, gamma);
    });

    writeBinary(dir / "nonintegral.bin", {2.5, 7.0, 2.0});
    expectThrowsContains("nonintegral header", "not an integer", [&]() {
        ReynoldsAverages3D((dir / "out_nonintegral.bin").string(), (dir / "nonintegral.bin").string(), 1, 0, gamma);
    });

    writeBinary(dir / "bad_npe.bin", {0.0, 7.0, 2.0});
    expectThrowsContains("bad npe", "must be positive", [&]() {
        ReynoldsAverages3D((dir / "out_bad_npe.bin").string(), (dir / "bad_npe.bin").string(), 1, 0, gamma);
    });

    writeBinary(dir / "bad_nc.bin", {2.0, 4.0, 2.0, 1.0});
    expectThrowsContains("bad nc", "nc >= 5", [&]() {
        ReynoldsAverages3D((dir / "out_bad_nc.bin").string(), (dir / "bad_nc.bin").string(), 1, 0, gamma);
    });

    writeBinary(dir / "bad_ne.bin", {2.0, 7.0, 0.0});
    expectThrowsContains("bad ne", "must be positive", [&]() {
        ReynoldsAverages3D((dir / "out_bad_ne.bin").string(), (dir / "bad_ne.bin").string(), 1, 0, gamma);
    });

    std::vector<double> incomplete = serializeCase(c);
    incomplete.pop_back();
    writeBinary(dir / "incomplete.bin", incomplete);
    expectThrowsContains("incomplete payload", "complete snapshots", [&]() {
        ReynoldsAverages3D((dir / "out_incomplete.bin").string(), (dir / "incomplete.bin").string(), 1, 0, gamma);
    });

    expectThrowsContains("nsteps <= 0", "nsteps must be positive", [&]() {
        ReynoldsAverages3D((dir / "out_nsteps.bin").string(), goodInput.string(), 0, 0, gamma);
    });

    expectThrowsContains("negative offset", "stepoffsets must be nonnegative", [&]() {
        ReynoldsAverages3D((dir / "out_offset.bin").string(), goodInput.string(), 1, -1, gamma);
    });

    const std::filesystem::path truncatedOutput = dir / "out_range_truncated.bin";
    ReynoldsAverages3D(truncatedOutput.string(), goodInput.string(), 5, 0, gamma);
    verifyOutput(truncatedOutput, expectedAverages(c, 0, 4, gamma), c.npe, c.ne);

    const std::filesystem::path tailOutput = dir / "out_tail_truncated.bin";
    ReynoldsAverages3D(tailOutput.string(), goodInput.string(), 5, 2, gamma);
    verifyOutput(tailOutput, expectedAverages(c, 2, 2, gamma), c.npe, c.ne);

    const std::filesystem::path lastStepsOutput = dir / "out_last_steps.bin";
    ReynoldsAverages3D(lastStepsOutput.string(), goodInput.string(), 2, gamma);
    verifyOutput(lastStepsOutput, expectedAverages(c, 2, 2, gamma), c.npe, c.ne);

    expectThrowsContains("last nsteps too large", "exceeds available snapshots", [&]() {
        ReynoldsAverages3D((dir / "out_last_too_large.bin").string(), goodInput.string(), 5, gamma);
    });

    expectThrowsContains("offset starts past data", "outside the available snapshot range", [&]() {
        ReynoldsAverages3D((dir / "out_range.bin").string(), goodInput.string(), 1, 4, gamma);
    });

    expectThrowsContains("invalid gamma", "gamma must be greater than 1", [&]() {
        ReynoldsAverages3D((dir / "out_gamma.bin").string(), goodInput.string(), 1, 0, 1.0);
    });

    SnapshotCase invalid = c;
    invalid.snapshots[2][stateIndex(1, 0, 1, invalid.npe, invalid.nc)] = 0.0;
    writeBinary(dir / "invalid_density.bin", serializeCase(invalid));
    expectThrowsContains("invalid density", "Invalid density", [&]() {
        ReynoldsAverages3D((dir / "out_invalid_density.bin").string(),
                           (dir / "invalid_density.bin").string(),
                           4, 0, gamma);
    });

    const std::filesystem::path missingDirOut = dir / "missing_dir" / "out.bin";
    expectThrowsContains("output creation failure", "Unable to open file", [&]() {
        ReynoldsAverages3D(missingDirOut.string(), goodInput.string(), 1, 0, gamma);
    });

    expectThrowsContains("identical paths", "must differ from input", [&]() {
        ReynoldsAverages3D(goodInput.string(), goodInput.string(), 1, 0, gamma);
    });
}

} // namespace

int main()
{
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / "exasim_reynolds_averages_3d_test";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);

    try {
        testFileFormatAndVaryingState(dir);
        testConstantState(dir);
        testOffsetAndExtraComponentIgnore(dir);
        testValidationErrors(dir);
    } catch (...) {
        std::filesystem::remove_all(dir);
        throw;
    }

    std::filesystem::remove_all(dir);
    return 0;
}
