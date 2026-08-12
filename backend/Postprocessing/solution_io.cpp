// c++ -std=c++17 -Wall -Wextra -pedantic -O3 solution_io.cpp -o solution_io
// Example:
//   ./solution_io dataout/outudg datain/mesh sol2davg 736 --average2d 9 18
//   ./solution_io dataout/outxg xg 4
//   ./solution_io dataout/outuf uf 4 --getufavg 9 8
//   ./solution_io dataout/outudg udgf 4 --getudgf 2 10
//   ./solution_io dataout/outudg udgavg 4 --averageudgf 10 20
//   ./solution_io dataout/outudg datain/mesh udg 4 --getudg 9
//   ./solution_io dataout/outudg datain/mesh qcrit 4 --qcriterion 9
//     writes fields: qcrit, u, pressure, Mach, |grad rho|
//   ./solution_io dataout/outudgavg datain/mesh udgavg 4 --getudgavg 9
//   ./solution_io dataout/outudgavg datain/mesh udg2davg 4 --averageudgavg 27 9 80

// IOANDES=/lustre/orion/ard196/proj-shared/Exasim/backend/Postprocessing/ioandes
// input="case${caseid}/dataout/outbouxdg"
// output="outbou/case${caseid}/outbouxdg"
// "$IOANDES" "$input" "$output" 736

// input="case${caseid}/dataout/outboundg"
// output="outbou/case${caseid}/outboundg"
// "$IOANDES" "$input" "$output" 736

// input="case${caseid}/dataout/outbouuhavg"
// output="outbou/case${caseid}/outbouuhavg"
// "$IOANDES" "$input" "$output" 736 --getufavg 9 5

// input="case${caseid}/dataout/outbouudgavg"
// output="outbou/case${caseid}/outbouudgavg"
// "$IOANDES" "$input" "$output" 736 --getufavg 9 20

// input="case${caseid}/dataout/outbouuhat"
// output="outbou/case${caseid}/outbouuhmean"
// "$IOANDES" "$input" "$output" 736 --averageudgf 500 1500

// input="case${caseid}/dataout/outbouudg"
// output="outbou/case${caseid}/outbouudgmean"
// "$IOANDES" "$input" "$output" 736 --averageudgf 500 1500


#include <cstdlib>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "solution_io_lib.cpp"

namespace {

void printUsage(const char* program)
{
    std::cerr
        << "Usage:\n"
        << "  " << program
        << " <in_base> <out_base> <nprocs>\n"
        << "  " << program
        << " <in_base> <out_base> <nprocs> --getufavg <npf> <ncu>\n"
        << "  " << program
        << " <in_base> <out_base> <nprocs> --getudgf [nsteps] [stepoffsets]\n"
        << "  " << program
        << " <in_base> <out_base> <nprocs> --averageudgf [nsteps] [stepoffsets]\n"
        << "  " << program
        << " <sol_base> <elempart_base> <out_base> <nprocs> [nsteps] [stepoffsets]\n"
        << "  " << program
        << " <in_base> <elempart_base> <out_base> <nprocs> --getudg <npe>\n"
        << "  " << program
        << " <in_base> <elempart_base> <out_base> <nprocs> --qcriterion <npe>\n"
        << "  " << program
        << " <in_base> <elempart_base> <out_base> <nprocs> --getudgavg <npe>\n"
        << "  " << program
        << " <in_base> <elempart_base> <out_base> <nprocs>"
        << " --averageudgavg <npe> <npe2d> <ne_z>\n"
        << "  " << program
        << " <sol_base> <elempart_base> <out_base> <nprocs>"
        << " --extract2d <npe2d> <ne_z> <i_matlab_csv> <j_matlab_csv>"
        << " [nsteps] [stepoffsets]\n"
        << "  " << program
        << " <sol_base> <elempart_base> <out_base> <nprocs>"
        << " --average2d <npe2d> <ne_z>"
        << " [nsteps] [stepoffsets]\n\n"
        << "Examples:\n"
        << "  " << program << " dataout/outxg xg 4\n"
        << "  " << program << " dataout/outuf uf 4 --getufavg 9 8\n"
        << "  " << program << " dataout/outudg udgf 4 --getudgf 2 10\n"
        << "  " << program << " dataout/outudg udgavg 4 --averageudgf 10 20\n"
        << "  " << program << " dataout/outudg datain/mesh sol 4\n"
        << "  " << program << " dataout/outudg datain/mesh udg 4 --getudg 9\n"
        << "  " << program << " dataout/outudg datain/mesh qcrit 4 --qcriterion 9\n"
        << "  " << program << " dataout/outudgavg datain/mesh udgavg 4 --getudgavg 9\n"
        << "  " << program
        << " dataout/outudgavg datain/mesh udg2davg 4 --averageudgavg 27 9 80\n"
        << "  " << program
        << " dataout/outudg datain/mesh sol2d 4 --extract2d 9 80 2,2,2 20,40,60\n"
        << "  " << program
        << " dataout/outudg datain/mesh sol2davg 4 --average2d 9 80\n";
}

int parsePositiveInt(const char* text, const std::string& name)
{
    const int value = std::atoi(text);
    if (value <= 0) {
        throw std::runtime_error(name + " must be positive.");
    }
    return value;
}

int parseNonnegativeInt(const char* text, const std::string& name)
{
    const int value = std::atoi(text);
    if (value < 0) {
        throw std::runtime_error(name + " must be nonnegative.");
    }
    return value;
}

void writeVector(const std::string& filename, std::vector<double>& values)
{
    if (values.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("Output vector too large for writearray2file: " + filename);
    }
    writearray2file(filename, values.data(), static_cast<int>(values.size()));
}

std::string stepFilename(const std::string& out_base, int step)
{
    return out_base + "_step_" + std::to_string(step) + ".bin";
}

std::string binFilename(const std::string& out_base)
{
    const std::string suffix = ".bin";
    if (out_base.size() >= suffix.size() &&
        out_base.compare(out_base.size() - suffix.size(), suffix.size(), suffix) == 0) {
        return out_base;
    }
    return out_base + suffix;
}

void checkMatlabIndices(const std::vector<int>& i_matlab,
                        const std::vector<int>& j_matlab,
                        int p1,
                        int ne_z)
{
    if (i_matlab.empty()) {
        throw std::runtime_error("i_matlab cannot be empty.");
    }
    if (i_matlab.size() != j_matlab.size()) {
        throw std::runtime_error("i_matlab and j_matlab must have the same size.");
    }

    for (std::size_t k = 0; k < i_matlab.size(); ++k) {
        if (i_matlab[k] < 1 || i_matlab[k] > p1) {
            throw std::runtime_error("i_matlab[" + std::to_string(k) + "]=" +
                                     std::to_string(i_matlab[k]) +
                                     " out of range [1," + std::to_string(p1) + "]");
        }
        if (j_matlab[k] < 1 || j_matlab[k] > ne_z) {
            throw std::runtime_error("j_matlab[" + std::to_string(k) + "]=" +
                                     std::to_string(j_matlab[k]) +
                                     " out of range [1," + std::to_string(ne_z) + "]");
        }
    }
}

std::size_t avg2dIndex(int a, int b, int c, int npe2d, int nc)
{
    return static_cast<std::size_t>(a) +
           static_cast<std::size_t>(npe2d) *
               (static_cast<std::size_t>(b) +
                static_cast<std::size_t>(nc) * static_cast<std::size_t>(c));
}

std::vector<double> averageSol2D(const std::vector<double>& sol3d_flat,
                                 int npe2d,
                                 int p1,
                                 int nc,
                                 int ne2d,
                                 int ne_z)
{
    const std::size_t expected =
        static_cast<std::size_t>(npe2d) *
        static_cast<std::size_t>(p1) *
        static_cast<std::size_t>(nc) *
        static_cast<std::size_t>(ne2d) *
        static_cast<std::size_t>(ne_z);
    if (sol3d_flat.size() != expected) {
        throw std::runtime_error("averageSol2D: sol3d_flat has unexpected size.");
    }

    std::vector<double> sol2davg(static_cast<std::size_t>(npe2d) *
                                 static_cast<std::size_t>(nc) *
                                 static_cast<std::size_t>(ne2d),
                                 0.0);

    for (int j = 0; j < ne_z; ++j) {
        for (int c = 0; c < ne2d; ++c) {
            for (int b = 0; b < nc; ++b) {
                for (int i = 0; i < p1; ++i) {
                    for (int a = 0; a < npe2d; ++a) {
                        sol2davg[avg2dIndex(a, b, c, npe2d, nc)] +=
                            sol3d_flat[idx5(a, i, b, c, j, npe2d, p1, nc, ne2d)];
                    }
                }
            }
        }
    }

    const double scale =
        1.0 / (static_cast<double>(p1) * static_cast<double>(ne_z));
    for (double& value : sol2davg) {
        value *= scale;
    }

    return sol2davg;
}

std::size_t fieldIndex(int i, int c, int e, int npe, int nc)
{
    return static_cast<std::size_t>(i) +
           static_cast<std::size_t>(npe) *
               (static_cast<std::size_t>(c) +
                static_cast<std::size_t>(nc) * static_cast<std::size_t>(e));
}

std::vector<double> qcriterionVisField(const std::vector<double>& udg,
                                       int npe,
                                       int nc,
                                       int ne)
{
    constexpr double gamma = 1.4;
    constexpr int nfields = 5;

    if (nc < 20) {
        throw std::runtime_error(
            "UDG must contain at least 20 components to compute qcriterion.");
    }

    const std::size_t expected =
        static_cast<std::size_t>(npe) *
        static_cast<std::size_t>(nc) *
        static_cast<std::size_t>(ne);
    if (udg.size() != expected) {
        throw std::runtime_error("qcriterionVisField: UDG size mismatch.");
    }

    std::vector<double> visfield(static_cast<std::size_t>(npe) *
                                 static_cast<std::size_t>(nfields) *
                                 static_cast<std::size_t>(ne),
                                 0.0);

    for (int e = 0; e < ne; ++e) {
        for (int i = 0; i < npe; ++i) {
            const double rho = udg[fieldIndex(i, 0, e, npe, nc)];
            if (!std::isfinite(rho) || rho == 0.0) {
                throw std::runtime_error(
                    "qcriterionVisField: invalid density at point " +
                    std::to_string(i) + ", element " + std::to_string(e) + ".");
            }

            const double rhou = udg[fieldIndex(i, 1, e, npe, nc)];
            const double rhov = udg[fieldIndex(i, 2, e, npe, nc)];
            const double rhow = udg[fieldIndex(i, 3, e, npe, nc)];
            const double rhoE = udg[fieldIndex(i, 4, e, npe, nc)];

            const double rx  = udg[fieldIndex(i, 5,  e, npe, nc)];
            const double rux = udg[fieldIndex(i, 6,  e, npe, nc)];
            const double rvx = udg[fieldIndex(i, 7,  e, npe, nc)];
            const double rwx = udg[fieldIndex(i, 8,  e, npe, nc)];

            const double ry  = udg[fieldIndex(i, 10, e, npe, nc)];
            const double ruy = udg[fieldIndex(i, 11, e, npe, nc)];
            const double rvy = udg[fieldIndex(i, 12, e, npe, nc)];
            const double rwy = udg[fieldIndex(i, 13, e, npe, nc)];

            const double rz  = udg[fieldIndex(i, 15, e, npe, nc)];
            const double ruz = udg[fieldIndex(i, 16, e, npe, nc)];
            const double rvz = udg[fieldIndex(i, 17, e, npe, nc)];
            const double rwz = udg[fieldIndex(i, 18, e, npe, nc)];

            const double invrho = 1.0 / rho;
            const double u = rhou * invrho;
            const double v = rhov * invrho;
            const double w = rhow * invrho;
            const double velocityMagnitude = std::sqrt(u * u + v * v + w * w);
            const double pressure = std::abs(
                (gamma - 1.0) *
                (rhoE - 0.5 * (rhou * u + rhov * v + rhow * w)));
            const double mach = velocityMagnitude / std::sqrt(gamma * pressure * invrho);

            const double ux = (rux - u * rx) * invrho;
            const double vx = (rvx - v * rx) * invrho;
            const double wx = (rwx - w * rx) * invrho;

            const double uy = (ruy - u * ry) * invrho;
            const double vy = (rvy - v * ry) * invrho;
            const double wy = (rwy - w * ry) * invrho;

            const double uz = (ruz - u * rz) * invrho;
            const double vz = (rvz - v * rz) * invrho;
            const double wz = (rwz - w * rz) * invrho;

            const double qcrit =
                -0.5 * ux * ux
                -0.5 * vy * vy
                -0.5 * wz * wz
                - uy * vx
                - uz * wx
                - vz * wy;
            const double gradRhoMagnitude = std::sqrt(rx * rx + ry * ry + rz * rz);

            visfield[fieldIndex(i, 0, e, npe, nfields)] = qcrit;
            visfield[fieldIndex(i, 1, e, npe, nfields)] = u;
            visfield[fieldIndex(i, 2, e, npe, nfields)] = pressure;
            visfield[fieldIndex(i, 3, e, npe, nfields)] = mach;
            visfield[fieldIndex(i, 4, e, npe, nfields)] = gradRhoMagnitude;
        }
    }

    return visfield;
}

} // namespace

int main(int argc, char** argv)
{
    try {
        if (argc == 4) {
            const std::string in_base = argv[1];
            const std::string out_base = argv[2];
            const int nprocs = parsePositiveInt(argv[3], "nprocs");

            std::vector<double> xf;
            int n1 = 0;
            int n2 = 0;
            int n3 = 0;
            getxf(in_base, nprocs, xf, n1, n2, n3);
            writeFieldWithHeader(out_base, xf, n1, n2, n3);

            std::cout << "Wrote " << binFilename(out_base)
                      << " (n1=" << n1
                      << ", n2=" << n2
                      << ", n3=" << n3
                      << ", xf.size()=" << xf.size() << ")\n";
            return 0;
        }

        if (argc >= 5 && argc <= 7 && std::string(argv[4]) == "--getudgf") {
            const std::string in_base = argv[1];
            const std::string out_base = argv[2];
            const int nprocs = parsePositiveInt(argv[3], "nprocs");
            const int nsteps = (argc >= 6) ? parsePositiveInt(argv[5], "nsteps") : 1;
            const int stepoffsets =
                (argc >= 7) ? parseNonnegativeInt(argv[6], "stepoffsets") : 0;

            std::vector<double> udgf;
            int n1 = 0;
            int n2 = 0;
            int n3 = 0;
            int n4 = 0;
            getudgf(in_base, nprocs, nsteps, stepoffsets, udgf, n1, n2, n3, n4);
            writeFieldWithHeader4(out_base, udgf, n1, n2, n3, n4);

            std::cout << "Wrote " << binFilename(out_base)
                      << " (n1=" << n1
                      << ", n2=" << n2
                      << ", n3=" << n3
                      << ", nsteps=" << n4
                      << ", udgf.size()=" << udgf.size() << ")\n";
            return 0;
        }

        if (argc >= 5 && argc <= 7 && std::string(argv[4]) == "--averageudgf") {
            const std::string in_base = argv[1];
            const std::string out_base = argv[2];
            const int nprocs = parsePositiveInt(argv[3], "nprocs");
            const int nsteps = (argc >= 6) ? parsePositiveInt(argv[5], "nsteps") : 1;
            const int stepoffsets =
                (argc >= 7) ? parseNonnegativeInt(argv[6], "stepoffsets") : 0;

            std::vector<double> udgf;
            int n1 = 0;
            int n2 = 0;
            int n3 = 0;
            averageudgf(in_base, nprocs, nsteps, stepoffsets, udgf, n1, n2, n3);
            writeFieldWithHeader(out_base, udgf, n1, n2, n3);

            std::cout << "Wrote " << binFilename(out_base)
                      << " (n1=" << n1
                      << ", n2=" << n2
                      << ", n3=" << n3
                      << ", averaged_steps=" << nsteps
                      << ", udgf.size()=" << udgf.size() << ")\n";
            return 0;
        }

        if (argc == 7 && std::string(argv[4]) == "--getufavg") {
            const std::string in_base = argv[1];
            const std::string out_base = argv[2];
            const int nprocs = parsePositiveInt(argv[3], "nprocs");
            const int npf = parsePositiveInt(argv[5], "npf");
            const int ncu = parsePositiveInt(argv[6], "ncu");

            std::vector<double> uf;
            int n1 = 0;
            int n2 = 0;
            int n3 = 0;
            getufavg(in_base, nprocs, npf, ncu, uf, n1, n2, n3);
            writeFieldWithHeader(out_base, uf, n1, n2, n3);

            std::cout << "Wrote " << binFilename(out_base)
                      << " (npf=" << n1
                      << ", nf=" << n2
                      << ", ncu=" << n3
                      << ", uf.size()=" << uf.size() << ")\n";
            return 0;
        }

        if (argc < 5) {
            printUsage(argv[0]);
            return 1;
        }

        const std::string sol_base = argv[1];
        const std::string elempart_base = argv[2];
        const std::string out_base = argv[3];
        const int nprocs = parsePositiveInt(argv[4], "nprocs");

        bool extract2d = false;
        bool average2d = false;
        bool getUdg = false;
        bool qCriterion = false;
        bool getUdgAvg = false;
        bool averageUdgAvg = false;
        int arg = 5;

        int npe = 0;
        int npe2d = 0;
        int ne_z = 0;
        std::vector<int> i_matlab;
        std::vector<int> j_matlab;

        if (arg < argc && std::string(argv[arg]) == "--getudg") {
            getUdg = true;
            if (argc != arg + 2) {
                printUsage(argv[0]);
                return 1;
            }

            npe = parsePositiveInt(argv[arg + 1], "npe");
            arg += 2;
        } else if (arg < argc && std::string(argv[arg]) == "--qcriterion") {
            qCriterion = true;
            if (argc != arg + 2) {
                printUsage(argv[0]);
                return 1;
            }

            npe = parsePositiveInt(argv[arg + 1], "npe");
            arg += 2;
        } else if (arg < argc && std::string(argv[arg]) == "--getudgavg") {
            getUdgAvg = true;
            if (argc != arg + 2) {
                printUsage(argv[0]);
                return 1;
            }

            npe = parsePositiveInt(argv[arg + 1], "npe");
            arg += 2;
        } else if (arg < argc && std::string(argv[arg]) == "--averageudgavg") {
            averageUdgAvg = true;
            if (argc != arg + 4) {
                printUsage(argv[0]);
                return 1;
            }

            npe = parsePositiveInt(argv[arg + 1], "npe");
            npe2d = parsePositiveInt(argv[arg + 2], "npe2d");
            ne_z = parsePositiveInt(argv[arg + 3], "ne_z");
            arg += 4;
        } else if (arg < argc && std::string(argv[arg]) == "--extract2d") {
            extract2d = true;
            if (argc < arg + 5) {
                printUsage(argv[0]);
                return 1;
            }

            npe2d = parsePositiveInt(argv[arg + 1], "npe2d");
            ne_z = parsePositiveInt(argv[arg + 2], "ne_z");
            i_matlab = parseCSVInts(argv[arg + 3]);
            j_matlab = parseCSVInts(argv[arg + 4]);
            arg += 5;
        } else if (arg < argc && std::string(argv[arg]) == "--average2d") {
            average2d = true;
            if (argc < arg + 3) {
                printUsage(argv[0]);
                return 1;
            }

            npe2d = parsePositiveInt(argv[arg + 1], "npe2d");
            ne_z = parsePositiveInt(argv[arg + 2], "ne_z");
            arg += 3;
        }

        std::vector<std::vector<int>> elempartpts(nprocs);
        std::vector<std::vector<int>> elempart(nprocs);
        readelempart(elempart_base, elempart, elempartpts, nprocs);

        if (getUdg) {
            std::vector<double> udg;
            int n1 = 0;
            int n2 = 0;
            int ne = 0;
            getudg(sol_base, elempartpts, elempart, npe, udg, n1, n2, ne);
            writeFieldWithHeader(out_base, udg, n1, n2, ne);

            std::cout << "Wrote " << binFilename(out_base)
                      << " (npe=" << n1
                      << ", nc=" << n2
                      << ", ne=" << ne
                      << ", udg.size()=" << udg.size() << ")\n";
            return 0;
        }

        if (qCriterion) {
            std::vector<double> udg;
            int n1 = 0;
            int n2 = 0;
            int ne = 0;
            getudg(sol_base, elempartpts, elempart, npe, udg, n1, n2, ne);

            std::vector<double> visfield = qcriterionVisField(udg, n1, n2, ne);
            writeFieldWithHeader(out_base, visfield, n1, 5, ne);

            std::cout << "Wrote " << binFilename(out_base)
                      << " (npe=" << n1
                      << ", nfields=5"
                      << ", ne=" << ne
                      << ", visfield.size()=" << visfield.size() << ")\n";
            return 0;
        }

        if (getUdgAvg) {
            std::vector<double> udgavg;
            int n1 = 0;
            int n2 = 0;
            int ne = 0;
            getudgavg(sol_base, elempartpts, elempart, npe, udgavg, n1, n2, ne);
            writeFieldWithHeader(out_base, udgavg, n1, n2, ne);

            std::cout << "Wrote " << binFilename(out_base)
                      << " (npe=" << n1
                      << ", nc=" << n2
                      << ", ne=" << ne
                      << ", udgavg.size()=" << udgavg.size() << ")\n";
            return 0;
        }

        if (averageUdgAvg) {
            std::vector<double> udgavg;
            int n1 = 0;
            int n2 = 0;
            int ne = 0;
            getudgavg(sol_base, elempartpts, elempart, npe, udgavg, n1, n2, ne);

            if (n1 % npe2d != 0) {
                throw std::runtime_error("npe is not divisible by npe2d.");
            }
            if (ne % ne_z != 0) {
                throw std::runtime_error("ne is not divisible by ne_z.");
            }

            const int nc = n2;
            const int p1 = n1 / npe2d;
            const int ne2 = ne / ne_z;
            std::vector<double> udg2davg =
                averageSol2D(udgavg, npe2d, p1, nc, ne2, ne_z);
            writeFieldWithHeader(out_base, udg2davg, npe2d, nc, ne2);

            std::cout << "Wrote " << binFilename(out_base)
                      << " (npe2d=" << npe2d
                      << ", nc=" << nc
                      << ", ne2d=" << ne2
                      << ", udg2davg.size()=" << udg2davg.size() << ")\n";
            return 0;
        }

        const int nsteps = (arg < argc) ? parsePositiveInt(argv[arg++], "nsteps") : 1;
        const int stepoffsets = (arg < argc) ? parseNonnegativeInt(argv[arg++], "stepoffsets") : 0;
        if (arg != argc) {
            printUsage(argv[0]);
            return 1;
        }

        std::vector<double> sol3dGlobal;
        for (int step = 0; step < nsteps; ++step) {
            int n1 = 0;
            int n2 = 0;
            int ne = 0;

            sol3dGlobal.clear();
            readsolution(sol_base,
                         elempartpts,
                         elempart,
                         sol3dGlobal,
                         1,
                         stepoffsets + step,
                         n1,
                         n2,
                         ne);

            if (!extract2d && !average2d) {
                const std::string out = stepFilename(out_base, stepoffsets + step);
                writeVector(out, sol3dGlobal);
                std::cout << "Wrote " << out
                          << " (sol.size()=" << sol3dGlobal.size() << ")\n";
                continue;
            }

            if (n1 % npe2d != 0) {
                throw std::runtime_error("n1 is not divisible by npe2d.");
            }
            if (ne % ne_z != 0) {
                throw std::runtime_error("ne is not divisible by ne_z.");
            }

            const int nc = n2;
            const int p1 = n1 / npe2d;
            const int ne2 = ne / ne_z;

            if (average2d) {
                std::vector<double> sol2davg =
                    averageSol2D(sol3dGlobal,
                                 npe2d,
                                 p1,
                                 nc,
                                 ne2,
                                 ne_z);

                const std::string out = stepFilename(out_base, stepoffsets + step);
                writeVector(out, sol2davg);
                std::cout << "Wrote " << out
                          << " (sol2davg.size()=" << sol2davg.size() << ")\n";
                continue;
            }

            checkMatlabIndices(i_matlab, j_matlab, p1, ne_z);

            std::vector<double> sol2d =
                extractSol2D(sol3dGlobal,
                             npe2d,
                             p1,
                             nc,
                             ne2,
                             ne_z,
                             i_matlab,
                             j_matlab);

            const std::string out = stepFilename(out_base, stepoffsets + step);
            writeVector(out, sol2d);
            std::cout << "Wrote " << out
                      << " (sol2d.size()=" << sol2d.size() << ")\n";
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 2;
    }
}
