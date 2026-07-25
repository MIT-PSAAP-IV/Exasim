// c++ -std=c++17 -Wall -Wextra -pedantic -O3 solution_io.cpp -o solution_io
// Example:
//   ./solution_io dataout/outudg datain/mesh sol2davg 736 --average2d 9 18

#include <cstdlib>
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
        << " <sol_base> <elempart_base> <out_base> <nprocs> [nsteps] [stepoffsets]\n"
        << "  " << program
        << " <sol_base> <elempart_base> <out_base> <nprocs>"
        << " --extract2d <npe2d> <ne_z> <i_matlab_csv> <j_matlab_csv>"
        << " [nsteps] [stepoffsets]\n"
        << "  " << program
        << " <sol_base> <elempart_base> <out_base> <nprocs>"
        << " --average2d <npe2d> <ne_z>"
        << " [nsteps] [stepoffsets]\n\n"
        << "Examples:\n"
        << "  " << program << " dataout/outudg datain/mesh sol 4\n"
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

} // namespace

int main(int argc, char** argv)
{
    try {
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
        int arg = 5;

        int npe2d = 0;
        int ne_z = 0;
        std::vector<int> i_matlab;
        std::vector<int> j_matlab;

        if (arg < argc && std::string(argv[arg]) == "--extract2d") {
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

        const int nsteps = (arg < argc) ? parsePositiveInt(argv[arg++], "nsteps") : 1;
        const int stepoffsets = (arg < argc) ? parseNonnegativeInt(argv[arg++], "stepoffsets") : 0;
        if (arg != argc) {
            printUsage(argv[0]);
            return 1;
        }

        std::vector<std::vector<int>> elempartpts(nprocs);
        std::vector<std::vector<int>> elempart(nprocs);
        readelempart(elempart_base, elempart, elempartpts, nprocs);

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
