// c++ -std=c++17 -Wall -Wextra -pedantic -O3 reynolds_averages_3d_main.cpp -o reynolds_averages_3d

#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include "solution_io_lib.cpp"
#include "reynolds_averages_3d.cpp"

namespace {

void printUsage(const char* program)
{
    std::cerr
        << "Usage:\n"
        << "  " << program << " <sol_base> <nprocs> [nsteps] [stepoffsets] [gamma]\n\n"
        << "Examples:\n"
        << "  " << program << " dataout/outudg 4 100\n"
        << "  " << program << " dataout/outudg 4 100 20\n"
        << "  " << program << " dataout/outudg 4 100 20 1.4\n\n"
        << "Input files are <sol_base>_np<rank>.bin for rank = 0, ..., nprocs-1.\n"
        << "Output files are <sol_base>_avg_np<rank>.bin.\n";
}

int parsePositiveInt(const char* text, const std::string& name)
{
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (*text == '\0' || *end != '\0' || value <= 0 ||
        value > static_cast<long>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(name + " must be a positive integer.");
    }
    return static_cast<int>(value);
}

int parseNonnegativeInt(const char* text, const std::string& name)
{
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (*text == '\0' || *end != '\0' || value < 0 ||
        value > static_cast<long>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(name + " must be a nonnegative integer.");
    }
    return static_cast<int>(value);
}

double parseDouble(const char* text, const std::string& name)
{
    char* end = nullptr;
    const double value = std::strtod(text, &end);
    if (*text == '\0' || *end != '\0') {
        throw std::runtime_error(name + " must be a floating-point number.");
    }
    return value;
}

std::string rankInputFile(const std::string& solBase, int rank)
{
    return solBase + "_np" + std::to_string(rank) + ".bin";
}

std::string rankOutputFile(const std::string& solBase, int rank)
{
    return solBase + "_avg_np" + std::to_string(rank) + ".bin";
}

} // namespace

int main(int argc, char** argv)
{
    try {
        if (argc < 3 || argc > 6) {
            printUsage(argv[0]);
            return 1;
        }

        const std::string solBase = argv[1];
        const int nprocs = parsePositiveInt(argv[2], "nprocs");
        const int nsteps = (argc >= 4) ? parsePositiveInt(argv[3], "nsteps") : 1;
        const bool hasStepOffsets = (argc >= 5);
        const int stepoffsets = hasStepOffsets ? parseNonnegativeInt(argv[4], "stepoffsets") : 0;
        const double gamma = (argc >= 6) ? parseDouble(argv[5], "gamma") : 1.4;

        for (int rank = 0; rank < nprocs; ++rank) {
            const std::string filein = rankInputFile(solBase, rank);
            const std::string fileout = rankOutputFile(solBase, rank);

            if (hasStepOffsets) {
                ReynoldsAverages3D(fileout, filein, nsteps, stepoffsets, gamma);
            } else {
                ReynoldsAverages3D(fileout, filein, nsteps, gamma);
            }

            std::cout << "Wrote " << fileout << " from " << filein << "\n";
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 2;
    }
}
