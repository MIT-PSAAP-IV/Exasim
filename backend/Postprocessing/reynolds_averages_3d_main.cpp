// c++ -std=c++17 -Wall -Wextra -pedantic -O3 reynolds_averages_3d_main.cpp -o reynolds_averages_3d
// mpicxx -std=c++17 -Wall -Wextra -pedantic -O3 -D_MPI reynolds_averages_3d_main.cpp -o reynolds_averages_3d_mpi

#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#ifdef _MPI
#define HAVE_MPI
#endif

#ifdef HAVE_MPI
#include <mpi.h>
#endif

#include "solution_io_lib.cpp"
#include "reynolds_averages_3d.cpp"

namespace {

void printUsage(const char* program)
{
    std::cerr
        << "Usage:\n"
        << "  " << program << " <sol_base> <fileout_base> <nprocs> [nsteps] [stepoffsets] [gamma]\n\n"
        << "Examples:\n"
        << "  " << program << " dataout/outudg dataout/reynolds_avg 4 100\n"
        << "  " << program << " dataout/outudg dataout/reynolds_avg 4 100 20\n"
        << "  " << program << " dataout/outudg dataout/reynolds_avg 4 100 20 1.4\n\n"
        << "Input files are <sol_base>_np<rank>.bin for rank = 0, ..., nprocs-1.\n"
        << "Output files are <fileout_base>_np<rank>.bin.\n"
        << "With MPI, the number of MPI ranks must be <= nprocs.\n";
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

std::string rankOutputFile(const std::string& fileoutBase, int rank)
{
    return fileoutBase + "_np" + std::to_string(rank) + ".bin";
}

} // namespace

int main(int argc, char** argv)
{
#ifdef HAVE_MPI
    MPI_Init(&argc, &argv);
#endif

    int mpiRank = 0;
    int mpiSize = 1;
#ifdef HAVE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
#endif

    try {
        if (argc < 4 || argc > 7) {
#ifdef HAVE_MPI
            if (mpiRank == 0) {
                printUsage(argv[0]);
            }
#else
            printUsage(argv[0]);
#endif
#ifdef HAVE_MPI
            MPI_Finalize();
#endif
            return 1;
        }

        const std::string solBase = argv[1];
        const std::string fileoutBase = argv[2];
        const int nprocs = parsePositiveInt(argv[3], "nprocs");
        const int nsteps = (argc >= 5) ? parsePositiveInt(argv[4], "nsteps") : 1;
        const bool hasStepOffsets = (argc >= 6);
        const int stepoffsets = hasStepOffsets ? parseNonnegativeInt(argv[5], "stepoffsets") : 0;
        const double gamma = (argc >= 7) ? parseDouble(argv[6], "gamma") : 1.4;

        if (mpiSize > nprocs) {
            throw std::runtime_error("The number of MPI ranks must be less than or equal to nprocs.");
        }

        for (int rank = mpiRank; rank < nprocs; rank += mpiSize) {
            const std::string filein = rankInputFile(solBase, rank);
            const std::string fileout = rankOutputFile(fileoutBase, rank);

            if (hasStepOffsets) {
                ReynoldsAverages3D(fileout, filein, nsteps, stepoffsets, gamma);
            } else {
                ReynoldsAverages3D(fileout, filein, nsteps, gamma);
            }

#ifdef HAVE_MPI
            std::cout << "MPI rank " << mpiRank << " wrote "
                      << fileout << " from " << filein << "\n";
#else
            std::cout << "Wrote " << fileout << " from " << filein << "\n";
#endif
        }

#ifdef HAVE_MPI
        MPI_Finalize();
#endif
        return 0;
    } catch (const std::exception& e) {
#ifdef HAVE_MPI
        std::cerr << "ERROR [MPI rank " << mpiRank << "]: " << e.what() << "\n";
        MPI_Finalize();
#else
        std::cerr << "ERROR: " << e.what() << "\n";
#endif
        return 2;
    }
}
