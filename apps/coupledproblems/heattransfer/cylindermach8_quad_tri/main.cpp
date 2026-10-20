// Coupled heat-transfer Text2Code application (see RunText2CodeApp in <exasim/ExasimSolverSetup.hpp>):
//   mpirun -np <ranks1> build/exasimapp pdeapp0.txt <ranks1>                       (stand-alone flow)
//   mpirun -np <ranks1+ranks2> build/exasimapp pdeapp1.txt <ranks1> pdeapp2.txt <ranks2>
// After the stand-alone flow solve, its final state and external fields are gathered into
// udg1.bin and vdg1.bin, which pdeapp1.txt reads as udgfile/vdgfile.
#include <filesystem>
#include <exasim/ExasimSolverSetup.hpp>

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    const int err = RunText2CodeApp(argc, argv);   // leaves MPI to this main
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (argc == 3 && err == 0 && rank == 0) {
        const auto dir = std::filesystem::absolute(argv[1]).parent_path();
        const int ranks = std::stoi(argv[2]);
        GatherSolution(dir / "datain", dir / "dataout", "udg", ranks, dir / "udg1.bin");
        GatherSolution(dir / "datain", dir / "dataout", "vdg", ranks, dir / "vdg1.bin");
    }
    MPI_Finalize();
    return err;
}
