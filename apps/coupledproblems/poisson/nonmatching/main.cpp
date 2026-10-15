// Coupled Text2Code application; see RunText2CodeApp in <exasim/ExasimSolverSetup.hpp>.
//   mpirun -np <ranks1> build/exasimapp pdeapp0.txt <ranks1>                       (stand-alone stage)
//   mpirun -np <ranks1+ranks2> build/exasimapp pdeapp1.txt <ranks1> pdeapp2.txt <ranks2>
#include <exasim/ExasimSolverSetup.hpp>

int main(int argc, char** argv) { return RunText2CodeApp(argc, argv); }
