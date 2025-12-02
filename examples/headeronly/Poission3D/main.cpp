#include "exasim/Main/libexasimheader.h"
#include "exasim/Main/BuiltIn/Poisson3D/model.hpp"


int main(int argc, char ** argv) {
  // call exasim stuff
  premain<Poisson3D::Poisson3D>(argc, argv);
}
