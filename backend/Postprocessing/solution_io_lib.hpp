#pragma once

#include <string>
#include <vector>

std::vector<int> parseCSVInts(const std::string& s);

void writearray2file(const std::string& filename, const double* values, int count);

void readelempart(const std::string& base,
                  std::vector<std::vector<int>>& elempart,
                  std::vector<std::vector<int>>& elempartpts,
                  int nprocs);

void readsolution(const std::string& base,
                  const std::vector<std::vector<int>>& elempartpts,
                  const std::vector<std::vector<int>>& elempart,
                  std::vector<double>& sol3dGlobal,
                  int nsteps,
                  int stepoffsets,
                  int& n1_out,
                  int& n2_out,
                  int& ne_out);

std::vector<double> extractSol2D(const std::vector<double>& sol3dnew_flat,
                                 int npe2,
                                 int p1,
                                 int nc,
                                 int ne2,
                                 int ne_z,
                                 const std::vector<int>& i_matlab,
                                 const std::vector<int>& j_matlab);
