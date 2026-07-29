#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

void readDoubles(std::ifstream& in,
                 double* dst,
                 std::size_t count,
                 const std::string& fname);

void writeDoubles(std::ofstream& out,
                  const double* values,
                  std::size_t count,
                  const std::string& fname);

std::int64_t fileSizeBytes(const std::string& fname);

std::vector<int> parseCSVInts(const std::string& s);

void writearray2file(const std::string& filename, const double* values, int count);

void writeFieldWithHeader(const std::string& filename,
                          const std::vector<double>& values,
                          int n1,
                          int n2,
                          int n3);

void writeFieldWithHeader4(const std::string& filename,
                           const std::vector<double>& values,
                           int n1,
                           int n2,
                           int n3,
                           int n4);

void getxf(const std::string& base,
           int nprocs,
           std::vector<double>& xf,
           int& n1_out,
           int& n2_out,
           int& n3_out);

void getufavg(const std::string& base,
              int nprocs,
              int npf,
              int ncu,
              std::vector<double>& uf,
              int& n1_out,
              int& n2_out,
              int& n3_out);

void getudgf(const std::string& base,
             int nprocs,
             int nsteps,
             int stepoffsets,
             std::vector<double>& udgf,
             int& n1_out,
             int& n2_out,
             int& n3_out,
             int& n4_out);

void averageudgf(const std::string& base,
                 int nprocs,
                 int nsteps,
                 int stepoffsets,
                 std::vector<double>& udgf,
                 int& n1_out,
                 int& n2_out,
                 int& n3_out);

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
