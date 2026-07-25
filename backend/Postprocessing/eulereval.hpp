#pragma once

#include <string>

void eulereval1d(double* sca,
                 const double* u,
                 const std::string& quantity,
                 double gamma,
                 int npe,
                 int nc,
                 int ne);

void eulereval2d(double* sca,
                 const double* u,
                 const std::string& quantity,
                 double gamma,
                 int npe,
                 int nc,
                 int ne);

void eulereval3d(double* sca,
                 const double* u,
                 const std::string& quantity,
                 double gamma,
                 int npe,
                 int nc,
                 int ne);

void eulereval(double* sca,
               const double* u,
               const std::string& quantity,
               double gamma,
               int npe,
               int nc,
               int ne,
               int nd);
