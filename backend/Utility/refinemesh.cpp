#include "refinemesh.hpp"

#include <cstddef>
#include <stdexcept>

int refine_nchild(int nd, int nref)
{
    if (nd < 1 || nd > 3 || nref < 1) throw std::runtime_error("refine: nd in 1..3, nref >= 1");
    int n = 1;
    for (int d = 0; d < nd; ++d) n *= nref;
    return n;
}

void refine_child_refnodes(double* xic, const double* plocal, int npe, int nd, int nref)
{
    const int nchild = refine_nchild(nd, nref);
    for (int c = 0; c < nchild; ++c) {
        // decode the lexicographic subcell offset o[0..nd) in [0,nref)^nd
        int off[3] = {0, 0, 0}, t = c;
        for (int d = 0; d < nd; ++d) { off[d] = t % nref; t /= nref; }
        for (int d = 0; d < nd; ++d)
            for (int i = 0; i < npe; ++i)
                xic[i + npe * (d + nd * c)] =
                    (off[d] + plocal[i + npe * d]) / static_cast<double>(nref);
    }
}

void refinemesh(double* refined, const double* dgnodes, const double* Pc,
                int npe, int ncx, int ne, int nchild)
{
    // refined(:,:,c*ne+e) = Pc(:,:,c) * dgnodes(:,:,e)
    for (int c = 0; c < nchild; ++c) {
        const double* Pcc = Pc + static_cast<std::size_t>(npe) * npe * c;   // [npe x npe]
        for (int e = 0; e < ne; ++e) {
            const double* De = dgnodes + static_cast<std::size_t>(npe) * ncx * e;
            double* Re = refined + static_cast<std::size_t>(npe) * ncx * (static_cast<std::size_t>(c) * ne + e);
            for (int j = 0; j < ncx; ++j)
                for (int i = 0; i < npe; ++i) {
                    double s = 0.0;
                    for (int a = 0; a < npe; ++a)
                        s += Pcc[i + npe * a] * De[a + npe * j];   // Pc[i,a] * parent[a,j]
                    Re[i + npe * j] = s;
                }
        }
    }
}
