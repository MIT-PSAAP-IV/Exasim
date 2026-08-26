#include "unirefpt.hpp"

#include <array>
#include <algorithm>
#include <map>
#include <stdexcept>

namespace {

// A refinement template for one element type:
//   generators[g] = local CORNER indices whose mean is a new vertex
//                   (new vertex's local node index is nv + g)
//   children[c]   = the nv local node indices of child c (index < nv is an
//                   original corner; index >= nv is generator (index-nv))
struct RefTemplate {
    int nchild;
    std::vector<std::vector<int>> generators;
    std::vector<std::vector<int>> children;
};

// tri (nv=3): 3 edge midpoints -> 4 children (uniref.m)
RefTemplate tri_template()
{
    return { 4,
        { {0,1},{0,2},{1,2} },                      // nodes 3,4,5
        { {0,3,4},{3,5,4},{1,5,3},{2,4,5} } };
}

// quad (nv=4): 4 edge midpoints + 1 cell center -> 4 children (unirefquad.m)
RefTemplate quad_template()
{
    return { 4,
        { {0,1},{1,2},{2,3},{3,0},{0,1,2,3} },      // edges 4..7, center 8
        { {0,4,8,7},{4,1,5,8},{8,5,2,6},{7,8,6,3} } };
}

// tet (nv=4): 6 edge midpoints -> 8 children (uniref3d.m)
RefTemplate tet_template()
{
    return { 8,
        { {0,1},{0,2},{0,3},{1,2},{1,3},{2,3} },    // edges 4..9
        { {0,4,5,6},{4,1,7,8},{5,7,2,9},{6,8,9,3},
          {4,7,5,6},{8,7,4,6},{9,7,8,6},{5,7,9,6} } };
}

// hex (nv=8): 12 edges + 6 face centers + 1 cell center -> 8 children.
// No MATLAB reference; generated from the tensor-lexicographic corner ordering
// c = x + 2y + 4z (x,y,z in {0,1}). Positions use half-units {0,1,2} = {0,.5,1}.
RefTemplate hex_template()
{
    auto corner_index = [](int a, int b, int c) { return (a/2) + 2*(b/2) + 4*(c/2); };

    std::map<std::array<int,3>, int> pos2node;   // half-unit position -> local node
    std::vector<std::vector<int>> generators;

    // corners: even positions -> local 0..7
    for (int c = 0; c <= 2; c += 2)
        for (int b = 0; b <= 2; b += 2)
            for (int a = 0; a <= 2; a += 2)
                pos2node[{a,b,c}] = corner_index(a,b,c);

    // generators: the non-corner positions, in lexicographic order -> local 8..26
    int g = 8;
    for (int c = 0; c <= 2; ++c)
        for (int b = 0; b <= 2; ++b)
            for (int a = 0; a <= 2; ++a) {
                const bool corner = (a%2==0) && (b%2==0) && (c%2==0);
                if (corner) continue;
                // parent corners: each half (odd) coord expands to {0,2}
                std::vector<int> parents;
                for (int za : (a%2 ? std::vector<int>{0,2} : std::vector<int>{a}))
                    for (int zb : (b%2 ? std::vector<int>{0,2} : std::vector<int>{b}))
                        for (int zc : (c%2 ? std::vector<int>{0,2} : std::vector<int>{c}))
                            parents.push_back(corner_index(za,zb,zc));
                pos2node[{a,b,c}] = g++;
                generators.push_back(parents);
            }

    // children: 8 octants (tensor-lex), each an 8-corner hex in tensor-lex order
    std::vector<std::vector<int>> children;
    for (int ok = 0; ok < 2; ++ok)
        for (int oj = 0; oj < 2; ++oj)
            for (int oi = 0; oi < 2; ++oi) {
                std::vector<int> ch(8);
                for (int cz = 0; cz < 2; ++cz)
                    for (int cy = 0; cy < 2; ++cy)
                        for (int cx = 0; cx < 2; ++cx)
                            ch[cx + 2*cy + 4*cz] = pos2node[{oi+cx, oj+cy, ok+cz}];
                children.push_back(ch);
            }

    return { 8, generators, children };
}

RefTemplate pick_template(int nv, int nd)
{
    if (nv == 3 && nd == 2) return tri_template();
    if (nv == 4 && nd == 2) return quad_template();
    if (nv == 4 && nd == 3) return tet_template();
    if (nv == 8 && nd == 3) return hex_template();
    throw std::runtime_error("unirefpt: unsupported (nv, nd) -- want tri(3,2) quad(4,2) tet(4,3) hex(8,3)");
}

// One uniform refinement pass, in place.
void refine_once(std::vector<double>& p, std::vector<int>& t, int nv, int nd, const RefTemplate& rt)
{
    const int np = (int)(p.size() / nd);
    const int nt = (int)(t.size() / nv);
    const int ng = (int)rt.generators.size();
    const int nchild = rt.nchild;

    std::map<std::vector<int>, int> vmap;   // sorted global-corner tuple -> new vertex index
    std::vector<double> newcoords;          // appended new-vertex coords (discovery order)
    std::vector<int> tnew((std::size_t)nt * nchild * nv);
    int nextidx = np;

    for (int e = 0; e < nt; ++e) {
        const int* gv = &t[(std::size_t)e * nv];
        std::vector<int> gennode(ng);
        for (int gi = 0; gi < ng; ++gi) {
            const std::vector<int>& gen = rt.generators[gi];
            std::vector<int> key(gen.size());
            for (std::size_t j = 0; j < gen.size(); ++j) key[j] = gv[gen[j]];
            std::sort(key.begin(), key.end());
            auto it = vmap.find(key);
            if (it == vmap.end()) {
                const int idx = nextidx++;
                vmap.emplace(key, idx);
                for (int d = 0; d < nd; ++d) {
                    double s = 0.0;
                    for (int lc : gen) s += p[(std::size_t)gv[lc] * nd + d];
                    newcoords.push_back(s / (double)gen.size());
                }
                gennode[gi] = idx;
            } else {
                gennode[gi] = it->second;
            }
        }
        for (int c = 0; c < nchild; ++c)
            for (int k = 0; k < nv; ++k) {
                const int ln = rt.children[c][k];
                tnew[((std::size_t)e * nchild + c) * nv + k] = (ln < nv) ? gv[ln] : gennode[ln - nv];
            }
    }

    p.insert(p.end(), newcoords.begin(), newcoords.end());
    t.swap(tnew);
}

} // namespace

int unirefpt(std::vector<double>& p, std::vector<int>& t, int nv, int nd, int nref)
{
    const RefTemplate rt = pick_template(nv, nd);   // also validates (nv,nd)
    if (nref < 0) throw std::runtime_error("unirefpt: nref must be >= 0");
    for (int r = 0; r < nref; ++r) refine_once(p, t, nv, nd, rt);
    return rt.nchild;
}
