#include "unirefpt.hpp"

#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(bool ok, const std::string& l) { if (!ok) throw std::runtime_error("FAIL: " + l); }

// does the mesh contain a vertex at the given coords (within tol)?
bool hasVertex(const std::vector<double>& p, int nd, std::vector<double> x, double tol = 1e-12)
{
    const int np = (int)(p.size() / nd);
    for (int i = 0; i < np; ++i) {
        bool eq = true;
        for (int d = 0; d < nd; ++d) if (std::fabs(p[i*nd+d] - x[d]) > tol) { eq = false; break; }
        if (eq) return true;
    }
    return false;
}
void checkConnValid(const std::vector<int>& t, int nv, int np, const std::string& tag)
{
    for (std::size_t i = 0; i < t.size(); ++i)
        check(t[i] >= 0 && t[i] < np, tag + " connectivity index out of range");
    check(t.size() % nv == 0, tag + " connectivity size");
}
} // namespace

int main()
{
    // ---- tri: 1 element -> 4 children, 3 edge midpoints ------------------------
    {
        std::vector<double> p = {0,0, 1,0, 0,1};
        std::vector<int> t = {0,1,2};
        int nc = unirefpt(p, t, 3, 2, 1);
        check(nc == 4, "tri nchild");
        check(p.size()/2 == 6, "tri np (3 + 3 mids)");
        check(t.size()/3 == 4, "tri nt");
        check(hasVertex(p,2,{0.5,0.0}), "tri mid 01");
        check(hasVertex(p,2,{0.0,0.5}), "tri mid 02");
        check(hasVertex(p,2,{0.5,0.5}), "tri mid 12");
        checkConnValid(t, 3, 6, "tri");
    }
    // tri conformity: 2 triangles sharing edge (1,2) -> shared midpoint deduped
    {
        std::vector<double> p = {0,0, 1,0, 0,1, 1,1};
        std::vector<int> t = {0,1,2, 1,3,2};
        unirefpt(p, t, 3, 2, 1);
        // unique edges: 01,02,12,13,23 = 5 -> np = 4 + 5 = 9 (not 4+6)
        check(p.size()/2 == 9, "tri conformity: shared edge midpoint deduped");
        check(t.size()/3 == 8, "tri conformity nt");
    }

    // ---- quad: 1 element -> 4 children, 4 edge mids + 1 center -----------------
    {
        std::vector<double> p = {0,0, 1,0, 1,1, 0,1};   // CCW corners
        std::vector<int> t = {0,1,2,3};
        int nc = unirefpt(p, t, 4, 2, 1);
        check(nc == 4, "quad nchild");
        check(p.size()/2 == 9, "quad np (4 + 4 mids + center)");
        check(t.size()/4 == 4, "quad nt");
        check(hasVertex(p,2,{0.5,0.5}), "quad center");
        check(hasVertex(p,2,{0.5,0.0}), "quad edge mid");
        checkConnValid(t, 4, 9, "quad");
    }

    // ---- tet: 1 element -> 8 children, 6 edge midpoints -----------------------
    {
        std::vector<double> p = {0,0,0, 1,0,0, 0,1,0, 0,0,1};
        std::vector<int> t = {0,1,2,3};
        int nc = unirefpt(p, t, 4, 3, 1);
        check(nc == 8, "tet nchild");
        check(p.size()/3 == 10, "tet np (4 + 6 mids)");
        check(t.size()/4 == 8, "tet nt");
        check(hasVertex(p,3,{0.5,0.5,0.0}), "tet mid 12");
        checkConnValid(t, 4, 10, "tet");
    }

    // ---- hex: 1 element -> 8 children, 12 edges + 6 faces + 1 cell = 19 new ----
    {
        // tensor-lex corners c = x + 2y + 4z
        std::vector<double> p = {0,0,0, 1,0,0, 0,1,0, 1,1,0, 0,0,1, 1,0,1, 0,1,1, 1,1,1};
        std::vector<int> t = {0,1,2,3,4,5,6,7};
        int nc = unirefpt(p, t, 8, 3, 1);
        check(nc == 8, "hex nchild");
        check(p.size()/3 == 27, "hex np (8 + 12 + 6 + 1)");
        check(t.size()/8 == 8, "hex nt");
        check(hasVertex(p,3,{0.5,0.5,0.5}), "hex cell center");
        check(hasVertex(p,3,{0.5,0.0,0.0}), "hex edge mid (x)");
        check(hasVertex(p,3,{0.5,0.5,0.0}), "hex face center (z=0)");
        check(hasVertex(p,3,{0.0,0.5,0.5}), "hex face center (x=0)");
        checkConnValid(t, 8, 27, "hex");
        // child 0 (octant 0,0,0) must be the [0,0.5]^3 sub-cube: its 8 corners
        // span exactly [0,0.5] in each axis, corner 0 at origin, corner 7 at center.
        auto vx = [&](int e, int k, int d) { return p[t[e*8+k]*3 + d]; };
        for (int d = 0; d < 3; ++d) { check(vx(0,0,d)==0.0, "hex child0 corner0=origin"); check(vx(0,7,d)==0.5, "hex child0 corner7=center"); }
        double mn[3]={9,9,9}, mx[3]={-9,-9,-9};
        for (int k=0;k<8;++k) for (int d=0;d<3;++d){ double v=vx(0,k,d); if(v<mn[d])mn[d]=v; if(v>mx[d])mx[d]=v; }
        for (int d=0;d<3;++d){ check(mn[d]==0.0 && mx[d]==0.5, "hex child0 spans [0,0.5]"); }
    }
    // hex conformity: 2 hexes sharing the x=1/x=0 face -> that face's 4 edge mids
    // + 1 face center are shared. Two unit cubes stacked in x.
    {
        std::vector<double> p = {
            0,0,0, 1,0,0, 0,1,0, 1,1,0, 0,0,1, 1,0,1, 0,1,1, 1,1,1,   // cube A [0,1]^3
            2,0,0, 2,1,0, 2,0,1, 2,1,1                                 // extra corners at x=2
        };
        // cube B [1,2]x[0,1]x[0,1], tensor-lex corners
        std::vector<int> t = {0,1,2,3,4,5,6,7,  1,8,3,9,5,10,7,11};
        int npA = 12;
        unirefpt(p, t, 8, 3, 1);
        // without dedup: 2*19 = 38 new; the shared face (x=1) contributes 4 edge
        // mids + 1 center = 5 shared -> new = 38 - 5 = 33; np = 12 + 33 = 45.
        check(p.size()/3 == (std::size_t)(npA + 33), "hex conformity: shared face deduped");
        check(t.size()/8 == 16, "hex conformity nt");
    }

    // nref=2 compounds (tri: 4^2 = 16 elements)
    {
        std::vector<double> p = {0,0, 1,0, 0,1};
        std::vector<int> t = {0,1,2};
        unirefpt(p, t, 3, 2, 2);
        check(t.size()/3 == 16, "tri nref=2 -> 16 elements");
    }

    std::cout << "unirefpt tests passed\n";
    return 0;
}
