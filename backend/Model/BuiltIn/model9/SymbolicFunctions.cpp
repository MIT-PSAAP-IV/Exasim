#include "SymbolicFunctions.hpp"

std::vector<Expression> Flux(const std::vector<Expression>& x, const std::vector<Expression>& uq, const std::vector<Expression>& v, const std::vector<Expression>& w, const std::vector<Expression>& eta, const std::vector<Expression>& mu, const Expression& t) {
    std::vector<Expression> f;
    f.resize(4);

    Expression kappa1 = mu[0];
    Expression kappa2 = mu[1];
    f[0]  =  kappa1*uq[2];
    f[1]  =  kappa1*uq[3];
    f[2]  =  kappa2*uq[4];
    f[3]  =  kappa2*uq[5];
    return f;
}

std::vector<Expression> Source(const std::vector<Expression>& x, const std::vector<Expression>& uq, const std::vector<Expression>& v, const std::vector<Expression>& w, const std::vector<Expression>& eta, const std::vector<Expression>& mu, const Expression& t) {
    std::vector<Expression> s;
    s.resize(2);

    Expression x1 = x[0];
    Expression x2 = x[1];
    s[0]  =  0.0;
    s[1]  =  0.0;
    return s;
}

std::vector<Expression> Tdfunc(const std::vector<Expression>& x, const std::vector<Expression>& uq, const std::vector<Expression>& v, const std::vector<Expression>& w, const std::vector<Expression>& eta, const std::vector<Expression>& mu, const Expression& t) {
    std::vector<Expression> m;
    m.resize(2);

    for (int i = 0; i < 2; ++i) {
         m[i] = Expression(1);
    }
    return m;
}

std::vector<Expression> Fbou(const std::vector<Expression>& x, const std::vector<Expression>& uq, const std::vector<Expression>& v, const std::vector<Expression>& w, const std::vector<Expression>& uhat, const std::vector<Expression>& n, const std::vector<Expression>& tau, const std::vector<Expression>& eta, const std::vector<Expression>& mu, const Expression& t) {
    std::vector<Expression> fb;
    fb.resize(2);

    auto f = Flux(x, uq, v, w, eta, mu, t);
    fb[0]  =  f[0]*n[0] + f[1]*n[1] + tau[0]*(uq[0]-uhat[0]);
    fb[1]  =  f[2]*n[0] + f[3]*n[1] + tau[1]*(uq[1]-uhat[1]);
    return fb;
}

std::vector<Expression> Ubou(const std::vector<Expression>& x, const std::vector<Expression>& uq, const std::vector<Expression>& v, const std::vector<Expression>& w, const std::vector<Expression>& uhat, const std::vector<Expression>& n, const std::vector<Expression>& tau, const std::vector<Expression>& eta, const std::vector<Expression>& mu, const Expression& t) {
    std::vector<Expression> ub;
    ub.resize(2);

    ub[0]  =  0.0;
    ub[1]  =  0.0;
    return ub;
}

std::vector<Expression> FbouHdg(const std::vector<Expression>& x, const std::vector<Expression>& uq, const std::vector<Expression>& v, const std::vector<Expression>& w, const std::vector<Expression>& uhat, const std::vector<Expression>& n, const std::vector<Expression>& tau, const std::vector<Expression>& eta, const std::vector<Expression>& mu, const Expression& t) {
    std::vector<Expression> fb;
    fb.resize(6);

    Expression ub1 = x[0] + x[1];
    Expression ub2 = x[0] - x[1];
    fb[0]  =  tau[0]*(ub1 - uhat[0]);
    fb[1]  =  tau[1]*(ub2 - uhat[1]);
    fb[2]  =  tau[0]*(ub1 - uhat[0]);
    fb[3]  =  tau[1]*(ub2 - uhat[1]);
    fb[4]  =  tau[0]*(ub1 - uhat[0]);
    fb[5]  =  tau[1]*(ub2 - uhat[1]);
    return fb;
}

std::vector<Expression> Fint(const std::vector<Expression>& x, const std::vector<Expression>& uq, const std::vector<Expression>& v, const std::vector<Expression>& w, const std::vector<Expression>& uhat, const std::vector<Expression>& n, const std::vector<Expression>& tau, const std::vector<Expression>& eta, const std::vector<Expression>& mu, const Expression& t) {
    std::vector<Expression> fb;
    fb.resize(2);

    auto f = Flux(x, uq, v, w, eta, mu, t);
    fb[0]  =  f[0]*n[0] + f[1]*n[1] + tau[0]*(uq[0] - uhat[0]);
    fb[1]  =  f[2]*n[0] + f[3]*n[1] + tau[1]*(uq[1] - uhat[1]);
    return fb;
}

std::vector<Expression> Fext(const std::vector<Expression>& x, const std::vector<Expression>& uq, const std::vector<Expression>& v, const std::vector<Expression>& w, const std::vector<Expression>& uhat, const std::vector<Expression>& n, const std::vector<Expression>& uext, const std::vector<Expression>& tau, const std::vector<Expression>& eta, const std::vector<Expression>& mu, const Expression& t) {
    std::vector<Expression> fb;
    fb.resize(2);

    fb[0]  =  uext[0] - uhat[0];
    fb[1]  =  uext[1] - uhat[1];
    return fb;
}

std::vector<Expression> Initu(const std::vector<Expression>& x, const std::vector<Expression>& eta, const std::vector<Expression>& mu) {
    std::vector<Expression> ui;
    ui.resize(2);

    ui[0]  =  0.0;
    ui[1]  =  0.0;
    return ui;
}

std::vector<Expression> VisScalars(const std::vector<Expression>& x, const std::vector<Expression>& uq, const std::vector<Expression>& v, const std::vector<Expression>& w, const std::vector<Expression>& eta, const std::vector<Expression>& mu, const Expression& t) {
    std::vector<Expression> s;
    s.resize(6);

    s[0]  =  uq[0];
    s[1]  =  uq[1];
    s[2]  =  uq[2];
    s[3]  =  uq[3];
    s[4]  =  uq[4];
    s[5]  =  uq[5];
    return s;
}

std::vector<Expression> VisVectors(const std::vector<Expression>& x, const std::vector<Expression>& uq, const std::vector<Expression>& v, const std::vector<Expression>& w, const std::vector<Expression>& eta, const std::vector<Expression>& mu, const Expression& t) {
    std::vector<Expression> s;
    s.resize(4);

    s[0]  =  uq[2];
    s[1]  =  uq[3];
    s[2]  =  uq[4];
    s[3]  =  uq[5];
    return s;
}

std::vector<Expression> QoIvolume(const std::vector<Expression>& x, const std::vector<Expression>& uq, const std::vector<Expression>& v, const std::vector<Expression>& w, const std::vector<Expression>& eta, const std::vector<Expression>& mu, const Expression& t) {
    std::vector<Expression> s;
    s.resize(4);

    Expression x1 = x[0];
    Expression x2 = x[1];
    auto t1 = Expression(SymEngine::pi);
    auto t2 = sin(t1*x1);
    auto t3 = sin(t1*x2);
    Expression uexact1 = x1;
    Expression uexact2 = x2;
    s[0]  =  (uq[0] - uexact1)*(uq[0] - uexact1);
    s[1]  =  (uq[1] - uexact2)*(uq[1] - uexact2);
    s[2]  =  uq[0];
    s[3]  =  uq[1];
    return s;
}

std::vector<Expression> QoIboundary(const std::vector<Expression>& x, const std::vector<Expression>& uq, const std::vector<Expression>& v, const std::vector<Expression>& w, const std::vector<Expression>& uhat, const std::vector<Expression>& n, const std::vector<Expression>& tau, const std::vector<Expression>& eta, const std::vector<Expression>& mu, const Expression& t) {
    std::vector<Expression> fb;
    fb.resize(2);

    auto f = Flux(x, uq, v, w, eta, mu, t);
    fb[0]  =  f[0]*n[0] + f[1]*n[1] + tau[0]*(uq[0]-uhat[0]);
    fb[1]  =  f[2]*n[0] + f[3]*n[1] + tau[1]*(uq[1]-uhat[1]);
    return fb;
}

std::vector<Expression> Initv(const std::vector<Expression>& x, const std::vector<Expression>& eta, const std::vector<Expression>& mu) {
    std::vector<Expression> vi;
    vi.resize(2);

    vi[0]  =  0.5;
    vi[1]  =  0.5;
    return vi;
}

