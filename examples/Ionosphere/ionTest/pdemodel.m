function pde = pdemodel
pde.mass = @mass;
pde.flux = @flux;
pde.source = @source;
pde.fbou = @fbou;
pde.fbouhdg = @fbouhdg;
pde.ubou = @ubou;
pde.initu = @initu;
end

function m = mass(u, q, w, v, x, t, mu, eta)
m = sym(ones(5,1));
end

function f = flux(u, q, w, v, x, t, mu, eta)
m = mu(1);
kB = mu(2);
u1 = u(1);
u2 = u(2:4);
u3 = u(5);
kappa = 1.24e4*(u3/(kB*u1))^2.5;
q1 = [q(1); q(6); q(11)];
q3 = [q(5); q(10); q(15)];
f1 = u2/m;
f2 = u2*u2.'/(m*u1) + u3*eye(3);
f3 = u3*u2/(m*u1) - 2*kappa*(u3*q1-u1*q3)/(3*kB*u1^2);
f = [f1.'; f2; f3.'];
end

function s = source(u, q, w, v, x, t, mu, eta)
m = mu(1);
e = mu(3);
g = mu(4);
E = v(1:3);
B = v(4:6);
vn = v(7:9);
u1 = u(1);
u2 = u(2:4);
u3 = u(5);
q1 = [q(1); q(6); q(11)];
trq2 = q(2) + q(8) + q(14);
r = sqrt(x(1)^2+x(2)^2+x(3)^2);

% Match the existing Python example's deliberately simplified source.
nu = sym(0.0);
P = sym(0.0);
L = sym(0.0);
H = sym(0.0);
C = sym(0.0);
s1 = P-L*u1;
s21 = (P/u1-L)*u2(1) + u1*m*g*x(1)/r + u1*e*E(1) + e*(u2(2)*B(3)-u2(3)*B(2))/m + nu*(u1*m*vn(1)-u2(1));
s22 = (P/u1-L)*u2(2) + u1*m*g*x(2)/r + u1*e*E(2) + e*(u2(3)*B(1)-u2(1)*B(3))/m + nu*(u1*m*vn(2)-u2(2));
s23 = (P/u1-L)*u2(3) + u1*m*g*x(3)/r + u1*e*E(3) + e*(u2(1)*B(2)-u2(2)*B(1))/m + nu*(u1*m*vn(3)-u2(3));
s3 = 2*u3*(u1*trq2-u2.'*q1)/(3*m*u1) + u3*(P/u1-L) + 2*(H-C*u3)/3;
s = [s1; s21; s22; s23; s3];
end

function ub = ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau)
m = mu(1);
vn = v(7:9);
P = v(11);
L = v(12);
H = v(13);
C = v(14);
equilibrium = [P/L; m*P*vn(1)/L; m*P*vn(2)/L; m*P*vn(3)/L; H/C];
ub = [equilibrium, u];
end

function fb = fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau)
f = flux(u, q, w, v, x, t, mu, eta);
physical = f(:,1)*n(1) + f(:,2)*n(2) + f(:,3)*n(3) + tau*(u-uhat);
fb = [physical, sym(zeros(5,1))];
end

function fb = fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau)
f = flux(u, q, w, v, x, t, mu, eta);
physical = f(:,1)*n(1) + f(:,2)*n(2) + f(:,3)*n(3) + tau*(u-uhat);
m = mu(1);
vn = v(7:9);
P = v(11);
L = v(12);
H = v(13);
C = v(14);
equilibrium = [P/L; m*P*vn(1)/L; m*P*vn(2)/L; m*P*vn(3)/L; H/C];
fb = [tau*(equilibrium-uhat), physical];
end

function u0 = initu(x, mu, eta)
u0 = sym(zeros(5,1));
end
