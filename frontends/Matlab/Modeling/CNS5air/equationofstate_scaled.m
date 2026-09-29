function f = equationofstate_scaled(T,rhos,rhoe,rhoe_scale,alpha)
%EQUATIONOFSTATE_SCALED Dimensionless mixture internal-energy residual.

if nargin < 5
    alpha = 1e3;
end

[thermo,Mw,RU] = thermodynamicsModels();
fT = elementaryfunctions(T);
computedRhoe = 0*T;

for i = 1:length(Mw)
    fsw = switchfunctions(T,thermo{i}.T1,thermo{i}.T2,alpha);

    c1 = nasa9_hcoeff(thermo{i}.a1,thermo{i}.b1);
    c2 = nasa9_hcoeff(thermo{i}.a2,thermo{i}.b2);
    c3 = nasa9_hcoeff(thermo{i}.a3,thermo{i}.b3);

    H1 = sum(c1.*fT);
    H2 = sum(c2.*fT);
    H3 = sum(c3.*fT);
    H = fsw(1)*H1 + fsw(2)*H2 + fsw(3)*H3;

    computedRhoe = computedRhoe + rhos(i)*(RU/Mw(i))*T*(H-1);
end

f = (computedRhoe-rhoe)/rhoe_scale;
end
