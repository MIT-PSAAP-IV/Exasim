function [fT, dfT] = elementaryfunctions(T)

logT = log(T);
Tinv = 1/T;
T2 = T * T;
T3 = T2 * T;
T4 = T3 * T;
T2inv = 1/T2;
logTTinv = logT*Tinv;

if isa(T, 'sym')
    zero = sym(0);
    one = sym(1);
else
    zero = zeros('like', T);
    one = ones('like', T);
end

fT = [one T T2 T3 T4 Tinv T2inv logT logTTinv];
dfT = [zero, one, 2*T, 3*T2, 4*T3, -T2inv, -2*Tinv*T2inv, Tinv, (one - logT)*T2inv];
 
