function [fT, dfT] = elementaryfunctions(T)

logT = log(T);
Tinv = 1/T;
T2 = T * T;
T3 = T2 * T;
T4 = T3 * T;
T2inv = 1/T2;
logTTinv = logT*Tinv;

fT = [sym(1) T T2 T3 T4 Tinv T2inv logT logTTinv];
dfT = [sym(0), sym(1), 2*T, 3*T2, 4*T3, -T2inv, -2*Tinv*T2inv, Tinv, (1 - logT)*T2inv];
 
