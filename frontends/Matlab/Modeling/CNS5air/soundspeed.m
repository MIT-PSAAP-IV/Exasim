function [a, gammaMix, cpMix, cvMix, Rmix] = soundspeed(T, rho_i, alpha)
%SOUNDSPEED Frozen-composition speed of sound for the five-species-air model.
%   A = SOUNDSPEED(T, RHO_I) returns the dimensional frozen speed of sound
%   [m/s] at temperature T [K] and species densities RHO_I [kg/m^3]. For
%   batched DG data, RHO_I has size [npe,5,ne] and T has size [npe,1,ne];
%   all outputs then have size [npe,1,ne]. A single state may be supplied as
%   a five-entry row or column vector and a scalar temperature.
%
%   [A, GAMMA, CP, CV, RMIX] also returns the local mixture heat-capacity
%   ratio, specific heats [J/(kg K)], and gas constant [J/(kg K)]. Species
%   mass fractions are held fixed during the acoustic perturbation.
%
%   ALPHA controls the smooth transition between NASA-9 temperature ranges
%   and defaults to 1e3, consistently with the CNS5air thermodynamics.

if nargin < 3
    alpha = 1e3;
end

[speciesThermo, Mw, RU] = thermodynamicsModels();
Mw = Mw(:);
isSymbolic = isa(T, 'sym') || isa(rho_i, 'sym');
ns = numel(Mw);

if isvector(rho_i) && numel(rho_i) == ns
    rho_i = reshape(rho_i, [1, ns]);
    outputSize = [1, 1];
    if ~isscalar(T)
        error('soundspeed:InvalidTemperatureSize', ...
              'T must be scalar when rho_i describes one state.');
    end
elseif size(rho_i,2) == ns
    outputSize = size(rho_i);
    outputSize(2) = 1;
    if isscalar(T) && ~isSymbolic
        T = T + zeros(outputSize, 'like', rho_i);
    elseif numel(T) == prod(outputSize)
        T = reshape(T, outputSize);
    else
        error('soundspeed:InvalidTemperatureSize', ...
              'T must be scalar or have the size of rho_i with dimension 2 equal to one.');
    end
else
    error('soundspeed:InvalidSpeciesCount', ...
          'rho_i must contain %d species densities in dimension 2.', ns);
end
if ~isSymbolic
    if any(~isfinite(T(:))) || any(T(:) <= 0)
        error('soundspeed:InvalidTemperature', 'T must contain positive finite values.');
    end
    rho = sum(rho_i,2);
    if any(~isfinite(rho_i(:))) || any(rho_i(:) < 0) || any(rho(:) <= 0)
        error('soundspeed:InvalidDensity', ...
              'Species densities must be finite, nonnegative, and have positive total density.');
    end
else
    rho = sum(rho_i,2);
end

Y = rho_i./rho;
cpSpecies = 0*rho_i;

for i = 1:ns
    thermo = speciesThermo{i};
    cpSpecies(:,i,:) = localSpeciesCp(T, thermo, alpha, RU/Mw(i));
end

mwShape = ones(1, max(ndims(rho_i),2));
mwShape(2) = ns;
Rmix = RU*sum(Y./reshape(Mw,mwShape),2);
cpMix = sum(Y.*cpSpecies,2);
cvMix = cpMix - Rmix;
gammaMix = cpMix./cvMix;
a = sqrt(gammaMix.*Rmix.*T);

if ~isSymbolic && (any(~isfinite(a(:))) || ~isreal(a) || any(cvMix(:) <= 0))
    error('soundspeed:InvalidThermodynamicState', ...
          'The supplied state does not produce a positive finite sound speed.');
end
end

function cp = localSpeciesCp(T, thermo, alpha, gasConstant)
sw1 = 0.5*tanh(-alpha*(T-thermo.T1)/pi) + 0.5;
sw3 = 0.5*tanh( alpha*(T-thermo.T2)/pi) + 0.5;
sw2 = 1.0 - sw1 - sw3;

cp1 = localCpOverR(T, nasa9_cpcoeff(thermo.a1));
cp2 = localCpOverR(T, nasa9_cpcoeff(thermo.a2));
cp3 = localCpOverR(T, nasa9_cpcoeff(thermo.a3));
cp = gasConstant.*(sw1.*cp1 + sw2.*cp2 + sw3.*cp3);
end

function cp = localCpOverR(T, c)
T2 = T.*T;
cp = c(1) + c(2).*T + c(3).*T2 + c(4).*T2.*T + c(5).*T2.*T2 ...
   + c(6)./T + c(7)./T2;
end
