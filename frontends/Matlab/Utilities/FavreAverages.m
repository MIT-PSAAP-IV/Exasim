function faavg = FavreAverages(reavg)
%FAVREAVERAGES Compute Favre and fluctuation statistics from Reynolds averages.
%
%   faavg = FavreAverages(reavg)
%
%   Input:
%       reavg has size npe x 30 x ne, where npe is the number of
%       interpolation points per element and ne is the number of elements.
%       The second dimension stores the following Reynolds-averaged
%       quantities:
%
      %  1 mean(rho)        2 mean(rho*u)      3 mean(rho*v)
      %  4 mean(rho*w)      5 mean(rho*E)      6 mean(u)
      %  7 mean(v)          8 mean(w)          9 mean(p)
      % 10 mean(T)         11 mean(rho*u^2)   12 mean(rho*v^2)
      % 13 mean(rho*w^2)   14 mean(rho*u*v)   15 mean(rho*u*w)
      % 16 mean(rho*v*w)   17 mean(u^2)       18 mean(v^2)
      % 19 mean(w^2)       20 mean(u*v)       21 mean(u*w)
      % 22 mean(v*w)       23 mean(rho^2)     24 mean(p^2)
      % 25 mean(T^2)       26 mean(rho*T)     27 mean(rho*T^2)
      % 28 mean(rho*u*T)   29 mean(rho*v*T)   30 mean(rho*w*T)
%
%   Reynolds averages use overbars, for example mean(u) = \bar{u}.
%   Favre averages use density weighting, for example
%       \tilde{u} = \overline{rho*u} / \bar{rho}.
%
%   Prime fluctuations are Reynolds fluctuations:
%       u = \bar{u} + u'.
%   Double-prime fluctuations are Favre fluctuations:
%       u = \tilde{u} + u''.
%
%   Output:
%       faavg has size npe x 48 x ne with component ordering:
%
%        1 mean(rho)              2 Favre mean u
%        3 Favre mean v           4 Favre mean w
%        5 Favre mean E           6 Favre mean T
%        7 Favre mean u^2         8 Favre mean v^2
%        9 Favre mean w^2        10 Favre mean u*v
%       11 Favre mean u*w        12 Favre mean v*w
%       13 Favre mean T^2        14 Favre mean u*T
%       15 Favre mean v*T        16 Favre mean w*T
%       17 Favre variance u      18 Favre variance v
%       19 Favre variance w      20 Favre covariance u,v
%       21 Favre covariance u,w  22 Favre covariance v,w
%       23 Favre variance T      24 Favre covariance u,T
%       25 Favre covariance v,T  26 Favre covariance w,T
%       27 tau11                 28 tau22
%       29 tau33                 30 tau12
%       31 tau13                 32 tau23
%       33 turbulent kinetic energy k
%       34 Reynolds variance u   35 Reynolds variance v
%       36 Reynolds variance w   37 Reynolds covariance u,v
%       38 Reynolds covariance u,w
%       39 Reynolds covariance v,w
%       40 uRMS                  41 vRMS
%       42 wRMS                  43 pressure variance
%       44 pressure RMS          45 density variance
%       46 density RMS           47 temperature variance
%       48 temperature RMS

if size(reavg, 2) ~= 30
    error('FavreAverages:InvalidInputSize', ...
          'The input reavg must have size npe x 30 x ne.');
end

npe = size(reavg, 1);
ne = size(reavg, 3);
faavg = zeros(npe, 48, ne, 'like', reavg);

rhoBar = reavg(:, 1, :);
rhoUBar = reavg(:, 2, :);
rhoVBar = reavg(:, 3, :);
rhoWBar = reavg(:, 4, :);
rhoEBar = reavg(:, 5, :);
uBar = reavg(:, 6, :);
vBar = reavg(:, 7, :);
wBar = reavg(:, 8, :);
pBar = reavg(:, 9, :);
TBar = reavg(:, 10, :);
rhoUUBar = reavg(:, 11, :);
rhoVVBar = reavg(:, 12, :);
rhoWWBar = reavg(:, 13, :);
rhoUVBar = reavg(:, 14, :);
rhoUWBar = reavg(:, 15, :);
rhoVWBar = reavg(:, 16, :);
uuBar = reavg(:, 17, :);
vvBar = reavg(:, 18, :);
wwBar = reavg(:, 19, :);
uvBar = reavg(:, 20, :);
uwBar = reavg(:, 21, :);
vwBar = reavg(:, 22, :);
rhoRhoBar = reavg(:, 23, :);
pPBar = reavg(:, 24, :);
TTBar = reavg(:, 25, :);
rhoTBar = reavg(:, 26, :);
rhoTTBar = reavg(:, 27, :);
rhoUTBar = reavg(:, 28, :);
rhoVTBar = reavg(:, 29, :);
rhoWTBar = reavg(:, 30, :);

if any(rhoBar(:) <= 0)
    error('FavreAverages:NonpositiveMeanDensity', ...
          'The mean density must be positive at every point.');
end

uFavre = rhoUBar ./ rhoBar;
vFavre = rhoVBar ./ rhoBar;
wFavre = rhoWBar ./ rhoBar;
EFavre = rhoEBar ./ rhoBar;
TFavre = rhoTBar ./ rhoBar;

uuFavre = rhoUUBar ./ rhoBar;
vvFavre = rhoVVBar ./ rhoBar;
wwFavre = rhoWWBar ./ rhoBar;
uvFavre = rhoUVBar ./ rhoBar;
uwFavre = rhoUWBar ./ rhoBar;
vwFavre = rhoVWBar ./ rhoBar;
TTFavre = rhoTTBar ./ rhoBar;
uTFavre = rhoUTBar ./ rhoBar;
vTFavre = rhoVTBar ./ rhoBar;
wTFavre = rhoWTBar ./ rhoBar;

uuFavreVariance = max(uuFavre - uFavre.^2, 0);
vvFavreVariance = max(vvFavre - vFavre.^2, 0);
wwFavreVariance = max(wwFavre - wFavre.^2, 0);
uvFavreCovariance = uvFavre - uFavre .* vFavre;
uwFavreCovariance = uwFavre - uFavre .* wFavre;
vwFavreCovariance = vwFavre - vFavre .* wFavre;

TTFavreVariance = max(TTFavre - TFavre.^2, 0);
uTFavreCovariance = uTFavre - uFavre .* TFavre;
vTFavreCovariance = vTFavre - vFavre .* TFavre;
wTFavreCovariance = wTFavre - wFavre .* TFavre;

tau11 = rhoBar .* uuFavreVariance;
tau22 = rhoBar .* vvFavreVariance;
tau33 = rhoBar .* wwFavreVariance;
tau12 = rhoBar .* uvFavreCovariance;
tau13 = rhoBar .* uwFavreCovariance;
tau23 = rhoBar .* vwFavreCovariance;

turbulentKineticEnergy = 0.5 * ...
    (uuFavreVariance + vvFavreVariance + wwFavreVariance);

uVariance = max(uuBar - uBar.^2, 0);
vVariance = max(vvBar - vBar.^2, 0);
wVariance = max(wwBar - wBar.^2, 0);
uvCovariance = uvBar - uBar .* vBar;
uwCovariance = uwBar - uBar .* wBar;
vwCovariance = vwBar - vBar .* wBar;

uRMS = sqrt(uVariance);
vRMS = sqrt(vVariance);
wRMS = sqrt(wVariance);

pVariance = max(pPBar - pBar.^2, 0);
pRMS = sqrt(pVariance);

rhoVariance = max(rhoRhoBar - rhoBar.^2, 0);
rhoRMS = sqrt(rhoVariance);

TVariance = max(TTBar - TBar.^2, 0);
TRMS = sqrt(TVariance);

faavg(:, 1, :) = rhoBar;
faavg(:, 2, :) = uFavre;
faavg(:, 3, :) = vFavre;
faavg(:, 4, :) = wFavre;
faavg(:, 5, :) = EFavre;
faavg(:, 6, :) = TFavre;
faavg(:, 7, :) = uuFavre;
faavg(:, 8, :) = vvFavre;
faavg(:, 9, :) = wwFavre;
faavg(:, 10, :) = uvFavre;
faavg(:, 11, :) = uwFavre;
faavg(:, 12, :) = vwFavre;
faavg(:, 13, :) = TTFavre;
faavg(:, 14, :) = uTFavre;
faavg(:, 15, :) = vTFavre;
faavg(:, 16, :) = wTFavre;
faavg(:, 17, :) = uuFavreVariance;
faavg(:, 18, :) = vvFavreVariance;
faavg(:, 19, :) = wwFavreVariance;
faavg(:, 20, :) = uvFavreCovariance;
faavg(:, 21, :) = uwFavreCovariance;
faavg(:, 22, :) = vwFavreCovariance;
faavg(:, 23, :) = TTFavreVariance;
faavg(:, 24, :) = uTFavreCovariance;
faavg(:, 25, :) = vTFavreCovariance;
faavg(:, 26, :) = wTFavreCovariance;
faavg(:, 27, :) = tau11;
faavg(:, 28, :) = tau22;
faavg(:, 29, :) = tau33;
faavg(:, 30, :) = tau12;
faavg(:, 31, :) = tau13;
faavg(:, 32, :) = tau23;
faavg(:, 33, :) = turbulentKineticEnergy;
faavg(:, 34, :) = uVariance;
faavg(:, 35, :) = vVariance;
faavg(:, 36, :) = wVariance;
faavg(:, 37, :) = uvCovariance;
faavg(:, 38, :) = uwCovariance;
faavg(:, 39, :) = vwCovariance;
faavg(:, 40, :) = uRMS;
faavg(:, 41, :) = vRMS;
faavg(:, 42, :) = wRMS;
faavg(:, 43, :) = pVariance;
faavg(:, 44, :) = pRMS;
faavg(:, 45, :) = rhoVariance;
faavg(:, 46, :) = rhoRMS;
faavg(:, 47, :) = TVariance;
faavg(:, 48, :) = TRMS;
end
