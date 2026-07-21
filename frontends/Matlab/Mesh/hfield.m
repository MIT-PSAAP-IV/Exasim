function h = hfield(d,hmin,hmax,dmin,dmax,alpha)

% slope = (hmax-hmin)/(dmax-dmin)
% h = hmin + max(slope*(d-dmin),0);
% h = min(h-hmax,0)+hmax;

slope = (hmax-hmin)/(dmax-dmin);
h = hmin + lmax(slope*(d-dmin),alpha);
h = lmin(h-hmax,alpha) + hmax;


