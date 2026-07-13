function [p,t] = refinecartgrid(x, y, yref)

[p,t] = cart2dg(1,1,x,y);
[p,t] = fixmesh(p,t);

n = length(yref);
if n>0
    yref = sort(yref,'descend');
    for i = 1:n
        [p,t] = refineaty(p,t,yref(i));
        [p,t] = fixmesh(p,t);
    end
    [p,t] = fixmesh(p,t);
end
