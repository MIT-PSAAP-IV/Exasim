
datafile = fullfile(fileparts(mfilename('fullpath')), 'circlewithholes.mat');
if exist(datafile, 'file')
    load(datafile);
else
    R = 10;
    ph = local_three_circular_holes(5, 0.7, 64);
end

% R = 10;
nc = 125;
pc = zeros(nc,2);
t = linspace(0, 2*pi, nc+1)';
pc(:,1) = R*cos(t(1:end-1));
pc(:,2) = R*sin(t(1:end-1));

dmin = 0.3;
dmax = 7.1;
alpha = 30;
beta = 0.96;
gamma = 1.1;
Gmsh = 'gmsh';
[p,t] = gmsh_circlewithholes(pc,ph,1,dmin,dmax,alpha,beta,gamma,Gmsh);

figure(1);clf;simpplot(p', t');
hold on;
plot(pc(:,1), pc(:,2), 'o-');
for i = 1:length(ph)
    loop = ph{i};
    plot(loop([1:end 1],1), loop([1:end 1],2), 'o-');
end
axis equal tight;
title('Gmsh boundary loops');

% windturbine2d_plot(wt);
% hold on; 
% simpplot(p', t');

function ph = local_three_circular_holes(Rrotor, Rhole, npts)
theta = linspace(0, 2*pi, npts+1)';
theta = theta(1:end-1);
angles = 2*pi*(0:2)/3;
ph = cell(1, numel(angles));
for i = 1:numel(angles)
    center = Rrotor*[cos(angles(i)), sin(angles(i))];
    loop = center + Rhole*[cos(theta), sin(theta)];
    ph{i} = flipud(loop);
end
end
