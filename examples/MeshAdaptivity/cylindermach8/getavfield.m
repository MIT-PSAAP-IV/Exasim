function avField = getavfield(udg,qdg,vdg,param)

% Get base variables
r = udg(:,1,:);
ru = udg(:,2,:);
rv = udg(:,3,:);
rx = qdg(:,1,:);
rux = qdg(:,2,:);
ry = qdg(:,5,:);
rvy = qdg(:,7,:);

% Regularization of density
r1 = 1./r;
uv = ru.*r1;
vv = rv.*r1;

% Computing derivatives for the sensors
ux = (rux - rx.*uv).*r1;
vy = (rvy - ry.*vv).*r1;
div = (ux + vy);

% limit  divergence
avField = limiting(div.*tanh(param(end-2).*vdg(:,1,:)), 0, param(end-3), 1e3, 0);
