run(fullfile(fileparts(mfilename('fullpath')), '..', '..', '..', 'frontends', 'Matlab', 'exasim_setup.m'));

porder = 4;                     % polynomial degree
tau = 3.0;                      % stabilization parameter
gam = 1.4;                      % gas constant
Minf = 0.2;                     % freestream mach number
alpha = 0*pi/180;               % angle of attack
rinf = 1.0;                     % freestream density
ruinf = cos(alpha);             % freestream horizontal velocity
rvinf = sin(alpha);             % freestream vertical velocity
pinf = 1/(gam*Minf^2);          % freestream pressure
rEinf = 0.5+pinf/(gam-1);       % freestream energy
Re = 1000;                       % Reynolds number 
Pr = 0.72;                      % Prandtl number 
ui = [ 1, cos(alpha), sin(alpha), 0.5+pinf/(gam-1)];

% initialize pde structure and mesh structure
[pde,~] = initializeexasim();

pde.model = "ModelD";          % ModelC, ModelD, ModelW
pde.modelfile = "pdemodel";    % name of a file defining the PDE model

% Choose computing platform and set number of processors
%pde.platform = "gpu";         % choose this option if NVIDIA GPUs are available
pde.mpiprocs = 4;              % number of MPI processors
pde.hybrid = 1;
pde.debugmode = 0;
pde.porder = porder;
pde.pgauss = 2*porder;

pde.physicsparam = [gam Re Pr Minf rinf ruinf rvinf rEinf];
pde.tau = tau;              % DG stabilization parameter
pde.GMRESrestart = 400;
pde.linearsolvertol = 1e-3;
pde.linearsolveriter = 400;
pde.ppdegree = 0;
pde.RBdim = 0;
pde.preconditioner = 1;
pde.neb = 512;
pde.gencode = 1;
%pde.codegenerator = "text2code";
%pde.cartgridpart = [2 64 16 4 4 1];

% naca mesh
%[mesh, dgnodes] = mkmesh_naca0012(porder,1,2);
%mesh.dgnodes = dgnodes;

[mesh, dgnodes] = mkmesh_naca0012(porder,1,1);
pde.uniformrefinementlevel = 2;

% call exasim to generate and run C++ code to solve the PDE model
[sol,pde,mesh,master,dmd] = exasim(pde,mesh);

% plot solution
mesh.porder = porder;
mesh.dgnodes = createdgnodes(mesh.p,mesh.t,mesh.f,mesh.curvedboundary,mesh.curvedboundaryexpr,porder);    
figure(1); clf; scaplot(mesh,eulereval(sol(:,1:4,:),'M',gam),[],2); axis off; axis equal; axis tight;

return;

dmd = extendmd(dmd);
UH = fetchuhat(dmd,pde);
checkuhat(dmd,UH);

pde.denseblock = 1;
pde.source = 'source';
pde.flux = 'flux';
pde.fbou = 'fbou';
pde.fhat = 'fhat';
pde.arg = {gam,0.0,Re,Pr,Minf,tau};
pde.bcm  = [2,1];  % 2: Wall, 1: Far-field
pde.bcs  = [ui;ui];
pde.fc_q = 1;
pde.fc_u = 1;
pde.time = 0;
pde.tdep = 0;
pde.nd = 2;
mesh1 = hdgmesh(mesh, porder);
master = Master(pde);
UDG0 = initu(mesh1,{ui(1),ui(2),ui(3),ui(4); 0,0,0,0; 0,0,0,0});
UH0 = getuhat(UDG0, mesh1.f2t, master.perm, 4);
[UDG,UH] = hdgsolve(master,mesh1,pde,UDG0,UH0,[]);


% compareexasim(master, mesh1, pde);
% 
pde.GMRESrestart = 400;
pde.linearsolvertol = 1e-3;
pde.linearsolveriter = 400;

pde.denseblock = 1;
[UDG,UH] = hdgsolve(master,mesh1,pde,UDG0,UH0,[]);

pde.denseblock = 2;
[UDG,UH] = hdgsolve(master,mesh1,pde,UDG0,UH0,[]);

% pde.denseblock = 3;
% mesh1.epath = 1:1024;
% mesh1.nelems = cumsum([0 8*ones(1,128)]);
% [UDG,UH] = hdgsolve(master,mesh1,pde,UDG0,UH0,[]);
% 
pde.denseblock = 3;
mesh1.epath = 1:1024;
mesh1.nelems = cumsum([0 16*ones(1,64)]);
[UDG,UH] = hdgsolve(master,mesh1,pde,UDG0,UH0,[]);
% 
% pde.denseblock = 3;
% mesh1.epath = 1:16:1009;
% for i = 2:16
%   mesh1.epath = [mesh1.epath i:16:(1009+i-1)];
% end
% mesh1.nelems = cumsum([0 64*ones(1,16)]);
% [UDG,UH] = hdgsolve(master,mesh1,pde,UDG0,UH0,[]);
% 
pde.denseblock = 3;
mesh1.epath = 1:16:1009;
for i = 2:16
  mesh1.epath = [mesh1.epath i:16:(1009+i-1)];
end
mesh1.nelems = cumsum([0 16*ones(1,64)]);
[UDG,UH] = hdgsolve(master,mesh1,pde,UDG0,UH0,[]);

pde.denseblock = 4;
mesh1.epath = 1:16:1009;
for i = 2:16
  mesh1.epath = [mesh1.epath i:16:(1009+i-1)];
end
nep = 16;
mesh1.epath = reshape(mesh1.epath,[nep length(mesh1.epath)/nep])';
[UDG,UH] = hdgsolve(master,mesh1,pde,UDG0,UH0,[]);

e = [1 17 33 49];
e = [e e+1 e+2 e+3];
for i = 2:4
  e = [e; e(end,:)+4];
end
elem = e;
for i=2:16
  elem = [elem; e+64*(i-1)];
end

e = [1 17];
e = [e e+1];
for i = 2:8
  e = [e; e(end,:)+2];
end
elem = e;
for i=2:32
  elem = [elem; e+32*(i-1)];
end

e = [1 17 33 49 65 81 97 113];
e = [e e+1 e+2 e+3 e+4 e+5 e+6 e+7];
for i = 2:2
  e = [e; e(end,:)+8];
end
elem = e;
for i=2:8
  elem = [elem; e+128*(i-1)];
end

pde.denseblock = 5;
mesh1.elem = elem;
[UDG,UH] = hdgsolve(master,mesh1,pde,UDG0,UH0,[]);

pde.denseblock = 6;
mesh1.elem = elem;
[UDG,UH] = hdgsolve(master,mesh1,pde,UDG0,UH0,[]);

pde.denseblock = 7;
mesh1.elem = elem;
[UDG,UH] = hdgsolve(master,mesh1,pde,UDG0,UH0,[]);

%[row_ptr, col_ind, face, f2e] = facereordering(elem, mesh1.t, mesh1.t2f, elemtype, nd);
%[row_ptr, col_ind, face, f2e, row_ptr2, col_ind2, face2, color, idr1, idr2, idr3, idx1, idx2, idx3] = facereordering(elem, mesh1.t, mesh1.t2f, elemtype, nd);

[row_ptr, col_ind, face] = crs_faceordering(elem, mesh1.f2t);    
    
tm = readbin(pde.buildpath + "/dataout/outelem.bin", 'int');
max(abs(tm(:)+1-elem(:)))
tm = readbin(pde.buildpath + "/dataout/outf2e.bin", 'int');
max(abs(tm(:)+1-mesh1.f2t(:)))
tm = readbin(pde.buildpath + "/dataout/outrow_ptr.bin", 'int');
max(abs(tm(:)-row_ptr(:)))
tm = readbin(pde.buildpath + "/dataout/outcol_ind.bin", 'int');
max(abs(tm(:)+1-col_ind(:)))
tm = readbin(pde.buildpath + "/dataout/outface.bin", 'int');
max(abs(tm(:)+1-face(:)))

[ind_ii, ind_ji, ind_jl, ind_il, num_ji, num_jl, Lind_ji, Uind_ji, Lnum_ji, Unum_ji] = crs_indexingilu0(row_ptr, col_ind, nfe);

tm = readbin(pde.buildpath + "/dataout/outind_ii.bin", 'int');
max(abs(tm(:)+1-ind_ii(:)))
tm = readbin(pde.buildpath + "/dataout/outind_ji.bin", 'int');
max(abs(tm(:)+1-ind_ji(:)))
tm = readbin(pde.buildpath + "/dataout/outind_jl.bin", 'int');
max(abs(tm(:)+1-ind_jl(:)))
tm = readbin(pde.buildpath + "/dataout/outind_il.bin", 'int');
max(abs(tm(:)+1-ind_il(:)))
tm = readbin(pde.buildpath + "/dataout/outnum_ji.bin", 'int');
max(abs(tm(:)-num_ji(:)))
tm = readbin(pde.buildpath + "/dataout/outnum_jl.bin", 'int');
max(abs(tm(:)-num_jl(:)))
tm = readbin(pde.buildpath + "/dataout/outLind_ji.bin", 'int');
max(abs(tm(:)+1-Lind_ji(:)))
tm = readbin(pde.buildpath + "/dataout/outUind_ji.bin", 'int');
max(abs(tm(:)+1-Uind_ji(:)))
tm = readbin(pde.buildpath + "/dataout/outLnum_ji.bin", 'int');
max(abs(tm(:)-Lnum_ji(:)))
tm = readbin(pde.buildpath + "/dataout/outUnum_ji.bin", 'int');
max(abs(tm(:)-Unum_ji(:)))

tm = readbin(pde.buildpath + "/dataout/outAE.bin");
[MinvC, MinvE] = qequationint(master, mesh1.dgnodes);
[AE, FE] = uequationint(master,mesh1,pde,UDG0,UH0,[],MinvC,MinvE);
tm = reshape(tm, size(AE));
max(abs(tm(:)-AE(:)))

npf = porder+1;
nfe = 4;
ncu = 4;
ne = size(mesh1.dgnodes,3);
BE = crs_fullassembly(AE, reshape(mesh1.elcon, [npf nfe ne]), mesh1.f2t, face, row_ptr, col_ind, ncu, npf, nfe);      
tm = readbin(pde.buildpath + "/dataout/outBE.bin");
tm = reshape(tm, size(BE));
max(abs(tm(:)-BE(:)))

BE = crs_parblockilu0(ind_ii, ind_ji, ind_jl, ind_il, num_ji, num_jl, BE);
tm = readbin(pde.buildpath + "/dataout/outBE1.bin");
tm = reshape(tm, size(BE));
max(abs(tm(:)-BE(:)))

b = assembleRHS(FE, mesh1.elcon);
[neb, nfeb] = size(face);
ncf = size(BE,1);
b1 = faceextract(reshape(b,ncf,[]), face);
b2 = crs_parblockilu0_solve2(Lind_ji, Uind_ji, Lnum_ji, Unum_ji, BE, b1);    
x = faceinsert(reshape(0*b,ncf,[]), b2, face);

tm = readbin(pde.buildpath + "/dataout/outK.bin");
tm = reshape(tm, size(BE));
max(abs(tm(:)-BE(:)))

tm = readbin(pde.buildpath + "/dataout/outb.bin");
tm = reshape(tm, size(b));
max(abs(tm(:)-b(:)))

tm = readbin(pde.buildpath + "/dataout/outb1.bin");
tm = reshape(tm, size(b1));
max(abs(tm(:)-b1(:)))

tm = readbin(pde.buildpath + "/dataout/outb2.bin");
tm = reshape(tm, size(b2));
max(abs(tm(:)-b2(:)))

tm = readbin(pde.buildpath + "/dataout/outx.bin");
tm = reshape(tm, size(x));
max(abs(tm(:)-x(:)))


[MinvC, MinvE] = qequationint(master, mesh1.dgnodes);
[AE, FE] = uequationint(master,mesh1,pde,UDG0,UH0,[],MinvC,MinvE);
[K, F] = assemblelinearsystem(AE, FE, mesh1.elcon);  
F = reshape(F, pde.ncu *master.npf, []);

nn = 41;

ncu = 4; nfe = 4; npf = mesh1.porder+1;
[val, A] = crs_assembly(AE, mesh1.elcon, mesh1.f2t, face(nn,:), row_ptr, col_ind, ncu, npf, nfe);

count1 = size(idr1,2);
count2 = size(idr2,2);
count3 = size(idr3,2);
count4 = row_ptr2(end);
[A1, A2, A3, B1, B2, B3, C1, C2, C3, D] = matrix_assembly(AE, mesh1.elcon, mesh1.f2t, face, row_ptr, col_ind, color, count1, count2, count3, count4, ncu, npf, nfe);

D2 = 0*D;
for i = 1:size(face2,1)
  tm = crs_assembly(AE, mesh1.elcon, mesh1.f2t, face2(i,:), row_ptr2, col_ind2, ncu, npf, nfe);
  D2(:,:,i,:) = tm;
end
max(abs(D(:)-D2(:)))

D_full = crs2full(row_ptr2, col_ind2, squeeze(D(:,:,nn,:)));
nf1 = count1 + 2*count2 + 3*count3;
n = ncu*npf*nf1;
e = D_full - A(n+1:end,n+1:end);
max(abs(e(:)))

[A1, A2, A3, C1, C2, C3, D] = matrix_compute(A1, A2, A3, B1, B2, B3, C1, C2, C3, D, idx1, idx2, idx3);

D4 = A(n+1:end,n+1:end) - A(n+1:end,1:n)*inv(A(1:n,1:n))*A(1:n,n+1:end);
D_full = crs2full(row_ptr2, col_ind2, squeeze(D(:,:,nn,:)));
e = D_full - D4;
max(abs(e(:)))

rhs = faceextract(F, face);
[r1, r2, r3, r4] = vector_compute(C1, C2, C3, rhs, idr1, idr2, idr3, size(face,1), nfe);

b = faceextract(F, face);
c = reshape(b(:,nn,:), [], 1);
u = A\c;

d = c(n+1:end) - A(n+1:end,1:n)*inv(A(1:n,1:n))*c(1:n);
w = (A(n+1:end,n+1:end) - A(n+1:end,1:n)*inv(A(1:n,1:n))*A(1:n,n+1:end))\d;
max(abs(u(n+1:end)-w))

a = reshape(r4(:,nn,:), [], 1);
max(abs(d-a))

v = crs2full(row_ptr2, col_ind2, squeeze(D(:,:,nn,:)))\reshape(r4(:,nn,:), [], 1);
max(abs(u(n+1:end)-v))

BD = block_ilu0(row_ptr2, col_ind2, squeeze(D(:,:,nn,:)));   
u4 = block_ilu0_solve(row_ptr2, col_ind2, BD, reshape(r4(:,nn,:), [], 1));
max(abs(u(n+1:end)-u4))

u4 = zeros(size(D,1), size(face,1), length(row_ptr2)-1);
for kk = 1:size(face,1)
%   BD = block_ilu0(row_ptr2, col_ind2, squeeze(D(:,:,kk,:)));   
%   tm = block_ilu0_solve(row_ptr2, col_ind2, BD, reshape(r4(:,kk,:), [], 1));
  tm =  crs2full(row_ptr2, col_ind2, squeeze(D(:,:,kk,:)))\reshape(r4(:,kk,:), [], 1);
  u4(:,kk,:) = reshape(tm, [size(D,1) 1 length(row_ptr2)-1]);
end

uh = vector_apply(A1, A2, A3, B1, B2, B3, r1, r2, r3, u4, face, idr1, idr2, idr3, count1, count2, count3, nfe);
e = reshape(uh(:,nn,:), [], 1) - u;
max(abs(e(:)))

e = reshape(u4(:,nn,:), [], 1) - u(n+1:end);
max(abs(e(:)))


% [u(n+1:end) v(:)]
% 
% uh = faceextract(F, face);
% u1 = uh(:,:,1:count1);
% u2 = uh(:,:,(count1+1):2*count2);
% u3 = uh(:,:,(count1+2*count2+1):(count1+2*count2+3*count3));
% u4 = uh(:,:,(count1+2*count2+3*count3+1):end);
% for i = 1:count1
%   for j = 1:nbe
%     ut = reshape(C1(:,:,j,i)*u1(:,j,i), [M nfe-1]);
%     for k = 1:(nfe-1)
%       m = idr1(k,i);
%       u4(:,j,m) = ut(:,j);
%     end
%   end
% end
% 
% for i = 1:count2
%   for j = 1:nbe
%     ut = reshape(C2(:,:,j,i)*u2(:,j,i), [M nfe-2]);
%     for k = 1:(nfe-2)
%       m = idr2(k,i);
%       u4(:,j,m) = ut(:,j);
%     end
%   end
% end
% 
% for i = 1:count3
%   for j = 1:nbe
%     ut = reshape(C3(:,:,j,i)*u3(:,j,i), [M nfe-2]);
%     for k = 1:(nfe-3)
%       m = idr3(k,i);
%       u4(:,j,m) = ut(:,j);
%     end
%   end
% end
% 
% 
% nd = 2; elemtype=1;
% localface = getelemface(nd,elemtype);
% e = mesh1.t(:,mesh1.f2t(1,:));
% l = localface(:,mesh1.f2t(2,:));
% nf = size(mesh1.f2t,2);
% f = zeros(2,nf);
% for i = 1:nf
%   f(:,i) = e(l(:,i),i);
% end
% mesh2 = mesh1;
% mesh2.f = f';
% figure(1);clf;meshplot(mesh2, [0 0 0 0 1]);
% 

% pde.denseblock = 4;
% mesh1.epath = 1:1024;
% mesh1.nelems = cumsum([0 16*ones(1,64)]);
% mesh1.epath2 = 1:16:1009;
% for i = 2:16
%   mesh1.epath2 = [mesh1.epath2 i:16:(1009+i-1)];
% end
% mesh1.nelems2 = cumsum([0 64*ones(1,16)]);
% [UDG,UH] = hdgsolve(master,mesh1,pde,UDG0,UH0,[]);
% 
% [MinvC, MinvE] = qequationint(master, mesh1.dgnodes);
% [AE, FE] = uequationint(master,mesh1,pde,UDG0,UH0,[],MinvC,MinvE);
% [K, F] = assemblelinearsystem(AE, FE, mesh1.elcon);  
% F = reshape(F, pde.ncu *master.npf, []);
% 
% nep = 16;
% epath = 1:1024; 
% nelems = cumsum([0 nep*ones(1,length(epath)/nep)]);
% [fpath, lpath, fintf, lintf] = pathreordering(epath, nelems, mesh1.t2f);
% [A, B, C, D] = pathcompute(AE, epath, fpath, lpath, fintf, lintf, nelems, mesh1.f2t, mesh1.t2f, mesh1.elcon);
% uh1 = pathapply(A, B, C, D, F, fpath, fintf, nelems);
% 
% [fpath, lpath, fintf, lintf, epath] = pathreordering2(reshape(epath,[nep length(epath)/nep])', mesh1.t2f);
% [A, B1, B2, C1, C2, D1, D2, DL, DU] = pathcompute2(AE, epath, fpath, lpath, fintf, lintf, mesh1.f2t, mesh1.t2f, mesh1.elcon);
% uh2 = pathapply2(A, B1, B2, C1, C2, D1, D2, DL, DU, F, fpath, fintf, nep);
% max(abs(uh1(:)-uh(:)))


% pde.denseblock = 3;
% mesh1.epath = 1:256;
% mesh1.nelems = cumsum([0 8*ones(1,32)]);
% [UDG,UH] = hdgsolve(master,mesh1,pde,UDG0,UH0,[]);
% 
% pde.denseblock = 3;
% mesh1.epath = [1:8:249 2:8:250 3:8:251 4:8:252 5:8:253 6:8:254 7:8:255 8:8:256];
% %mesh1.epath = [8:8:256 7:8:255 6:8:254 5:8:253 4:8:252 3:8:251 2:8:250 1:8:249];
% mesh1.nelems = cumsum([0 32*ones(1,8)]);
% [UDG,UH] = hdgsolve(master,mesh1,pde,UDG0,UH0,[]);
% 
% 

% [MinvC, MinvE] = qequationint(master, mesh1.dgnodes);
% [AE, FE] = uequationint(master,mesh1,pde,UDG0,UH0,[],MinvC,MinvE);
% [K, F] = assemblelinearsystem(AE, FE, mesh1.elcon);  
% F = reshape(F, pde.ncu *master.npf, []);
% 
% 
% epath = 1:256;
% nelems = cumsum([0 8*ones(1,32)]);
% npaths = 32;
% [fpath, lpath, fintf, lintf] = pathreordering(epath, nelems, npaths, mesh1.t2f);
% 
% [A, B, C, D] = pathsystem(AE, epath, fpath, lpath, fintf, lintf, mesh1.f2t, mesh1.t2f, mesh1.elcon);
% for i = 1:npaths
%   ind = (nelems(i)+1):nelems(i+1);
%   [A(:,:,ind), C(:,:,ind), D(:,:,ind)] = pathlu(A(:,:,ind), B(:,:,ind), C(:,:,ind), D(:,:,ind));    
% end
%  
% m = 0.5*size(D,1);
% n1 = size(A,1)/m;
% for i = 1:npaths
%   ind = (nelems(i)+1):nelems(i+1);
%   nep = length(ind);
%   x = pathextract(F, fpath(:,ind), fintf(:,ind));
%   x = x(:);
%   y = pathsolve(A(:,:,ind), B(:,:,ind), C(:,:,ind), D(:,:,ind), x);  
%   H = form_fullmatrix_on_path(AE, epath(ind), fpath(:,ind), fintf(:,ind), mesh1.f2t, mesh1.t2f, mesh1.elcon);
%   z = H\x;
%   max(abs(z-y))
% end
% 
% 

% [pde,mesh,master,dmd] = preprocessing(pde,mesh);
% pde.codegenerator = "text2code";
% kkgencode(pde);
% compilerstr = compilepdemodel(pde);
% runstr = runcode(pde, 1); % run C++ code
% sol = fetchsolution(pde,master,dmd, pde.buildpath + '/dataout');
