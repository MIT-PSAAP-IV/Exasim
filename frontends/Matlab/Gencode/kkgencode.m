function kkgencode(app)

disp("generate code...");

if app.codegenerator == "text2code"
  runstr = "!" + exasim_install_prefix() + "/bin/text2code " + app.modelfile + ".txt";
  eval(char(runstr));
  return;
end

% Generated kernels go to the per-model build dir; cmakecompile points the
% external-model provider's include path here (no source-tree writes).
% Generate into a staging dir, then sync write-if-changed into kernels/ so
% an unchanged model does not touch mtimes (and thus avoids recompiles).
%
% External-model path (single model + combined multi-PDE): unsuffixed kernels
% in the per-model build dir (model 0 stays flat = builddir, byte-identical).
% Legacy path (interfacecondition coupling via kkgencodeall): suffixed kernels
% in the shared builddir, unchanged. The app.combinedmodel flag selects.
combined = isfield(app, 'combinedmodel') && app.combinedmodel;
if combined
    mbdir = model_builddir(app);
else
    mbdir = string(app.builddir);
end
kkdir = mbdir + "/kernels.gen";
if exist(char(kkdir), 'dir')
    rmdir(char(kkdir), 's');
end
mkdir(char(kkdir));

hdggencode(app);

[xdg, udg, udg1, udg2, wdg, wdg1, wdg2, odg, odg1, odg2, uhg, nlg, tau, uinf, param, time] = syminit(app);
pdemodel = str2func(app.modelfile);
pde = pdemodel();

if combined || app.modelnumber==0
    strn = "";        % unsuffixed: isolation is by per-model directory
else
    strn = num2str(app.modelnumber);   % legacy suffix for kkgencodeall
end

ncu = app.ncu;
u = udg(1:ncu);
u1 = udg1(1:ncu);
u2 = udg2(1:ncu);
if app.nc>app.ncu
    q = udg(ncu+1:end);
    q1 = udg1(ncu+1:end);
    q2 = udg2(ncu+1:end);
else
    q = [];
    q1 = [];
    q2 = [];
end

if isfield(pde, 'flux')    
    f = pde.flux(u, q, wdg, odg, xdg, time, param, uinf);    
    kkgencodeelem("Flux" + strn, f, xdg, udg, odg, wdg, uinf, param, time, kkdir);       
else
    error("pde.flux is not defined");
end
if isfield(pde, 'source')
    f = pde.source(u, q, wdg, odg, xdg, time, param, uinf);
    kkgencodeelem("Source" + strn, f, xdg, udg, odg, wdg, uinf, param, time, kkdir);    
else    
    error("pde.source is not defined");
end
if isfield(pde, 'materialstate')
    f = pde.materialstate(u, q, wdg, odg, xdg, time, param, uinf);
    f = f(:);
    app.nmaterialstate = length(f);
    kkgencodematerialstate("Materialstate" + strn, f, xdg, udg, odg, wdg, uinf, param, time, kkdir);
    if app.hybrid == 1
        hdgkkgencodematerialstate("Materialstate" + strn, f, xdg, udg, odg, wdg, uinf, param, time, kkdir);
    else
        hdgkknocodematerialstate("Materialstate" + strn, kkdir);
    end
else
    if ~isfield(app, 'nmaterialstate')
        app.nmaterialstate = 0;
    end
    kknocodematerialstate("Materialstate" + strn, kkdir);
    hdgkknocodematerialstate("Materialstate" + strn, kkdir);
end
if isfield(pde, 'visscalars')
    f = pde.visscalars(u, q, wdg, odg, xdg, time, param, uinf);
    kkgencodeelem("VisScalars" + strn, f, xdg, udg, odg, wdg, uinf, param, time, kkdir);    
else    
    kknocodeelem("VisScalars" + strn, kkdir);
end
if isfield(pde, 'visvectors')
    f = pde.visvectors(u, q, wdg, odg, xdg, time, param, uinf);
    kkgencodeelem("VisVectors" + strn, f, xdg, udg, odg, wdg, uinf, param, time, kkdir);    
else    
    kknocodeelem("VisVectors" + strn, kkdir);
end
if isfield(pde, 'vistensors')
    f = pde.vistensors(u, q, wdg, odg, xdg, time, param, uinf);
    kkgencodeelem("VisTensors" + strn, f, xdg, udg, odg, wdg, uinf, param, time, kkdir);    
else    
    kknocodeelem("VisTensors" + strn, kkdir);
end
if isfield(pde, 'qoivolume')
    f = pde.qoivolume(u, q, wdg, odg, xdg, time, param, uinf);
    kkgencodeelem("QoIvolume" + strn, f, xdg, udg, odg, wdg, uinf, param, time, kkdir);    
else    
    kknocodeelem("QoIvolume" + strn, kkdir);
end
if isfield(pde, 'eos')
    f = pde.eos(u, q, wdg, odg, xdg, time, param, uinf);
    kkgencodeelem2("EoS" + strn, f, xdg, udg, odg, wdg, uinf, param, time, kkdir);
    
    nf = length(f);
    nu = length(u);
    nw = length(wdg);
    
    dfdu = sym(zeros(nf,nu));
    for m = 1:nf
      for n = 1:nu
        dfdu(m,n) = diff(f(m),u(n));      
      end
    end
    kkgencodeelem2("EoSdu" + strn, dfdu, xdg, udg, odg, wdg, uinf, param, time, kkdir);    
    
    dfdw = sym(zeros(nf,nw));
    for m = 1:nf
      for n = 1:length(wdg)
        dfdw(m,n) = diff(f(m),wdg(n));      
      end
    end
    kkgencodeelem2("EoSdw" + strn, dfdw, xdg, udg, odg, wdg, uinf, param, time, kkdir);    
else    
    kknocodeelem2("EoS" + strn, kkdir);
    kknocodeelem2("EoSdu" + strn, kkdir);
    kknocodeelem2("EoSdw" + strn, kkdir);
end
if isfield(pde, 'sourcew')    
    f = pde.sourcew(u, q, wdg, odg, xdg, time, param, uinf);
    kkgencodeelem2("Sourcew" + strn, f, xdg, udg, odg, wdg, uinf, param, time, kkdir);
else    
    kknocodeelem2("Sourcew" + strn, kkdir);
end
if isfield(pde, 'mass')
    f = pde.mass(u, q, wdg, odg, xdg, time, param, uinf);
    kkgencodeelem("Tdfunc"  + strn, f, xdg, udg, odg, wdg, uinf, param, time, kkdir);
else    
    if app.model=="ModelW" || app.model == "modelW" || app.tdep==1
        error("pde.mass is not defined");
    else        
        kknocodeelem("Tdfunc" + strn, kkdir);
    end                
end
if isfield(pde, 'avfield')
    f = pde.avfield(u, q, wdg, odg, xdg, time, param, uinf);
    kkgencodeelem2("Avfield" + strn, f, xdg, udg, odg, wdg, uinf, param, time, kkdir);
else    
    kknocodeelem2("Avfield" + strn, kkdir);
end
if isfield(pde, 'output')
    f = pde.output(u, q, wdg, odg, xdg, time, param, uinf);
    kkgencodeelem2("Output" + strn, f, xdg, udg, odg, wdg, uinf, param, time, kkdir);
else    
    kknocodeelem2("Output" + strn, kkdir);
end
if isfield(pde, 'monitor')
    f = pde.monitor(u, q, wdg, odg, xdg, time, param, uinf);
    kkgencodeelem2("Monitor" + strn, f, xdg, udg, odg, wdg, uinf, param, time, kkdir);
else    
    kknocodeelem2("Monitor" + strn, kkdir);
end
if isfield(pde, 'fbou')    
    f = pde.fbou(u, q, wdg, odg, xdg, time, param, uinf, uhg, nlg, tau);
    f = reshape(f,ncu,[]);
    kkgencodeface("Fbou" + strn, f, xdg, udg, odg, wdg, uhg, nlg, tau, uinf, param, time, kkdir, true);
else
    % disp("WARNING: fbou is not defined in the PDE model")
    error("pde.fbou is not defined");
end
if isfield(pde, 'ubou')
    f = pde.ubou(u, q, wdg, odg, xdg, time, param, uinf, uhg, nlg, tau);
    f = reshape(f,ncu,[]);
    kkgencodeface("Ubou" + strn, f, xdg, udg, odg, wdg, uhg, nlg, tau, uinf, param, time, kkdir, true);
else
    % disp("WARNING: ubou is not defined in the PDE model")
    error("pde.ubou is not defined");
end
if isfield(pde, 'qoiboundary')    
    f = pde.qoiboundary(u, q, wdg, odg, xdg, time, param, uinf, uhg, nlg, tau);
    kkgencodeface("QoIboundary" + strn, f, xdg, udg, odg, wdg, uhg, nlg, tau, uinf, param, time, kkdir);
else
    kknocodeface("QoIboundary" + strn, kkdir);
end
if isfield(pde, 'fhat')    
    f = pde.fhat(u1, q1, wdg1, odg1, xdg, time, param, uinf, uhg, nlg, tau, u2, q2, wdg2, odg2);
    kkgencodeface2("Fhat" + strn, f, xdg, udg1, udg2, odg1, odg2, wdg1, wdg2, uhg, nlg, tau, uinf, param, time, kkdir);
else
    kknocodeface2("Fhat" + strn, kkdir);
end
if isfield(pde, 'uhat')
    f = pde.uhat(u1, q1, wdg1, odg1, xdg, time, param, uinf, uhg, nlg, tau, u2, q2, wdg2, odg2);
    kkgencodeface2("Uhat" + strn, f, xdg, udg1, udg2, odg1, odg2, wdg1, wdg2, uhg, nlg, tau, uinf, param, time, kkdir);
else
    kknocodeface2("Uhat" + strn, kkdir);
end
if isfield(pde, 'stab')
    f = pde.stab(u1, q1, wdg1, odg1, xdg, time, param, uinf, uhg, nlg, tau, u2, q2, wdg2, odg2);
    kkgencodeface3("Stab" + strn, f, xdg, udg1, udg2, odg1, odg2, wdg1, wdg2, uhg, nlg, tau, uinf, param, time, kkdir);
else
    kknocodeface2("Stab" + strn, kkdir);
end
if isfield(pde, 'initu')
    udg = pde.initu(xdg, param, uinf);
    kkgencodeelem3("Initu" + strn, udg, xdg, uinf, param, kkdir);
    kkgencodeelem4("Initu" + strn, udg, xdg, uinf, param, kkdir);
else
    error("pde.initu is not defined");
end 
if isfield(pde, 'initw')
    wdg = pde.initw(xdg, param, uinf);
    kkgencodeelem3("Initwdg" + strn, wdg, xdg, uinf, param, kkdir);
    kkgencodeelem4("Initwdg" + strn, wdg, xdg, uinf, param, kkdir);
else
    if app.model=="ModelW" || app.model == "modelW"
        error("pde.initw is not defined");
    else            
        kknocodeelem3("Initwdg" + strn, kkdir);
        kknocodeelem4("Initwdg" + strn, kkdir);
    end
end
if isfield(pde, 'initv')
    odg = pde.initv(xdg, param, uinf);
    kkgencodeelem3("Initodg" + strn, odg, xdg, uinf, param, kkdir);
    kkgencodeelem4("Initodg" + strn, odg, xdg, uinf, param, kkdir);
else
    kknocodeelem3("Initodg" + strn, kkdir);
    kknocodeelem4("Initodg" + strn, kkdir);
end
if isfield(pde, 'initq')
    u = pde.initu(xdg, param, uinf);
    q = pde.initq(xdg, param, uinf);
    udg = [u(:); q(:)];
    kkgencodeelem3("Initudg" + strn, udg, xdg, uinf, param, kkdir);
    kkgencodeelem3("Initq" + strn, udg, xdg, uinf, param, kkdir);
    kkgencodeelem4("Initudg" + strn, udg, xdg, uinf, param, kkdir);
    kkgencodeelem4("Initq" + strn, udg, xdg, uinf, param, kkdir);
else
    if app.model=="ModelW" || app.model == "modelW"
        error("pde.initq is not defined");
    else
        kknocodeelem3("Initudg" + strn, kkdir);
        kknocodeelem3("Initq" + strn, kkdir);
        kknocodeelem4("Initudg" + strn, kkdir);
        kknocodeelem4("Initq" + strn, kkdir);
    end
end 

% Write model_sizes.hpp so the FrontendGenerated provider can get
% compile-time size constants without requiring them in pdeapp.txt.
ncu_   = app.ncu;
nco_   = app.nco;
ncw_   = app.ncw;
nsca_  = app.nsca;
nvec_  = app.nvec;
nten_  = app.nten;
nsurf_ = app.nbqoi;
nvqoi_ = app.nvqoi;
nmaterialstate_ = app.nmaterialstate;
fid = fopen(kkdir + "/model_sizes.hpp", "w");
fprintf(fid, "#ifndef EXASIM_MODEL_SIZES_HPP\n");
fprintf(fid, "#define EXASIM_MODEL_SIZES_HPP\n");
fprintf(fid, "\n");
fprintf(fid, "namespace exasim_model_sizes {\n");
fprintf(fid, "    static constexpr int ncu   = %d;\n", ncu_);
fprintf(fid, "    static constexpr int nco   = %d;\n", nco_);
fprintf(fid, "    static constexpr int ncw   = %d;\n", ncw_);
fprintf(fid, "    static constexpr int nsca  = %d;\n", nsca_);
fprintf(fid, "    static constexpr int nvec  = %d;\n", nvec_);
fprintf(fid, "    static constexpr int nten  = %d;\n", nten_);
fprintf(fid, "    static constexpr int nsurf = %d;\n", nsurf_);
fprintf(fid, "    static constexpr int nvqoi = %d;\n", nvqoi_);
fprintf(fid, "    static constexpr int nmaterialstate = %d;\n", nmaterialstate_);
fprintf(fid, "}\n");
fprintf(fid, "\n");
fprintf(fid, "#endif\n");
fclose(fid);

exasim_sync_kernels(kkdir, mbdir + "/kernels");
end
