function frontend_matlab()
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(root,'frontends','Matlab','Preprocessing'));
temp = tempname; mkdir(temp); cleanup = onCleanup(@() rmdir(temp,'s'));
base = initializepde('');
assert(base.compositehdg == 0 && base.compositebatchsize == 1024);
base.coupledinterface = 0; base.coupledcondition = 0; base.coupledboundarycondition = 0;
for cap = [1,127,128,256,512,1024,2048,4096]
    app = base; app.compositehdg = 1; app.compositebatchsize = cap;
    app.interfacetype = 0; app.uniformrefinementlevel = 2;
    path = fullfile(temp,'app.bin'); writeapp(app,path,'native');
    fid = fopen(path,'r'); count = fread(fid,1,'double'); sizes = fread(fid,count,'double');
    dims = fread(fid,sizes(1),'double'); fread(fid,sizes(2),'double');
    problem = fread(fid,sizes(3),'double'); fclose(fid);
    assert(isequal(dims(24:25)',[1 cap]) && dims(23) == 0);
    assert(isequal(problem(34:35)',[2 0]));
    app.meshfile = 'unused.bin';
    mesh.boundarycondition = 1; mesh.boundaryexpr = {@(p) p(1,:) < 0};
    textfile = fullfile(temp,'pdeapp.txt'); writeinputfile(textfile,app,mesh);
    content = fileread(textfile);
    assert(contains(content,'compositehdg = 1;'));
    assert(contains(content,sprintf('compositebatchsize = %d;',cap)));
end
app = base; app.compositehdg = 1; app.interfacetype = 1; rejected = false;
try
    writeapp(app,fullfile(temp,'nonmatching.bin'),'native');
catch error
    rejected = contains(error.message,'not available in this source release');
end
assert(rejected);
for bad = {0,-1,1024.5,NaN,Inf,-Inf,2147483648,1i,[128 256],'1024'}
    app = base; app.compositebatchsize = bad{1}; rejected = false;
    try, writeapp(app,fullfile(temp,'invalid.bin'),'native'); catch, rejected = true; end
    assert(rejected);
end
for bad = {-1,2,0.5,NaN,Inf,1i,[0 1],'1'}
    app = base; app.compositehdg = bad{1}; rejected = false;
    try, writeapp(app,fullfile(temp,'invalid.bin'),'native'); catch, rejected = true; end
    assert(rejected);
end
app = rmfield(base,{'compositehdg','compositebatchsize'});
app = writeapp(app,fullfile(temp,'legacy.bin'),'native');
assert(app.ndims(24) == 0 && app.ndims(25) == 1024);
disp('COMPOSITE_MATLAB_FRONTEND_PASS');
end
