function result = run_multiple_interactions(porder,N,domainranks,outdir)
%RUN_MULTIPLE_INTERACTIONS Run the four-domain Poisson example.
if nargin ~= 4
    error('run_multiple_interactions requires porder, N, domainranks, and outdir.');
end
caseDirectory = fileparts(mfilename('fullpath'));
[caseName,dimension,families] = case_configuration(caseDirectory);
assert(isscalar(porder) && porder >= 1 && porder == floor(porder));
assert(isscalar(N) && N >= 1 && N == floor(N));
assert(isnumeric(domainranks) && numel(domainranks) == 4 && ...
    all(domainranks >= 1) && all(domainranks == floor(domainranks)));
assert((ischar(outdir) || (isstring(outdir) && isscalar(outdir))) && ...
    startsWith(char(outdir),filesep),'outdir must be an absolute path');

root = getenv('EXASIM_ROOT');
if isempty(root), root = fileparts(fileparts(fileparts(fileparts(caseDirectory)))); end
root = char(java.io.File(root).getCanonicalPath());
outdir = char(java.io.File(char(outdir)).getCanonicalPath());
assert(~strcmp(outdir,root) && ~startsWith([outdir filesep],[root filesep]), ...
    'outdir must lie outside the Exasim repository');
if ~isfolder(outdir), mkdir(outdir); end

oldDirectory = pwd;
cleanupDirectory = onCleanup(@() cd(oldDirectory)); %#ok<NASGU>
cd(outdir);
requestedPrefix = getenv('EXASIM_PREFIX');
run(fullfile(root,'frontends','Matlab','exasim_setup.m'));
if ~isempty(requestedPrefix), setenv('EXASIM_PREFIX',requestedPrefix); end

runDirectory = outdir; %#ok<NASGU>
pde = cell(4,1); mesh = cell(4,1);
for domain = 1:4
    run(fullfile(caseDirectory,sprintf('pdeapp%d.m',domain)));
    expectedVertices = 3 + families(domain);
    if dimension == 3, expectedVertices = 4 + 4*families(domain); end
    assert(size(mesh{domain}.t,1) == expectedVertices, ...
        'domain %d has the wrong element family',domain);
end
interactions = interaction();

logfile = fullfile(outdir,'solver.log');
diary(logfile); diary on;
try
    [sol,pde,mesh,master,dmd] = exasim(pde,mesh,interactions);
    diary off;
catch exception
    diary off;
    rethrow(exception);
end
logtext = fileread(logfile);
tokens = regexp(logtext,'Updated Norm:\s*([-+0-9.eE]+)','tokens');
assert(~isempty(tokens),'solver log contains no global Newton residual');
finalResidual = str2double(tokens{end}{1});
assert(isfinite(finalResidual) && finalResidual <= pde{1}.NLtol, ...
    'global nonlinear solve did not converge');
records = regexp(logtext,'points off donor face = ([0-9]+)','tokens');
assert(numel(records) >= 6 && all(cellfun(@(x) str2double(x{1}) == 0,records)), ...
    'donor-search validation failed');

maxError = 0;
for domain = 1:4
    values = sol{domain};
    if ndims(values) == 4, values = values(:,:,:,end); end
    assert(all(isfinite(values),'all'),'domain %d solution is non-finite',domain);
    faces = facenumbering(mesh{domain}.p,mesh{domain}.t,families(domain), ...
        mesh{domain}.boundaryexpr,[]);
    xdg = createdgnodes(mesh{domain}.p,mesh{domain}.t,faces,[],[],porder);
    exact = sin(pi*xdg(:,1,:)).*sin(pi*xdg(:,2,:));
    if dimension == 3, exact = exact.*sin(pi*xdg(:,3,:)); end
    maxError = max(maxError,max(abs(values(:,1,:)-exact),[],'all'));
    mesh{domain}.dgnodes = xdg;
end

result = struct('case',caseName,'workdir',outdir,'sol',{sol},'pde',{pde}, ...
    'mesh',{mesh},'master',{master},'dmd',{dmd}, ...
    'finalResidual',finalResidual,'maxError',maxError,'logFile',logfile);
fprintf('Global final residual: %.17e\n',finalResidual);
fprintf('Maximum nodal error: %.17e\n',maxError);
fprintf('EXASIM_OUTDIR=%s\n',outdir);
end

function [name,dimension,families] = case_configuration(caseDirectory)
[~,name] = fileparts(caseDirectory);
switch name
    case 'multiple_interactions_2d'
        dimension = 2; families = [1 1 1 1];
    case 'multiple_interactions_2d_quadtri'
        dimension = 2; families = [1 0 1 0];
    case 'multiple_interactions_3d'
        dimension = 3; families = [1 1 1 1];
    case 'multiple_interactions_3d_hextet'
        dimension = 3; families = [1 0 1 0];
    otherwise
        error('unsupported example directory: %s',name);
end
end
