function [sol,pde,mesh] = hdgldgsolve(pde,mesh,labels,nhdg)
%HDGLDGSOLVE Generate, preprocess and solve a coupled CPU HDG/LDG problem.
% labels(e) selects HDG (1) or LDG (0) for element e in mesh.t.
% The first nhdg MPI world ranks use HDG; the remaining ranks use LDG.
% nhdg defaults to 1.
% Alternatively, pass struct('seed_boundaries',[1],'layers',2) instead of
% labels to select two face-neighbour layers from boundary-condition code 1.
%
% Each call creates a fresh directory under pde.datapath. The returned sol
% contains all nc components in the ORIGINAL mesh.t element order.
% mesh.hdgldg is a row vector in that same order: 1 for HDG, 0 for LDG.
% The pure-scheme exasim entry point is unchanged.

if nargin < 4
    nhdg = 1;
end
boundaryLayers = isstruct(labels);
if ~boundaryLayers
    labels = double(labels(:)');
end
ne = size(mesh.t,2);

% 1. Check the supported model and the element/rank selection.
if pde.platform ~= "cpu" || pde.model ~= "ModelD" || pde.ncw ~= 0 || pde.nco ~= 0
    error('hdgldg:scope', ...
        'This frontend workflow supports CPU ModelD without auxiliary fields.');
end
invalidRanks = pde.mpiprocs < 2 || nhdg < 1 || nhdg >= pde.mpiprocs || fix(nhdg) ~= nhdg;
if boundaryLayers
    validSelection = isscalar(labels) && isfield(labels,'seed_boundaries') && ...
        isfield(labels,'layers');
    if validSelection
        seeds = labels.seed_boundaries;
        layers = labels.layers;
        validSelection = isnumeric(seeds) && isvector(seeds) && ~isempty(seeds) && ...
            all(isfinite(seeds(:)) & seeds(:) >= 1 & seeds(:) == fix(seeds(:))) && ...
            isnumeric(layers) && isscalar(layers) && isfinite(layers) && ...
            layers >= 1 && layers == fix(layers);
    end
    if ~validSelection
        error('hdgldg:selection','Provide positive integer seed_boundaries and layers.');
    end
    invalidLabels = false; % The backend checks the selected H/L region sizes.
else
    invalidLabels = numel(labels) ~= ne || any(labels ~= 0 & labels ~= 1) || ...
        ~any(labels == 0) || ~any(labels == 1);
end
if invalidRanks || invalidLabels
    error('hdgldg:selection', ...
        'Provide one H/L label per element and nonempty H/L rank groups.');
end

prefix = exasim_install_prefix();
preprocessor = fullfile(prefix,'bin','exasim_hdgldg_preprocess');
if ~isfile(preprocessor)
    error('hdgldg:install', ...
        'Rebuild/reinstall Exasim with CPU preprocessing: missing %s',preprocessor);
end

% 2. Create a fresh run directory and generate the native model kernels.
if ~isfolder(pde.datapath)
    mkdir(pde.datapath);
end
work = string(tempname(pde.datapath));
%work = string(pde.datapath);
mkdir(work);
inputDirectory = fullfile(work,'input');
mkdir(inputDirectory);

pde.datapath = work;
pde.dataoutpath = fullfile(work,'dataout');
pde.hybrid = 1; % Generate both kernel families, not a mixed trace layout.
pde.nd = size(mesh.p,1);
pde.ncx = pde.nd;
pde.nch = pde.ncu;
pde.ncq = pde.nd*pde.ncu;
pde.nc = pde.ncu+pde.ncq;
pde.modelid = resolve_modelid(pde);
pde.builtinmodelID = pde.modelid;
pde.saveSolOpt = 1;
pde.saveSolFreq = 1;
kkgencode(pde);
cmakecompile(pde);

% 3. Write native inputs. Shared CPU preprocessing performs selection,
%    partitioning and H/L connectivity construction.
writeinputs(pde,mesh,labels,nhdg,inputDirectory);
configFile = fullfile(inputDirectory,'pdeapp.txt');
selectionFile = fullfile(inputDirectory,'selection.txt');
datain = fullfile(work,'datain');
hdgldgpreprocessing(configFile,selectionFile,datain,preprocessor);

% 4. Solve one coupled MPI problem, then restore original element numbering.
runcoupledsolver(pde,datain,work);
[sol,mesh.hdgldg] = collectsolution(datain,pde.dataoutpath,ne,nhdg,pde.mpiprocs-nhdg);
fprintf('HDG/LDG results: %s\n',pde.dataoutpath);
end

function writeinputs(pde,mesh,labels,nhdg,inputDirectory)
% Reuse native serializers rather than another frontend mesh algorithm.
% The binary exporter concatenates p and t. An integer t would convert the
% coordinates to integers too; change only the exported copy, not mesh.
exportMesh = mesh;
exportMesh.t = double(mesh.t);
exporttext2codemesh(exportMesh,inputDirectory);

% writeinputfile uses the current directory for related input files.
originalDirectory = pwd;
restoreDirectory = onCleanup(@() cd(originalDirectory));
cd(inputDirectory);
app = pde;
app.meshfile = "grid.bin";
writeinputfile('pdeapp.txt',app,mesh);

fid = fopen('pdeapp.txt','a');
if fid < 0
    error('hdgldg:input','Cannot finish input metadata.');
end
fprintf(fid,'nsca = %d;\nnvec = %d;\nnten = %d;\nnsurf = %d;\n', ...
    pde.nsca,pde.nvec,pde.nten,pde.nbqoi);
fprintf(fid,'nvqoi = %d;\nnmaterialstate = %d;\nnsurfq = %d;\n', ...
    pde.nvqoi,pde.nmaterialstate,pde.nsurfq);
fclose(fid);
clear restoreDirectory; % Return to the caller's directory.

fid = fopen(fullfile(inputDirectory,'selection.txt'),'w');
if fid < 0
    error('hdgldg:input','Cannot write H/L selection.');
end
% Backend MPI rank numbers are zero-based.
hdgRanks = join(string(0:nhdg-1),',');
ldgRanks = join(string(nhdg:pde.mpiprocs-1),',');
fprintf(fid,'version = 1\nworld_size = %d\n',pde.mpiprocs);
fprintf(fid,'h_ranks = [%s]\nl_ranks = [%s]\n',hdgRanks,ldgRanks);
if isstruct(labels)
    fprintf(fid,'selection = boundary_layers\nseed_boundaries = [%s]\nlayers = %d\n', ...
        join(string(labels.seed_boundaries(:)'),','),labels.layers);
else
    fprintf(fid,'selection = labels\nlabels = [%s]\n',join(string(labels),','));
end
fclose(fid);
end

function runcoupledsolver(pde,datain,work)
buildDirectory = fullfile(model_builddir(pde),'build');
executable = fullfile(buildDirectory,'exasimapp');
mpirun = string(pde.mpirun);

% Prefer the MPI launcher recorded by the application build, when present.
launcherFile = fullfile(buildDirectory,'mpiexec.txt');
if isfile(launcherFile)
    mpirun = strtrim(string(fileread(launcherFile)));
end
arguments = {mpirun,'-np',num2str(pde.mpiprocs),executable, ...
    '--hdgldg',datain,pde.dataoutpath};
quotedArguments = cellfun(@shellquote,arguments,'UniformOutput',false);
command = strjoin(quotedArguments,' ');
[status,output] = system(command);
fprintf('%s',output);

logFile = fullfile(work,'solver.log');
fid = fopen(logFile,'w');
if fid >= 0
    fprintf(fid,'%s\n%s',command,output);
    fclose(fid);
end
if status ~= 0
    error('hdgldg:solve','Coupled solver failed; see %s',logFile);
end
end

function [sol,elementSchemes] = collectsolution(datain,dataout,ne,nhdg,nldg)
% Each rank writes local output. Its sidecar maps owned elements back to
% the input mesh; ghost elements must not be assembled a second time.
sol = [];
elementSchemes = zeros(1,ne);
seen = false(1,ne);
regions = {'H','L'};
rankCounts = [nhdg,nldg];

for regionIndex = 1:2
    region = regions{regionIndex};
    rankCount = rankCounts(regionIndex);
    for rank = 0:rankCount-1
        solutionFile = fullfile(dataout,region,sprintf('outudg_np%d.bin',rank));
        fid = fopen(solutionFile,'rb');
        if fid < 0
            error('hdgldg:output','Missing %s',solutionFile);
        end
        dimensions = fread(fid,3,'double')'; % [nodes, components, owned elements]
        values = fread(fid,Inf,'double');
        fclose(fid);
        if numel(dimensions) ~= 3 || any(dimensions <= 0) || ...
                any(dimensions ~= fix(dimensions)) || isempty(values) || ...
                mod(numel(values),prod(dimensions)) ~= 0 || any(~isfinite(values))
            error('hdgldg:output','Invalid native solution: %s',solutionFile);
        end
        snapshotCount = numel(values)/prod(dimensions);
        values = reshape(values,[dimensions,snapshotCount]);

        % Serial files have no suffix; parallel files use local rank+1.
        suffix = '';
        if rankCount > 1
            suffix = num2str(rank+1);
        end
        mappingFile = fullfile(datain,region,['hdgldg' suffix '.txt']);
        fid = fopen(mappingFile,'r');
        if fid < 0
            error('hdgldg:output','Missing %s',mappingFile);
        end
        fgetl(fid); % Format/version header.
        fgetl(fid); % World rank, region rank and scheme.
        localCount = str2double(fgetl(fid));
        elementMap = fscanf(fid,'%d',[3,localCount]);
        fclose(fid);

        % Owned elements precede ghosts. Row 2 holds zero-based input IDs.
        ownedCount = dimensions(3);
        ids = elementMap(2,1:ownedCount)+1;
        if any(ids < 1 | ids > ne) || any(seen(ids)) || numel(unique(ids)) ~= numel(ids)
            error('hdgldg:output','Duplicate or invalid owned global element IDs.');
        end
        if isempty(sol)
            sol = zeros(dimensions(1),dimensions(2),ne);
        end
        sol(:,:,ids) = values(:,:,:,end); % Return the final saved snapshot.
        elementSchemes(ids) = strcmp(region,'H');
        seen(ids) = true;
    end
end
if ~all(seen)
    error('hdgldg:output','Some owned elements are missing.');
end
end

function quoted = shellquote(value)
% Quote one POSIX shell argument, including embedded single quotes.
singleQuote = char(39);
doubleQuote = char(34);
escapedQuote = [singleQuote doubleQuote singleQuote doubleQuote singleQuote];
quoted = [singleQuote strrep(char(value),singleQuote,escapedQuote) singleQuote];
end
