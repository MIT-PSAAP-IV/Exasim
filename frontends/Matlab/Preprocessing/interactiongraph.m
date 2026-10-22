function [table, selectors, nfint, maxcondition] = interactiongraph(pde, mesh, interaction)
%INTERACTIONGRAPH Validate and encode the generalized type-1 interaction graph.

N = numel(pde);
if N < 2 || ~iscell(pde) || ~iscell(mesh) || numel(mesh) ~= N
    error('interaction graph requires matching pde/mesh cell arrays with at least two domains.');
end
if ~isstruct(interaction) || isempty(interaction)
    error('interaction must be a nonempty struct array.');
end
for d = 1:N
    if isfield(mesh{d},'interfacecondition') && any(mesh{d}.interfacecondition(:) > 0)
        error('generalized interactions cannot be combined with mesh.interfacecondition.');
    end
end
expected = sort({'domains','type','interface1','condition1','fluxmap1', ...
                 'interface2','condition2','fluxmap2'});
selectors = cell(N,1); nfint = ones(N,1); maxcondition = zeros(N,1);
implied = zeros(N,1); used = cell(N,1); records = cell(numel(interaction),1);
facelabels = cell(N,1);
for d = 1:N
    periodicexpr = [];
    if isfield(mesh{d},'periodicexpr'), periodicexpr = mesh{d}.periodicexpr; end
    nd = size(mesh{d}.p,1);
    nve = size(mesh{d}.t,1);
    elemtype = double(((nd == 2) && (nve == 4)) || ((nd == 3) && (nve == 8)));
    [facelabels{d},~,~] = facenumbering(mesh{d}.p,mesh{d}.t, ...
        elemtype,mesh{d}.boundaryexpr,periodicexpr);
end
for k = 1:numel(interaction)
    entry = interaction(k);
    if ~isequal(sort(fieldnames(entry))', expected)
        error('interaction entry %d must contain exactly the eight documented fields.', k);
    end
    domains = entry.domains(:)';
    if numel(domains) ~= 2 || any(domains ~= fix(domains)) || ...
            any(domains < 1) || any(domains > N) || domains(1) == domains(2)
        error('interaction entry %d field domains is invalid.', k);
    end
    if ~isscalar(entry.type) || entry.type ~= 1 || entry.type ~= fix(entry.type)
        error('interaction entry %d: generalized interactions support only type=1.', k);
    end
    endpoint = cell(2,1);
    maps = cell(2,1);
    for side = 1:2
        d = domains(side);
        s = entry.(sprintf('interface%d',side));
        c = entry.(sprintf('condition%d',side));
        map = entry.(sprintf('fluxmap%d',side)); map = map(:)';
        if ~isscalar(s) || s ~= fix(s) || s < 1 || ~isfield(mesh{d},'boundaryexpr') || s > numel(mesh{d}.boundaryexpr)
            error('interaction entry %d field interface%d is invalid.', k, side);
        end
        if isfield(mesh{d},'periodicboundary') && any(mesh{d}.periodicboundary(:) == s)
            error('interaction entry %d field interface%d selects a periodic boundary.', k, side);
        end
        if ~any(facelabels{d}(:) == s)
            error('interaction entry %d field interface%d selects no boundary face.', k, side);
        end
        if ~isfield(mesh{d},'boundarycondition') || s > numel(mesh{d}.boundarycondition)
            error('interaction entry %d field interface%d has no boundary condition.', k, side);
        end
        b = mesh{d}.boundarycondition(s);
        if b <= 0 || sum(mesh{d}.boundarycondition(:) == b) ~= 1
            error('interaction entry %d field interface%d has a nonunique boundary condition.', k, side);
        end
        if any(used{d} == s)
            error('interaction entry %d field interface%d overlaps another interaction.', k, side);
        end
        if ~isscalar(c) || c ~= fix(c) || c < 1
            error('interaction entry %d field condition%d must be a positive integer.', k, side);
        end
        if isempty(map) || any(map ~= fix(map)) || any(map < 1) || ...
                ~isfield(pde{d},'ncu') || any(map > pde{d}.ncu)
            error('interaction entry %d field fluxmap%d is invalid.', k, side);
        end
        model = str2func(pde{d}.modelfile); definition = model();
        if ~isfield(definition,'fint'), error('interaction entry %d domain %d does not define fint.',k,d); end
        used{d}(end+1) = s; selectors{d}(end+1) = s;
        maxcondition(d) = max(maxcondition(d), c);
        maps{side} = map;
        endpoint{side} = [s c numel(map) map-1];
    end
    for side = 1:2
        d = domains(side); width = numel(maps{3-side});
        if implied(d) ~= 0 && implied(d) ~= width
            error('interaction entry %d violates the one fint output width per domain rule.', k);
        end
        implied(d) = width; nfint(d) = width;
    end
    len = 9 + numel(maps{1}) + numel(maps{2});
    records{k} = [len domains-1 1 endpoint{1} endpoint{2}];
end
controls = {'temporalscheme','torder','nstage','dt','NLiter','NLtol', ...
            'linearsolver','linearsolvertol','GMRESrestart','GMRESortho', ...
            'linearsolveriter','preconditioner'};
for d = 2:N
    for j = 1:numel(controls)
        key = controls{j};
        if isfield(pde{1},key) && isfield(pde{d},key) && ~isequal(pde{1}.(key),pde{d}.(key))
            error('interaction graph requires identical solver control %s on every domain.', key);
        end
    end
end
table = [1 N numel(interaction) records{:}];
if any(table >= 2^31), error('interaction table value exceeds the supported 32-bit range.'); end
table = double(table(:));
end
