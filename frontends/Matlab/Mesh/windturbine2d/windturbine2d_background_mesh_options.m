function opts = windturbine2d_background_mesh_options(opts)
%WINDTURBINE2D_BACKGROUND_MESH_OPTIONS Validate background mesh controls.

if ~isfield(opts, 'backgroundMeshMode') || isempty(opts.backgroundMeshMode)
    opts.backgroundMeshMode = 'graded';
end
opts.backgroundMeshMode = lower(string(opts.backgroundMeshMode));

if ~isfield(opts, 'backgroundMeshSize') || isempty(opts.backgroundMeshSize)
    opts.backgroundMeshSize = 0.8;
end
if ~isfield(opts, 'backgroundMeshSizeFar') || isempty(opts.backgroundMeshSizeFar)
    opts.backgroundMeshSizeFar = 1.0;
end
if ~isfield(opts, 'backgroundMeshSizeNearBlade') || isempty(opts.backgroundMeshSizeNearBlade)
    opts.backgroundMeshSizeNearBlade = 0.35;
end
if ~isfield(opts, 'backgroundMeshSizeInterface')
    opts.backgroundMeshSizeInterface = [];
end
if ~isfield(opts, 'backgroundMeshDistMin')
    opts.backgroundMeshDistMin = [];
end
if ~isfield(opts, 'backgroundMeshSizeTransition') || isempty(opts.backgroundMeshSizeTransition)
    opts.backgroundMeshSizeTransition = 2.0;
end
if ~isfield(opts, 'backgroundMeshGrowthRate') || isempty(opts.backgroundMeshGrowthRate)
    opts.backgroundMeshGrowthRate = 1.3;
end
if ~isfield(opts, 'backgroundMeshMinSize') || isempty(opts.backgroundMeshMinSize)
    opts.backgroundMeshMinSize = min(opts.backgroundMeshSize, opts.backgroundMeshSizeNearBlade);
end
if ~isfield(opts, 'backgroundMeshMaxSize') || isempty(opts.backgroundMeshMaxSize)
    opts.backgroundMeshMaxSize = max(opts.backgroundMeshSize, opts.backgroundMeshSizeFar);
end
if ~isfield(opts, 'farfieldBoundarySpacing') || isempty(opts.farfieldBoundarySpacing)
    opts.farfieldBoundarySpacing = opts.backgroundMeshSizeFar;
end
if ~isfield(opts, 'minFarfieldPoints') || isempty(opts.minFarfieldPoints)
    opts.minFarfieldPoints = 32;
end
if ~isfield(opts, 'maxFarfieldPoints') || isempty(opts.maxFarfieldPoints)
    opts.maxFarfieldPoints = 256;
end
if ~isfield(opts, 'plotFarfieldBoundary') || isempty(opts.plotFarfieldBoundary)
    opts.plotFarfieldBoundary = false;
end
if ~isfield(opts, 'backgroundElementType') || isempty(opts.backgroundElementType)
    if isfield(opts, 'backgroundElemtype') && ~isempty(opts.backgroundElemtype)
        if opts.backgroundElemtype == 0
            opts.backgroundElementType = "tri";
        elseif opts.backgroundElemtype == 1
            opts.backgroundElementType = "quad";
        else
            error('backgroundElemtype must be 0 for tri or 1 for quad.');
        end
    else
        opts.backgroundElementType = "tri";
    end
end
[opts.backgroundElementType, opts.backgroundElemtype] = local_background_element_type(opts.backgroundElementType);
if ~isfield(opts, 'preserveInterfaceSegments') || isempty(opts.preserveInterfaceSegments)
    opts.preserveInterfaceSegments = true;
end
if ~isfield(opts, 'allowNonconformingQuadInterface') || isempty(opts.allowNonconformingQuadInterface)
    opts.allowNonconformingQuadInterface = false;
end
if ~(islogical(opts.preserveInterfaceSegments) || ...
        (isnumeric(opts.preserveInterfaceSegments) && isscalar(opts.preserveInterfaceSegments)))
    error('preserveInterfaceSegments must be a scalar logical value.');
end
if ~(islogical(opts.allowNonconformingQuadInterface) || ...
        (isnumeric(opts.allowNonconformingQuadInterface) && isscalar(opts.allowNonconformingQuadInterface)))
    error('allowNonconformingQuadInterface must be a scalar logical value.');
end
opts.preserveInterfaceSegments = logical(opts.preserveInterfaceSegments);
opts.allowNonconformingQuadInterface = logical(opts.allowNonconformingQuadInterface);
if opts.preserveInterfaceSegments && opts.backgroundElementType == "quad" && ...
        ~opts.allowNonconformingQuadInterface
    error(['Exact blade-interface preservation is currently supported for ' ...
        'triangular background meshes only. Gmsh full-quad recombination ' ...
        'subdivides or rejects one-segment blade curves, which breaks the ' ...
        'one-to-one blade/background interface. Use backgroundElementType="tri", ' ...
        'or set allowNonconformingQuadInterface=true only for diagnostics.']);
elseif opts.preserveInterfaceSegments && opts.backgroundElementType == "quad"
    warning(['Disabling preserveInterfaceSegments for quad diagnostics. ' ...
        'The resulting background mesh is not guaranteed to be conforming at ' ...
        'the structured blade interfaces.']);
    opts.preserveInterfaceSegments = false;
end

switch opts.backgroundMeshMode
    case "uniform"
        local_positive(opts.backgroundMeshSize, 'backgroundMeshSize');
        opts.backgroundMeshMinSize = min(opts.backgroundMeshMinSize, opts.backgroundMeshSize);
        opts.backgroundMeshMaxSize = max(opts.backgroundMeshMaxSize, opts.backgroundMeshSize);
    case "graded"
        local_positive(opts.backgroundMeshSizeFar, 'backgroundMeshSizeFar');
        local_positive(opts.backgroundMeshSizeNearBlade, 'backgroundMeshSizeNearBlade');
        if ~isempty(opts.backgroundMeshSizeInterface)
            local_positive(opts.backgroundMeshSizeInterface, 'backgroundMeshSizeInterface');
        end
        if ~isempty(opts.backgroundMeshDistMin)
            local_positive(opts.backgroundMeshDistMin, 'backgroundMeshDistMin');
        end
        local_positive(opts.backgroundMeshSizeTransition, 'backgroundMeshSizeTransition');
        local_positive(opts.backgroundMeshGrowthRate, 'backgroundMeshGrowthRate');
        if opts.backgroundMeshSizeNearBlade > opts.backgroundMeshSizeFar
            error('backgroundMeshSizeNearBlade must be <= backgroundMeshSizeFar in graded mode.');
        end
        if ~isempty(opts.backgroundMeshSizeInterface) && ...
                opts.backgroundMeshSizeInterface > opts.backgroundMeshSizeFar
            error('backgroundMeshSizeInterface must be <= backgroundMeshSizeFar in graded mode.');
        end
    otherwise
        error('Unsupported backgroundMeshMode: %s.', opts.backgroundMeshMode);
end

local_positive(opts.backgroundMeshMinSize, 'backgroundMeshMinSize');
local_positive(opts.backgroundMeshMaxSize, 'backgroundMeshMaxSize');
local_positive(opts.farfieldBoundarySpacing, 'farfieldBoundarySpacing');
if opts.backgroundMeshMinSize > opts.backgroundMeshMaxSize
    error('backgroundMeshMinSize must be <= backgroundMeshMaxSize.');
end
if ~(isnumeric(opts.minFarfieldPoints) && isscalar(opts.minFarfieldPoints) && ...
        isfinite(opts.minFarfieldPoints) && opts.minFarfieldPoints >= 8)
    error('minFarfieldPoints must be a finite scalar >= 8.');
end
if ~(isnumeric(opts.maxFarfieldPoints) && isscalar(opts.maxFarfieldPoints) && ...
        isfinite(opts.maxFarfieldPoints) && opts.maxFarfieldPoints >= opts.minFarfieldPoints)
    error('maxFarfieldPoints must be a finite scalar >= minFarfieldPoints.');
end
end

function local_positive(value, name)
if ~(isnumeric(value) && isscalar(value) && isfinite(value) && value > 0)
    error('%s must be a positive finite scalar.', name);
end
end

function [name, elemtype] = local_background_element_type(value)
name = lower(string(value));
switch name
    case {"tri", "triangle", "triangular"}
        name = "tri";
        elemtype = 0;
    case {"quad", "quadrilateral"}
        name = "quad";
        elemtype = 1;
    otherwise
        error('Unsupported backgroundElementType: %s.', name);
end
end
