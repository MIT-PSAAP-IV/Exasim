function dest = exporttext2codemesh(mesh, dest, suffix)
%EXPORTTEXT2CODEMESH Export Text2Code mesh-related binary files.
%
%   dest = exporttext2codemesh(mesh, dest, suffix)
%
%   This helper writes the mesh files used by Text2Code:
%
%     grid<suffix>.bin = [size(mesh.p) size(mesh.t) mesh.p(:)' mesh.t(:)']
%     xdg<suffix>.bin  = [size(mesh.dgnodes) mesh.dgnodes(:)']
%     udg<suffix>.bin  = [size(mesh.udg) mesh.udg(:)']
%     vdg<suffix>.bin  = [size(mesh.vdg) mesh.vdg(:)']
%     wdg<suffix>.bin  = [size(mesh.wdg) mesh.wdg(:)']
%
%   Optional fields are written only when present and nonempty. The binary
%   layout is the same MATLAB column-major layout used by exporttext2code.

if nargin < 3 || isempty(suffix)
    suffix = "";
end

dest = string(dest);
suffix = string(suffix);

if ~isfolder(dest)
    mkdir(dest);
end

if ~isfield(mesh, 'p') || isempty(mesh.p)
    error('exporttext2codemesh:MissingMeshP', 'mesh.p is required.');
end
if ~isfield(mesh, 't') || isempty(mesh.t)
    error('exporttext2codemesh:MissingMeshT', 'mesh.t is required.');
end

writebin(char(dest + "/grid" + suffix + ".bin"), ...
    [size(mesh.p) size(mesh.t) mesh.p(:)' mesh.t(:)']);

writeoptionalfield(mesh, dest, suffix, 'dgnodes', 'xdg');
writeoptionalfield(mesh, dest, suffix, 'udg', 'udg');
writeoptionalfield(mesh, dest, suffix, 'vdg', 'vdg');
writeoptionalfield(mesh, dest, suffix, 'wdg', 'wdg');

end

function writeoptionalfield(mesh, dest, suffix, fieldname, filenamebase)

if isfield(mesh, fieldname) && ~isempty(mesh.(fieldname))
    data = mesh.(fieldname);
    writebin(char(dest + "/" + filenamebase + suffix + ".bin"), ...
        [size(data) data(:)']);
end

end
