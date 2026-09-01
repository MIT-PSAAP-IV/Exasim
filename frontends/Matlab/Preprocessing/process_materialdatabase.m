function destination = process_materialdatabase(pde, output_dir, output_name)
%PROCESS_MATERIALDATABASE Stage an optional material database for the backend.
%
% If pde.materialdatabase is empty, no file is generated. Binary databases are
% copied byte-for-byte. Text databases are converted to Exasim's standard raw
% double binary layout, preserving the numeric values in file order.

if nargin < 3 || isempty(strtrim(char(output_name)))
    output_name = 'materialdatabase.bin';
end
destination = '';

if ~isfield(pde, 'materialdatabase') || isempty(strtrim(char(pde.materialdatabase)))
    return;
end

source = strtrim(char(pde.materialdatabase));
if ~exist(source, 'file')
    error('pde.materialdatabase file not found: %s', source);
end

destination = fullfile(char(output_dir), char(output_name));
[~,~,ext] = fileparts(source);
ext = lower(ext);

if strcmp(ext, '.bin')
    [ok,msg] = copyfile(source, destination);
    if ~ok
        error('Could not copy pde.materialdatabase to %s: %s', destination, msg);
    end
elseif strcmp(ext, '.dat')
    values = read_materialdatabase_dat_values(source);
    writebin(destination, values);
else
    error('Unsupported pde.materialdatabase format ''%s''. Expected .dat or .bin.', ext);
end
end

function values = read_materialdatabase_dat_values(filename)
fid = fopen(filename, 'r');
if fid < 0
    error('Could not open pde.materialdatabase file for reading: %s', filename);
end
cleanup = onCleanup(@() fclose(fid));

values = [];
while true
    line = fgetl(fid);
    if ~ischar(line)
        break;
    end
    line = regexprep(line, '(#|%|//).*$', '');
    nums = sscanf(line, '%f');
    if ~isempty(nums)
        values = [values; nums(:)]; %#ok<AGROW>
    end
end
end
