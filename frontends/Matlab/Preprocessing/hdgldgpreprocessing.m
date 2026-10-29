function [hdir, ldir] = hdgldgpreprocessing(config, manifest, outdir, executable)
% Shared CPU preprocessing only; native model generation remains unchanged.
% CONFIG must contain explicit native metadata (including ncu/ncv/ncw), mesh,
% field paths and solver controls. OUTDIR is a new absolute directory. This
% function neither generates model kernels nor starts the coupled solver.

config = char(config); manifest = char(manifest);
outdir = char(outdir); executable = char(executable);
if ispc
    error('hdgldg:platform', 'This launcher requires a POSIX shell.');
end
% if isempty(outdir) || outdir(1) ~= '/' || isfolder(outdir) || isfile(outdir)
%     error('hdgldg:output', 'Output must be a new absolute directory.');
% end
for path = {config, manifest, executable}
    if ~isfile(path{1})
        error('hdgldg:input', 'Input/executable must be a file: %s', path{1});
    end
end
arguments = {executable, config, manifest, outdir, '--native-metadata'};
quoted = cellfun(@shellquote, arguments, 'UniformOutput', false);
[status, text] = system(strjoin(quoted, ' '));
if status ~= 0
    error('hdgldg:backend', 'Preprocessing failed (%d):\n%s', status, text);
end
if ~isfile(fullfile(outdir, 'selection.txt'))
    error('hdgldg:files', 'Preprocessing omitted selection.txt.');
end
hdir = fullfile(outdir, 'H'); ldir = fullfile(outdir, 'L');
checkregion(hdir, 1); checkregion(ldir, 0);
end

function quoted = shellquote(value)
% POSIX single quoting also protects spaces, dollars, backticks and newlines.
sq = char(39); dq = char(34);
quoted = [sq strrep(value, sq, [sq dq sq dq sq]) sq];
end

function checkregion(directory, scheme)
fid = fopen(fullfile(directory, 'hdgldg.meta'), 'r');
if fid < 0
    error('hdgldg:metadata', 'Missing region metadata: %s', directory);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
if ~strcmp(strtrim(fgetl(fid)), 'EXASIM_HDGLDG_REGION 1')
    error('hdgldg:metadata', 'Invalid region metadata: %s', directory);
end
values = sscanf(fgetl(fid), '%d');
if numel(values) < 2 || values(1) ~= scheme || values(2) < 1
    error('hdgldg:metadata', 'Invalid region sizes: %s', directory);
end
count = values(2);
required = {'app.bin', 'master.bin'};
for rank = 1:count
    suffix = '';
    if count > 1, suffix = num2str(rank); end
    required = [required, {['mesh' suffix '.bin'], ['sol' suffix '.bin'], ...
                          ['hdgldg' suffix '.txt']}]; %#ok<AGROW>
end
for k = 1:numel(required)
    if ~isfile(fullfile(directory, required{k}))
        error('hdgldg:files', 'Preprocessing omitted %s', fullfile(directory, required{k}));
    end
end
end
