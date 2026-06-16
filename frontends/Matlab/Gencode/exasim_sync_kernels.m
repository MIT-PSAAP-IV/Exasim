function exasim_sync_kernels(srcdir, dstdir)
% Copy generated kernels into dstdir, rewriting only files whose content
% changed, so unchanged models keep their mtimes and skip recompilation.
if ~exist(char(dstdir), 'dir')
    mkdir(char(dstdir));
end
files = dir(char(srcdir));
for i = 1:numel(files)
    if files(i).isdir, continue; end
    src = fullfile(files(i).folder, files(i).name);
    dst = char(dstdir + "/" + files(i).name);
    newtext = fileread(src);
    if exist(dst, 'file') == 2
        oldtext = fileread(dst);
        if strcmp(oldtext, newtext), continue; end
    end
    fid = fopen(dst, 'w');
    fwrite(fid, newtext);
    fclose(fid);
end
rmdir(char(srcdir), 's');
end
