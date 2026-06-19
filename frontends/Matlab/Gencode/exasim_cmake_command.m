function cmakecmd = exasim_cmake_command(prefix)
% Resolve the cmake executable to invoke for building generated apps, quoted.
% Prefer the absolute path of the cmake that built/installed Exasim (recorded at
% install time) so this works when MATLAB is launched from the GUI app, whose
% PATH lacks the Homebrew/conda cmake. Falls back to PATH, then common dirs,
% then a bare "cmake". (Shared by cmakecompile and cmakecompile_combined.)
resolved = "";
cmakerec = char(prefix + "/lib/cmake/Exasim/cmake_command.txt");
if exist(cmakerec, 'file') == 2
    rec = strtrim(string(fileread(cmakerec)));
    if strlength(rec) > 0 && exist(char(rec), 'file') == 2
        resolved = rec;
    end
end
if strlength(resolved) == 0
    [st, out] = system('command -v cmake');
    if st == 0 && strlength(strtrim(string(out))) > 0
        resolved = strtrim(string(out));
    else
        for cand = ["/opt/homebrew/bin/cmake", "/usr/local/bin/cmake", ...
                    "/opt/local/bin/cmake", "/usr/bin/cmake"]
            if exist(char(cand), 'file') == 2
                resolved = cand; break;
            end
        end
    end
end
if strlength(resolved) > 0
    cmakecmd = """" + resolved + """";
else
    cmakecmd = "cmake";
end
end
