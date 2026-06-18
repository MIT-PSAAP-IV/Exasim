function d = model_builddir(app)
% Per-model build root under the hidden builddir. Model 0 keeps the historical
% flat layout (builddir itself) so single-model builds stay byte-identical;
% other models nest under models/<n>/ so their kernels/, build/, CMakeLists.txt
% and main.cpp never clobber. Mirrors the Python frontend's config.model_builddir.
strn = model_strn(app);
if strlength(strn) == 0
    d = string(app.builddir);
else
    d = string(app.builddir) + "/models/" + strn;
end
end
