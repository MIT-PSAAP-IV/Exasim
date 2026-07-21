function dest_path = genpdemodel(pde, dest_path)
% GENPDEMODEL  Write a text2code-compatible pdemodel.txt from the symbolic model.
%
%   genpdemodel(pde, dest_path) emits the higher-level text2code DSL
%   (pdemodel.txt) so that an *exported* app can regenerate its kernels from a
%   portable text file via the installed text2code tool, without re-running the
%   MATLAB frontend.
%
% This is the MATLAB port of frontends/Python/exasim/Gencode/genpdemodel.py.
% It mirrors the per-function dispatch in kkgencode.m (same u/q split out of
% udg, same set of model functions) but, instead of the CSE/flatten Kokkos
% path (symsassign/varsassign), it emits plain infix C++/SymEngine expressions:
% text2code function bodies accept them directly.
%
% Symbol renaming (frontend symbol -> text2code 0-based vector indexing):
%   xdgK   -> x[K-1]      (coordinates)
%   udgK   -> uq[K-1]     (u and q concatenated == uq)
%   wdgK   -> w[K-1]
%   odgK   -> v[K-1]
%   uhgK   -> uhat[K-1]
%   nlgK   -> n[K-1]
%   tauK   -> tau[K-1]
%   paramK -> mu[K-1]
%   uinfK  -> eta[K-1]
%   time   -> t
%
% NOTE on MATLAB-vs-Python printing: MATLAB's ccode inlines pi as a numeric
% double literal (3.141592653589793), so unlike the SymPy printer there is no
% M_PI / Expression(pi) special-casing to do; powers print as pow(b,e) which
% text2code accepts. The expression *form* therefore differs from the Python
% output, but the function set, output sizes, and 0-based indexing match.

[xdg, udg, ~, ~, wdg, ~, ~, odg, ~, ~, uhg, nlg, tau, uinf, param, time] = syminit(pde);

pdemodelfun = str2func(pde.modelfile);
pdem = pdemodelfun();

nc  = pde.nc;
ncu = pde.ncu;
u = udg(1:ncu);
if nc > ncu
    q = udg(ncu+1:end);
else
    q = sym([]);
end

% Argument lists used by text2code (must match the names declared in the
% scalars/vectors header). Element (volume) functions take the field args;
% face/boundary functions additionally take uhat, n, tau.
elemArgs = "x, uq, v, w, eta, mu, t";
faceArgs = "x, uq, v, w, uhat, n, tau, eta, mu, t";
initArgs = "x, eta, mu";

blocks  = strings(0, 1);
present = strings(0, 1);

% ----- Flux (required) -----
if ~isfield(pdem, 'flux')
    error("genpdemodel: pde.flux is not defined");
end
f = pdem.flux(u, q, wdg, odg, xdg, time, param, uinf);
blocks(end+1) = emit_function("Flux", elemArgs, "f", f);
present(end+1) = "Flux";

% ----- Source (required by text2code; default to zero if absent) -----
if isfield(pdem, 'source')
    f = pdem.source(u, q, wdg, odg, xdg, time, param, uinf);
else
    f = sym(zeros(ncu, 1));
end
blocks(end+1) = emit_function("Source", elemArgs, "s", f);
present(end+1) = "Source";

% ----- Tdfunc (mass; required by text2code; default to ones if absent) -----
if isfield(pdem, 'mass')
    f = pdem.mass(u, q, wdg, odg, xdg, time, param, uinf);
else
    f = sym(ones(ncu, 1));
end
blocks(end+1) = emit_function("Tdfunc", elemArgs, "m", f);
present(end+1) = "Tdfunc";

% ----- Ubou (required) -----
if ~isfield(pdem, 'ubou')
    error("genpdemodel: pde.ubou is not defined");
end
f = pdem.ubou(u, q, wdg, odg, xdg, time, param, uinf, uhg, nlg, tau);
blocks(end+1) = emit_function("Ubou", faceArgs, "ub", f);
present(end+1) = "Ubou";

% ----- Fbou (required) -----
if ~isfield(pdem, 'fbou')
    error("genpdemodel: pde.fbou is not defined");
end
f = pdem.fbou(u, q, wdg, odg, xdg, time, param, uinf, uhg, nlg, tau);
blocks(end+1) = emit_function("Fbou", faceArgs, "fb", f);
present(end+1) = "Fbou";

% ----- FbouHdg (required for hybrid) -----
if ~isfield(pdem, 'fbouhdg')
    error("genpdemodel: pde.fbouhdg is not defined");
end
f = pdem.fbouhdg(u, q, wdg, odg, xdg, time, param, uinf, uhg, nlg, tau);
blocks(end+1) = emit_function("FbouHdg", faceArgs, "fb", f);
present(end+1) = "FbouHdg";

% ----- Fint (interface coupling; optional for single-domain models) -----
if isfield(pdem, 'fint')
    f = pdem.fint(u, q, wdg, odg, xdg, time, param, uinf, uhg, nlg, tau);
    blocks(end+1) = emit_function("Fint", faceArgs, "fi", f);
    present(end+1) = "Fint";
end

% ----- Initu (required) -----
if ~isfield(pdem, 'initu')
    error("genpdemodel: pde.initu is not defined");
end
f = pdem.initu(xdg, param, uinf);
blocks(end+1) = emit_function("Initu", initArgs, "ui", f);
present(end+1) = "Initu";

% ----- optional visualization / QoI functions -----
if isfield(pdem, 'visscalars')
    f = pdem.visscalars(u, q, wdg, odg, xdg, time, param, uinf);
    blocks(end+1) = emit_function("VisScalars", elemArgs, "s", f);
    present(end+1) = "VisScalars";
end
if isfield(pdem, 'visvectors')
    f = pdem.visvectors(u, q, wdg, odg, xdg, time, param, uinf);
    blocks(end+1) = emit_function("VisVectors", elemArgs, "s", f);
    present(end+1) = "VisVectors";
end
if isfield(pdem, 'vistensors')
    f = pdem.vistensors(u, q, wdg, odg, xdg, time, param, uinf);
    blocks(end+1) = emit_function("VisTensors", elemArgs, "s", f);
    present(end+1) = "VisTensors";
end
if isfield(pdem, 'qoivolume')
    f = pdem.qoivolume(u, q, wdg, odg, xdg, time, param, uinf);
    blocks(end+1) = emit_function("QoIvolume", elemArgs, "s", f);
    present(end+1) = "QoIvolume";
end
if isfield(pdem, 'qoiboundary')
    f = pdem.qoiboundary(u, q, wdg, odg, xdg, time, param, uinf, uhg, nlg, tau);
    blocks(end+1) = emit_function("QoIboundary", faceArgs, "fb", f);
    present(end+1) = "QoIboundary";
end

% ----- assemble -----
lines = make_header(pde);
lines(end+1) = outputs_line(present);
lines(end+1) = "";
text = strjoin(lines, newline) + newline + strjoin(blocks, newline);
if ~endsWith(text, newline)
    text = text + newline;
end

fid = fopen(char(dest_path), 'w');
if fid < 0
    error("genpdemodel: cannot open %s for writing", dest_path);
end
fwrite(fid, char(text));
fclose(fid);

end

% ---------------------------------------------------------------------------

function blk = emit_function(name, args, output_name, exprs)
% Render one  function NAME(args) ... end  block.
%
% exprs is flattened in column-major order (like gencode); output_name is the
% text2code output vector name.
exprs = sym(exprs);
exprs = exprs(:);                 % column-major flatten, mirrors numpy 'F'
n = numel(exprs);
lines = strings(0, 1);
lines(end+1) = "function " + name + "(" + args + ")";
lines(end+1) = "  output_size(" + output_name + ") = " + num2str(n) + ";";
for i = 1:n
    lines(end+1) = "  " + output_name + "[" + num2str(i-1) + "] = " + ...
                   print_expr(exprs(i)) + ";"; %#ok<AGROW>
end
lines(end+1) = "end";
blk = strjoin(lines, newline) + newline;
end

% ---------------------------------------------------------------------------

function s = print_expr(e)
% Print a single scalar symbolic expression as a text2code infix string,
% renaming frontend symbols to 0-based vector indexing.
%
% ccode emits "  t0 = <expr>;"; strip that wrapper, then rename symbols.
c = ccode(e);
c = string(c);
% ccode may emit several lines for common subexpressions (t0, t1, ...). For
% the simple model functions here it is a single "  tN = expr;" line; handle
% the general case by taking the RHS of the final assignment and inlining any
% earlier temporaries.
c = strrep(c, sprintf('\r'), "");
parts = splitlines(c);
parts = strtrim(parts);
parts = parts(strlength(parts) > 0);

% Build a map of temporary-name -> expression, inline forward.
exprStr = "";
tmpNames = strings(0,1);
tmpVals  = strings(0,1);
for k = 1:numel(parts)
    ln = parts(k);
    ln = regexprep(ln, ";\s*$", "");
    tok = regexp(ln, "^(\w+)\s*=\s*(.*)$", "tokens", "once");
    if isempty(tok)
        continue;
    end
    lhs = string(tok{1});
    rhs = string(tok{2});
    % inline previously-defined temporaries into rhs
    for j = numel(tmpNames):-1:1
        rhs = regexprep(rhs, "\<" + regexptranslate("escape", tmpNames(j)) + "\>", ...
                        "(" + tmpVals(j) + ")");
    end
    tmpNames(end+1) = lhs; %#ok<AGROW>
    tmpVals(end+1)  = rhs; %#ok<AGROW>
    exprStr = rhs;         % last assignment is the result
end
if exprStr == ""
    exprStr = "0.0";
end
s = rename_symbols(exprStr);
end

% ---------------------------------------------------------------------------

function s = rename_symbols(s)
% Rename frontend symbols (udg12, xdg1, ..., time) to text2code 0-based vectors.
% Longest prefixes first so e.g. "param" is not shadowed; the numeric index is
% decremented (1-based MATLAB symbol -> 0-based text2code vector).
map = { ...
    "udg",   "uq"; ...
    "param", "mu"; ...
    "uinf",  "eta"; ...
    "xdg",   "x"; ...
    "wdg",   "w"; ...
    "odg",   "v"; ...
    "uhg",   "uhat"; ...
    "nlg",   "n"; ...
    "tau",   "tau"};
for k = 1:size(map, 1)
    pre = map{k, 1};
    vec = map{k, 2};
    pat = "\<" + pre + "(\d+)\>";
    % Dynamic replacement: $1 is the captured digit run; emit vec[$1-1].
    rep = "${[char(""" + vec + "[""),num2str(str2double($1)-1),']']}";
    s = regexprep(s, pat, rep);
end
% scalar time -> t
s = regexprep(s, "\<time\>", "t");
end

% ---------------------------------------------------------------------------

function lines = make_header(pde)
nd     = pde.nd;
nc     = pde.nc;
ncx    = pde.ncx;
ncu    = pde.ncu;
nco    = pde.nco;
ncw    = pde.ncw;
ntau   = length(pde.tau);
nuinf  = length(pde.externalparam);
nparam = length(pde.physicsparam);
if isfield(pde, 'externalparam') && ~isempty(pde.externalparam)
    next_ = length(pde.externalparam);
else
    next_ = 1;
end

lines = strings(0, 1);
lines(end+1) = "scalars t";
lines(end+1) = "";
lines(end+1) = sprintf( ...
    "vectors x(%d), uq(%d), v(%d), w(%d), uhat(%d), uext(%d), n(%d), tau(%d), mu(%d), eta(%d)", ...
    ncx, nc, nco, ncw, ncu, next_, nd, ntau, nparam, nuinf);
lines(end+1) = "";
lines(end+1) = "jacobian uq, w, uhat";
lines(end+1) = "";
lines(end+1) = "hessian";
lines(end+1) = "";
lines(end+1) = "batch x, uq, v, w, uhat, n, uext";
lines(end+1) = "";
end

% ---------------------------------------------------------------------------

function ln = outputs_line(present)
% The 'outputs' declaration, ordered to match text2code's canonical first-six
% requirement (Flux, Source, Tdfunc, Ubou, Fbou, FbouHdg) followed by any
% optional functions actually emitted.
order = ["Flux", "Source", "Tdfunc", "Ubou", "Fbou", "FbouHdg", ...
         "Fint", "Initu", "VisScalars", "VisVectors", "VisTensors", ...
         "QoIvolume", "QoIboundary"];
outs = order(ismember(order, present));
ln = "outputs " + strjoin(outs, ", ");
end
