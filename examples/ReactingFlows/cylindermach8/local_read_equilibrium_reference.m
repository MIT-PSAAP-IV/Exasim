function eqReference = local_read_equilibrium_reference(repoRoot)
eqReference = struct();
eqReference.available = false;
eqReference.reason = '';
eqDir = fullfile(repoRoot, 'examples', 'NavierStokes', 'equilibrium5air_cylindermach8');
appFile = fullfile(eqDir, 'datain', 'app.bin');
if exist(appFile, 'file') ~= 2
    eqReference.reason = sprintf('equilibrium app file not found: %s', appFile);
    return;
end
try
    eqApp = readappbin(appFile);
    eqMu = eqApp.physicsparam(:);
    eqReference.available = true;
    eqReference.appFile = appFile;
    eqReference.physicsparam = eqMu;
    eqReference.rhoRef = eqMu(1);
    eqReference.uRef = eqMu(2);
    eqReference.pRef = eqMu(3);
    eqReference.eRef = eqMu(4);
    eqReference.LRef = eqMu(5);
catch err
    eqReference.available = false;
    eqReference.reason = err.message;
end
end
