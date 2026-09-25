function [UDG, WDG, info] = local_initial_from_equilibrium_solution(repoRoot, physicsparam)

info = struct();
info.usedEquilibriumSolution = 0;
info.reason = '';
eqDir = fullfile(repoRoot, 'examples', 'NavierStokes', 'equilibrium5air_cylindermach8');
eqUFile = fullfile(eqDir, 'dataout', 'outudg_np0.bin');
eqAppFile = fullfile(eqDir, 'datain', 'app.bin');
dbFile = fullfile(repoRoot, 'apps', 'materialdatabases', 'equilibriumAir5logdensityexasim.dat');
if exist(eqUFile, 'file') ~= 2 || exist(eqAppFile, 'file') ~= 2 || exist(dbFile, 'file') ~= 2
    info.reason = 'equilibrium UDG, app, or material database file not found';
    return;
end

eqU = getsolutions(fullfile(eqDir, 'dataout', 'outudg'));
eqW = getsolutions(fullfile(eqDir, 'dataout', 'outwdg'));
eqU = eqU(:,:,:,end);

ns = 5;
[npe, ~, ne] = size(eqU);
UDG = zeros(npe,ns+3,ne);
WDG = zeros(npe,1,ne);


eqApp = readappbin(eqAppFile);
eqMu = eqApp.physicsparam(:);
eqRhoRef = eqMu(1);
eqURef = eqMu(2);
eqERef = eqMu(4);
db = local_read_material_database(dbFile);
fields = physicalfields(eqU, db, eqRhoRef, eqERef, eqURef);
%fields = physicalfields2(eqU, eqW, eqRhoRef, eqERef, eqURef);

rhoRef = physicsparam(1);
uRef = physicsparam(2);
rhoeRef = physicsparam(3);
TRef = physicsparam(4);

rhoEq = eqU(:,1,:);
uxEq = eqU(:,2,:)./rhoEq;
uyEq = eqU(:,3,:)./rhoEq;

rhoPhys = fields.rhoField;
uxPhys = eqURef*uxEq;
uyPhys = eqURef*uyEq;
TPhys = fields.temperatureField;

% physicalfields returns equilibrium-air species fields in the database order
% [N2, O2, NO, N, O]. CNS5air conservative variables use [N, O, NO, N2, O2].
Yeq = cat(2, fields.YN, fields.YO, fields.YNO, fields.YN2, fields.YO2);
Yeq = max(Yeq, 1e-300);
Yeq = Yeq ./ sum(Yeq,2);

for ie = 1:ne
    for ip = 1:npe
        rhoiPhys = squeeze(rhoPhys(ip,1,ie)) * squeeze(Yeq(ip,:,ie)).';
        vPhys = [uxPhys(ip,1,ie); uyPhys(ip,1,ie)];
        Tnode = TPhys(ip,1,ie);
        rhoEPhys = energyFromSpecies(rhoiPhys, Tnode, vPhys, 1e4);
        UDG(ip,1:ns,ie) = rhoiPhys(:)'/rhoRef;
        UDG(ip,ns+1,ie) = sum(rhoiPhys)*vPhys(1)/(rhoRef*uRef);
        UDG(ip,ns+2,ie) = sum(rhoiPhys)*vPhys(2)/(rhoRef*uRef);
        UDG(ip,ns+3,ie) = rhoEPhys/rhoeRef;
        WDG(ip,1,ie) = Tnode/TRef;
    end
end

info.usedEquilibriumSolution = 1;
info.eqDirectory = eqDir;
info.eqUFile = eqUFile;
info.eqAppFile = eqAppFile;
info.dbFile = dbFile;
info.usedPhysicalFieldsFunction = true;
info.rhoRangePhys = [min(rhoPhys(:)), max(rhoPhys(:))];
info.eRangePhys = fields.eRange;
info.TRangePhys = [min(TPhys(:)), max(TPhys(:))];
info.YsumErrorMax = max(abs(sum(Yeq,2)-1),[],'all');
info.minY = squeeze(min(min(Yeq,[],1),[],3)).';
info.maxY = squeeze(max(max(Yeq,[],1),[],3)).';
fprintf('\nInitialized from equilibrium5air_cylindermach8 solution\n');
fprintf('  source UDG: %s\n', eqUFile);
fprintf('  physical fields: CNSequilibrium5air/physicalfields.m with %s\n', dbFile);
fprintf('  mapped rho range = [%.8g, %.8g] kg/m^3\n', info.rhoRangePhys);
fprintf('  mapped T range   = [%.8g, %.8g] K\n', info.TRangePhys);
fprintf('  mapped Y range [N O NO N2 O2] min = [%.4e %.4e %.4e %.4e %.4e]\n', info.minY);
fprintf('  mapped Y range [N O NO N2 O2] max = [%.4e %.4e %.4e %.4e %.4e]\n', info.maxY);
end
