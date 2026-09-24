function flow = local_freestream_state(Minf, ReTarget, Tinf, LRef, gammaAir, RAir)
velocityPhys = Minf*sqrt(gammaAir*RAir*Tinf);
rhoPhys = 1.0e-3;
pPhys = rhoPhys*RAir*Tinf;

for iter = 1:12
    info = equilibrate(pPhys, Tinf, [velocityPhys;0]);
    rhoSpecies = info.rho_species(:);
    rhoPhys = sum(rhoSpecies);
    [~,~,~,~,muPhys,kappaPhys,~,cpSpecies,cvSpecies] = ...
        transportcoefficients(Tinf, rhoSpecies, 1e4);
    Y = rhoSpecies/rhoPhys;
    cpMix = sum(double(cpSpecies(:)).*Y);
    cvMix = sum(double(cvSpecies(:)).*Y);
    gammaMix = cpMix/cvMix;
    Rmix = pPhys/(rhoPhys*Tinf);
    aPhys = sqrt(gammaMix*Rmix*Tinf);
    velocityPhys = Minf*aPhys;
    rhoTarget = ReTarget*double(muPhys)/(velocityPhys*LRef);
    pPhys = rhoTarget*Rmix*Tinf;
    if abs(rhoTarget-rhoPhys) <= 1e-11*max(1.0, rhoPhys)
        break;
    end
end

info = equilibrate(pPhys, Tinf, [velocityPhys;0]);
rhoSpecies = info.rho_species(:);
rhoPhys = sum(rhoSpecies);
[rhoEPhys,pCheck,emixPhys] = energyFromSpecies(rhoSpecies, Tinf, [velocityPhys;0], 1e4);
[~,~,~,~,muPhys,kappaPhys,~,cpSpecies,cvSpecies] = ...
    transportcoefficients(Tinf, rhoSpecies, 1e4);
Y = rhoSpecies/rhoPhys;
cpMix = sum(double(cpSpecies(:)).*Y);
cvMix = sum(double(cvSpecies(:)).*Y);
gammaMix = cpMix/cvMix;
aPhys = velocityPhys/Minf;

flow = struct();
flow.rhoSpeciesPhys = rhoSpecies;
flow.rhoPhys = rhoPhys;
flow.Y = Y;
flow.pPhys = pCheck;
flow.requestedPressure = pPhys;
flow.TPhys = Tinf;
flow.velocityPhys = velocityPhys;
flow.aPhys = aPhys;
flow.Mach = Minf;
flow.rhovPhys = rhoPhys*[velocityPhys;0];
flow.rhoEPhys = rhoEPhys;
flow.ePhys = emixPhys;
flow.muPhys = double(muPhys);
flow.kappaPhys = double(kappaPhys);
flow.cpMix = cpMix;
flow.cvMix = cvMix;
flow.gammaMix = gammaMix;
flow.Re = rhoPhys*velocityPhys*LRef/flow.muPhys;
flow.pressureClosureRelativeError = abs(pCheck-pPhys)/max(abs(pCheck),1);
end
