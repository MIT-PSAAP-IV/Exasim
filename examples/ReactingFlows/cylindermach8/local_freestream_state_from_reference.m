function flow = local_freestream_state_from_reference(rhoRef, velocityRef, Tinf, LRef)
% Construct a CNS5air freestream whose physical density and velocity match
% the already-computed equilibrium-air cylinder reference case.
pPhys = rhoRef*287.05*Tinf;
for iter = 1:12
    info = equilibrate(pPhys, Tinf, [velocityRef;0]);
    rhoSpecies = info.rho_species(:);
    rhoPhys = sum(rhoSpecies);
    pPhys = pPhys * rhoRef/rhoPhys;
    if abs(rhoPhys-rhoRef) <= 1e-11*max(1.0, rhoRef)
        break;
    end
end

info = equilibrate(pPhys, Tinf, [velocityRef;0]);
rhoSpecies = info.rho_species(:);
rhoPhys = sum(rhoSpecies);
[rhoEPhys,pCheck,emixPhys] = energyFromSpecies(rhoSpecies, Tinf, [velocityRef;0], 1e4);
[~,~,~,~,muPhys,kappaPhys,~,cpSpecies,cvSpecies] = ...
    transportcoefficients(Tinf, rhoSpecies, 1e4);
Y = rhoSpecies/rhoPhys;
cpMix = sum(double(cpSpecies(:)).*Y);
cvMix = sum(double(cvSpecies(:)).*Y);
gammaMix = cpMix/cvMix;
aPhys = sqrt(gammaMix*pCheck/rhoPhys);

flow = struct();
flow.rhoSpeciesPhys = rhoSpecies;
flow.rhoPhys = rhoPhys;
flow.Y = Y;
flow.pPhys = pCheck;
flow.requestedPressure = pPhys;
flow.TPhys = Tinf;
flow.velocityPhys = velocityRef;
flow.aPhys = aPhys;
flow.Mach = velocityRef/aPhys;
flow.rhovPhys = rhoPhys*[velocityRef;0];
flow.rhoEPhys = rhoEPhys;
flow.ePhys = emixPhys;
flow.muPhys = double(muPhys);
flow.kappaPhys = double(kappaPhys);
flow.cpMix = cpMix;
flow.cvMix = cvMix;
flow.gammaMix = gammaMix;
flow.Re = rhoPhys*velocityRef*LRef/flow.muPhys;
flow.pressureClosureRelativeError = abs(pCheck-pPhys)/max(abs(pCheck),1);
end
