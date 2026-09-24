function fields = physicalfields2(UDG, WDG, rhoRef, eRef, uRef)
rho = UDG(:,1,:);
ux = UDG(:,2,:)./rho;
uy = UDG(:,3,:)./rho;
e = UDG(:,4,:)./rho - 0.5*(ux.^2 + uy.^2);
rhoPhys = rhoRef*rho;
ePhys = eRef*e;
xi = log(rhoPhys);
p = reshape(WDG(:,1,:), size(rho));
T = reshape(WDG(:,2,:), size(rho));
a = reshape(WDG(:,6,:), size(rho));
velocityPhys = sqrt(ux.^2 + uy.^2)*uRef;
Mach = velocityPhys./a;

fields = struct();
fields.xiRange = [min(xi(:)), max(xi(:))];
fields.rhoRange = exp(fields.xiRange);
fields.eRange = [min(ePhys(:)), max(ePhys(:))];
fields.pRange = [min(p(:)), max(p(:))];
fields.TRange = [min(T(:)), max(T(:))];
fields.velocityRange = [min(velocityPhys(:)), max(velocityPhys(:))];
fields.MachRange = [min(Mach(:)), max(Mach(:))];

fields.rhoField = rhoPhys;
fields.pressureField = p;
fields.temperatureField = T;
fields.velocityField = uRef*UDG(:,2:3,:)./UDG(:,1,:);
fields.energydensityField = rhoPhys.*(ePhys + 0.5*velocityPhys.^2);
fields.MachField = Mach;
fields.YN2 = reshape(WDG(:,11,:), size(rho));
fields.YO2 = reshape(WDG(:,12,:), size(rho));
fields.YNO = reshape(WDG(:,13,:), size(rho));
fields.YN = reshape(WDG(:,14,:), size(rho));
fields.YO = reshape(WDG(:,15,:), size(rho));
end
