function [pde, physicsparamwarmstartPosition] = replacephysicsparamwarmstart(filename, newphysicsparamwarmstart)

pde = readbin(filename);

if isempty(pde) || pde(1) < 2
    error("Invalid app binary file: %s", filename);
end

numsize = pde(1);
if numel(pde) < 1 + numsize
    error("Invalid app binary file: %s", filename);
end

nsize = pde(2:(1 + numsize));
if nsize(1) < 1 || nsize(2) < 19
    error("Invalid app binary layout or missing physicsparamwarmstart flag: %s", filename);
end

% writeapp stores physicsparamwarmstart as the 19th entry of app.flag.
physicsparamwarmstartIndex = 19;
flagStart = 2 + numsize + nsize(1);
physicsparamwarmstartPosition = flagStart + physicsparamwarmstartIndex - 1;
if numel(pde) < physicsparamwarmstartPosition
    error("Invalid app binary file: %s", filename);
end

% pde(physicsparamwarmstartPosition) = double(newphysicsparamwarmstart);
% writebin(filename, pde);

end
