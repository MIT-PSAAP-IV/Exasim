function checkconformalinterface(app)
% Preserve matching inputs; reject features absent from this source release.
if iscell(app)
    for k = 1:numel(app), checkconformalinterface(app{k}); end
    return;
end
if (isfield(app,'interfacetype') && any(app.interfacetype(:) ~= 0)) || ...
        (isfield(app,'interactiontable') && ~isempty(app.interactiontable)) || ...
        (isfield(app,'interactions') && ~isempty(app.interactions)) || ...
        (isfield(app,'interactionfile') && ~isempty(app.interactionfile))
    error('Nonmatching interfaces and generalized interactions are not available in this source release.');
end
end
