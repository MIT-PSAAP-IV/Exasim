function id = resolve_modelid(app)
% Resolve an auto (negative) modelid to 100 + modelnumber. An explicit
% non-negative pde.modelid is left as-is. Use the returned value (the caller
% may also store it back). Mirrors the Python frontend's config.resolve_modelid.
if app.modelid < 0
    id = 100 + app.modelnumber;
else
    id = app.modelid;
end
end
