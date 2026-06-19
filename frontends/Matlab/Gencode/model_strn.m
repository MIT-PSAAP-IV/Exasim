function strn = model_strn(app)
% Per-model path suffix: "" for model 0 (the historical flat layout), else the
% model number. Reused for datain/dataout and per-model build dirs so distinct
% models never share a directory. Mirrors the Python frontend's config.model_strn.
if app.modelnumber == 0
    strn = "";
else
    strn = string(app.modelnumber);
end
end
