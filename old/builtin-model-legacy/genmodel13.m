
modelID = 13;

cdir = pwd(); ii = strfind(cdir, "Exasim");
exasimpath = cdir(1:(ii+5));
%exasimpath = "/path/to/Exasim";

text2code = exasimpath + "/build/text2code";
modelpath = exasimpath + "/backend/Model/BuiltIn/model" + num2str(modelID);

editmodelhppcpp(modelID, exasimpath + "/backend/Model/BuiltIn");
editlibbuiltinmodel(modelID, exasimpath + "/backend/Model/BuiltIn/libbuiltinmodel.cpp");

system(text2code + " pdeapp" + num2str(modelID) + ".txt --out-dir " + modelpath);

