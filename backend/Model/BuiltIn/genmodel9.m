
modelID = 9;
exasimpath = "/Users/cuongnguyen/Documents/GitHub/hexascale/Exasim";

text2code = exasimpath + "/build/text2code";
modelpath = exasimpath + "/backend/Model/BuiltIn/model" + num2str(modelID);

editmodelhppcpp(modelID, exasimpath + "/backend/Model/BuiltIn");
editlibbuiltinmodel(modelID, exasimpath + "/backend/Model/BuiltIn/libbuiltinmodel.cpp");

system(text2code + " pdeapp" + num2str(modelID) + ".txt --out-dir " + modelpath);

