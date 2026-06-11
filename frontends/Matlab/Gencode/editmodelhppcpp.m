function editmodelhppcpp(id, modelpath)

kkdir = modelpath + "/model" + num2str(id);    
if ~exist(char(kkdir), 'dir')
    mkdir(char(kkdir));
end

text = fileread(char(modelpath + "/model.hpp")); 
fid = fopen(modelpath + "/model" + num2str(id) + "/model.hpp", 'w');  
t = string(text);          % ensure scalar string
t = strrep(t, "exasim_model_1", "exasim_model_" + num2str(id));
t = replace(t, newline, sprintf('\n'));  % normalize newlines
fwrite(fid, char(t), 'char');
fclose(fid);

text = fileread(char(modelpath + "/model.cpp")); 
fid = fopen(modelpath + "/model" + num2str(id) + "/model.cpp", 'w');  
t = string(text);          % ensure scalar string
t = strrep(t, "exasim_model_1", "exasim_model_" + num2str(id));
t = replace(t, newline, sprintf('\n'));  % normalize newlines
fwrite(fid, char(t), 'char');
fclose(fid);



