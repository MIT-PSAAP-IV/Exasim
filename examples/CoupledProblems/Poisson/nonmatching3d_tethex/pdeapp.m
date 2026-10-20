here=fileparts(mfilename('fullpath'));
repo=fileparts(fileparts(fileparts(fileparts(here))));
addpath(fullfile(repo,'tests','nonmatching'));
result=phase13_fullsolve('A','A-M1',2,2,1,1,false); %#ok<NASGU>
