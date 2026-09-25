% Export the Mach-8 cylinder application through Exasim's MATLAB
% exporttext2code path.  No text2code input file is assembled manually.
example_dir = fileparts(mfilename('fullpath'));
text2code_export_directory = fullfile(example_dir, '..', '..', '..', ...
    'apps', 'shockcapturing', 'cylindermach8');
text2code_export_only = true;
run(fullfile(example_dir, 'pdeapp.m'));
