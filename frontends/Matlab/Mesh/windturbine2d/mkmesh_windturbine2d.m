function wt = mkmesh_windturbine2d(opts)
%MKMESH_WINDTURBINE2D Automated 2-D wind-turbine mesh workflow.
%
%   wt = mkmesh_windturbine2d()
%   wt = mkmesh_windturbine2d(opts)
%
% The returned structure contains all intermediate meshes and geometry:
%   wt.baseMesh, wt.baseLoops, wt.blades, wt.bladeLoops,
%   wt.background, wt.mesh, wt.info.

if nargin < 1 || isempty(opts)
    opts = windturbine2d_options();
end

[baseMesh, baseParts, baseInfo] = windturbine2d_single_airfoil_mesh(opts);
baseLoops = windturbine2d_boundary_loops(baseMesh);
[blades, bladeLoops, bladeInfo] = windturbine2d_replicate_blades(baseMesh, opts);
[background, bgp, bgt, geofile, backgroundInfo, farLoop] = windturbine2d_background_mesh(bladeLoops, opts);
[mesh, mergeInfo] = windturbine2d_merge_components(background, blades, opts);

wt.opts = opts;
wt.baseMesh = baseMesh;
wt.baseParts = baseParts;
wt.baseInfo = baseInfo;
wt.baseLoops = baseLoops;
wt.blades = blades;
wt.bladeLoops = bladeLoops;
wt.farLoop = farLoop;
wt.background = background;
wt.backgroundP = bgp;
wt.backgroundT = bgt;
wt.geofile = geofile;
wt.mesh = mesh;
wt.info.merge = mergeInfo;
wt.info.blades = bladeInfo;
wt.info.background = backgroundInfo;
wt.info.validation = windturbine2d_validate(wt);

if opts.plot
    windturbine2d_plot(wt);
    hold on; 
    plot(farLoop(:,1),farLoop(:,2),'*');
    for i=1:length(bladeLoops)  
      plot(bladeLoops{i}.vertices(:,1),bladeLoops{i}.vertices(:,2),'*');
    end
end
if isfield(opts, 'plotDiagnostics') && opts.plotDiagnostics
    windturbine2d_plot_diagnostics(wt);
end

end
