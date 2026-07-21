function report = windturbine2d_validate(wt)
%WINDTURBINE2D_VALIDATE Basic geometry and mesh sanity checks.

report = struct();
report.Nb = numel(wt.blades);
report.bladeOuterLoopAreas = zeros(report.Nb, 1);
report.bladeOuterLoopOrientations = strings(report.Nb, 1);
for i = 1:report.Nb
    report.bladeOuterLoopAreas(i) = wt.bladeLoops{i}.area;
    report.bladeOuterLoopOrientations(i) = wt.bladeLoops{i}.orientation;
end

report.farRadius = wt.opts.Rfar;
report.rotorRadius = wt.opts.Rrotor;
report.holesInsideFarField = true;
for i = 1:report.Nb
    r = sqrt(sum(wt.bladeLoops{i}.vertices.^2, 2));
    report.holesInsideFarField = report.holesInsideFarField && ...
        all(r < wt.opts.Rfar - wt.opts.boundaryTolerance);
end

if isstruct(wt.background) && isfield(wt.background, 'p') && ~isempty(wt.background)
    report.backgroundElements = size(wt.background.t, 1);
    report.backgroundVertices = size(wt.background.p, 1);
    if isfield(wt.info, 'background')
        report.backgroundMinElementSize = wt.info.background.minElementSize;
        report.backgroundMaxElementSize = wt.info.background.maxElementSize;
        report.backgroundMinElementQuality = wt.info.background.minElementQuality;
        report.backgroundAvgElementQuality = wt.info.background.avgElementQuality;
        report.backgroundMaxElementQuality = wt.info.background.maxElementQuality;
        if isfield(wt.info.background, 'elementQualityStats')
            report.backgroundQualityP01 = wt.info.background.elementQualityStats.p01;
            report.backgroundQualityP05 = wt.info.background.elementQualityStats.p05;
            report.backgroundQualityBelow01 = wt.info.background.elementQualityStats.below01;
            report.backgroundQualityBelow02 = wt.info.background.elementQualityStats.below02;
        end
        report.backgroundElementType = wt.opts.backgroundElementType;
        report.backgroundElemtype = wt.opts.backgroundElemtype;
        if isfield(wt.info.background, 'gmsh')
            report.backgroundElementType = wt.info.background.gmsh.elementType;
            report.backgroundElemtype = wt.info.background.gmsh.elemtype;
            report.backgroundMeshSizeInterface = wt.info.background.gmsh.interfaceSize;
            report.backgroundMeshDistMin = wt.info.background.gmsh.distMin;
        end
        if isfield(wt.info.background, 'gmshElementCounts')
            report.backgroundGmshTriangles = wt.info.background.gmshElementCounts.triangles;
            report.backgroundGmshQuadrilaterals = wt.info.background.gmshElementCounts.quadrilaterals;
        end
        if isfield(wt.info.background, 'interfaceConformity')
            report.interfaceConforming = wt.info.background.interfaceConformity.conforming;
            report.interfaceExpectedEdges = wt.info.background.interfaceConformity.expectedInterfaceEdges;
            report.interfaceMatchedEdges = wt.info.background.interfaceConformity.matchedInterfaceEdges;
            report.interfaceMissingEdges = wt.info.background.interfaceConformity.missingInterfaceEdges;
            report.interfaceExtraSubedges = wt.info.background.interfaceConformity.extraInterfaceSubedges;
        end
        if isfield(wt.info.background, 'farfield')
            report.farfieldPointCount = wt.info.background.farfield.pointCount;
            report.farfieldMeanEdgeLength = wt.info.background.farfield.meanEdgeLength;
            report.farfieldMeanToRequestedRatio = wt.info.background.farfield.meanToRequestedRatio;
        end
    end
else
    report.backgroundElements = 0;
    report.backgroundVertices = 0;
    report.backgroundMinElementSize = NaN;
    report.backgroundMaxElementSize = NaN;
    report.backgroundMinElementQuality = NaN;
    report.backgroundAvgElementQuality = NaN;
    report.backgroundMaxElementQuality = NaN;
    report.backgroundElementType = "";
    report.backgroundElemtype = NaN;
    report.backgroundGmshTriangles = 0;
    report.backgroundGmshQuadrilaterals = 0;
    report.farfieldPointCount = 0;
    report.farfieldMeanEdgeLength = NaN;
    report.farfieldMeanToRequestedRatio = NaN;
end

if isfield(wt.info.merge, 'warning')
    report.mergeWarning = wt.info.merge.warning;
else
    report.mergeWarning = "";
end
end
