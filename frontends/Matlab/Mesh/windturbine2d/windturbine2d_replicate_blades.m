function [blades, loops, info] = windturbine2d_replicate_blades(baseMesh, opts)
%WINDTURBINE2D_REPLICATE_BLADES Create Nb rigid copies on the rotor circle.

blades = cell(opts.Nb, 1);
loops = cell(opts.Nb, 1);
azimuth = 2*pi*(0:opts.Nb-1)'/opts.Nb;
baseChordData = local_chord_data(baseMesh);
baseChord = baseChordData.direction;
baseChordAngle = atan2(baseChord(2), baseChord(1));

info = repmat(struct('blade', [], 'center', [], 'azimuth', [], ...
    'rotationAngle', [], 'radialDirection', [], 'tangentDirection', [], ...
    'chordDirection', [], 'radialDotChord', []), opts.Nb, 1);

for i = 1:opts.Nb
    theta = azimuth(i);
    radial = [cos(theta); sin(theta)];
    tangent = [-sin(theta); cos(theta)];
    tangentAngle = atan2(tangent(2), tangent(1));
    rotationAngle = opts.airfoilAngleOffset + tangentAngle - baseChordAngle;
    shift = opts.Rrotor * radial;

    blades{i} = windturbine2d_transform_mesh(baseMesh, rotationAngle, shift);
    allLoops = windturbine2d_boundary_loops(blades{i});
    loops{i} = allLoops(1);

    R = [cos(rotationAngle) -sin(rotationAngle); sin(rotationAngle) cos(rotationAngle)];
    chord = R * baseChord(:);
    chord = chord / norm(chord);
    info(i).blade = i;
    info(i).center = shift(:)';
    info(i).azimuth = theta;
    info(i).rotationAngle = rotationAngle;
    info(i).radialDirection = radial(:)';
    info(i).tangentDirection = tangent(:)';
    info(i).chordDirection = chord(:)';
    info(i).radialDotChord = abs(dot(chord(:), radial(:)));

    fprintf('Blade %d\n', i);
    fprintf('  center              = [%.16e, %.16e]\n', shift(1), shift(2));
    fprintf('  azimuth angle       = %.16e rad (%.8f deg)\n', theta, 180*theta/pi);
    fprintf('  rotation angle      = %.16e rad (%.8f deg)\n', rotationAngle, 180*rotationAngle/pi);
    fprintf('  radial direction    = [%.16e, %.16e]\n', radial(1), radial(2));
    fprintf('  tangent direction   = [%.16e, %.16e]\n', tangent(1), tangent(2));
    fprintf('  chord direction     = [%.16e, %.16e]\n', chord(1), chord(2));
    fprintf('  abs(dot(chord,radial)) = %.16e\n', info(i).radialDotChord);

    if info(i).radialDotChord >= 1e-12
        error('Blade %d chord is not tangent to the rotor circle.', i);
    end
end
end

function data = local_chord_data(mesh)
p = mesh.p;
if size(p, 2) ~= 2 && size(p, 1) == 2
    p = p';
end

% The airfoil chord is defined by the line from the leading-edge side
% (minimum x in the unreplicated base orientation) to the trailing-edge side
% (maximum x).  For rigidly transformed copies, the extremal points still
% identify the same chord direction after rotation.
[~, ile] = min(p(:,1));
[~, ite] = max(p(:,1));
chord = p(ite,:) - p(ile,:);
data.leadingPoint = p(ile,:);
data.trailingPoint = p(ite,:);
data.direction = chord(:) / norm(chord);
end
