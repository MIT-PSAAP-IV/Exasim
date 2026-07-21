function mesh = windturbine2d_transform_mesh(mesh, angle, shift)
%WINDTURBINE2D_TRANSFORM_MESH Rotate and translate all mesh coordinates.

R = [cos(angle) -sin(angle); sin(angle) cos(angle)];
mesh.p = mesh.p * R' + shift(:)';

for e = 1:size(mesh.dgnodes, 3)
    mesh.dgnodes(:,:,e) = mesh.dgnodes(:,:,e) * R' + shift(:)';
end
end
