function UDG3D = extrudesol(UDG2D, porder, nz)
[np2d,nc,ne2d] = size(UDG2D);
np1d = porder+1;
UDG3D = zeros(np2d,nc,ne2d,np1d,nz);
for i = 1:nz
    for j = 1:np1d
        UDG3D(:,:,:,j,i) = UDG2D; 
    end
end
UDG3D = permute(UDG3D,[1 4 2 3 5]);
UDG3D = reshape(UDG3D,[np2d*np1d,nc,ne2d*nz]);

