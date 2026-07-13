function UDG = getmeansolution(filename,dmd,npe)
%GETMEANSOLUTION Read Exasim time-averaged solution files.
%
%   UDG = getmeansolution(filename,dmd,npe)
%
% The backend writes one file per rank as
%   <filename>_np<rank>.bin
% where the last entry is the number of accumulated samples.

filename = char(filename);
nproc = length(dmd);

if nproc==1
    tmp = readmeanfile(filename,0);
    ne = length(dmd{1}.elempart(:));
    nc = getncomp(tmp,npe,ne,0);
    UDG = reshape(tmp(1:(npe*nc*ne))/tmp(end),npe,nc,ne);
else
    nei = zeros(1,nproc);
    for i = 1:nproc
        nei(i) = sum(dmd{i}.elempartpts(1:2));
    end
    ne = sum(nei);

    tmp = readmeanfile(filename,0);
    nc = getncomp(tmp,npe,nei(1),0);

    UDG = zeros(npe,nc,ne);
    for i = 1:nproc
        elempart = dmd{i}.elempart(1:nei(i));
        tmp = readmeanfile(filename,i-1);
        nci = getncomp(tmp,npe,nei(i),i-1);
        if nci ~= nc
            error('Rank %d has %d components, expected %d.', i-1, nci, nc);
        end
        UDG(:,:,elempart) = reshape(tmp(1:end-1),[npe nc nei(i)])/tmp(end);
    end
end

end

function tmp = readmeanfile(filename,rank)
fname = sprintf('%s_np%d.bin',filename,rank);
fileID = fopen(fname,'r');
if fileID < 0
    error('Cannot open mean solution file: %s', fname);
end
tmp = fread(fileID,'double');
fclose(fileID);
if isempty(tmp)
    error('Mean solution file is empty: %s', fname);
end
if tmp(end) == 0
    error('Mean solution file has zero accumulated samples: %s', fname);
end
end

function nc = getncomp(tmp,npe,ne,rank)
if ne == 0
    error('Rank %d has zero owned elements in getmeansolution.', rank);
end
payload = numel(tmp)-1;
denom = npe*ne;
if mod(payload,denom) ~= 0
    error('Mean solution size mismatch on rank %d: payload=%d, npe=%d, ne=%d.', ...
        rank, payload, npe, ne);
end
nc = payload/denom;
end
