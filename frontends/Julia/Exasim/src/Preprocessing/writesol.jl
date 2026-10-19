function writesol(filename,ifile,xdg, udg, odg, wdg, app, master, dmd)

if size(xdg,1)>0
    ncx = size(xdg,2);
else
    ncx = 0;
end
if size(udg,1)>0
    ncu = size(udg,2);
else
    ncu = 0;
end
if size(odg,1)>0
    nco = size(odg,2);
else
    nco = 0;
end
if size(wdg,1)>0
    ncw = size(wdg,2);
else
    ncw = 0;
end

ndims = zeros(12,1);
ndims[1] = length(dmd.elempart[:]);
# MATLAB leaves this entry at zero when the backend constructs the face
# partition. In that mode the frontend face partition is not authoritative.
if app.Cxxpreprocessing == 0
    ndims[2] = sum(dmd.facepartpts[:]);
end
ndims[3] = size(master.perm, 2);
ndims[4] = master.npe;
ndims[5] = master.npf;
ndims[6] = app.nc;
ndims[7] = app.ncu;
ndims[8] = app.ncq;
ndims[9] = ncw;
ndims[10] = nco;
ndims[11] = app.nch;
ndims[12] = ncx;

nsize = zeros(20,1);
nsize[1] = length(ndims[:]);
nsize[2] = length(xdg[:]);
nsize[3] = length(udg[:]);
nsize[4] = length(odg[:]);
nsize[5] = length(wdg[:]);

if (ifile>0)
    print("Writing solution into file " * string(ifile) * "\n");
    fileID = open(string(filename, string(string(ifile), ".bin")),"w");
else
    print("Writing solution into file\n");
    fileID = open(string(filename, ".bin"),"w");
end

write(fileID,Float64(length(nsize[:])));
write(fileID,Float64.(nsize[:]));

if nsize[1]>0
    write(fileID,Float64.(ndims[:]));
end
if nsize[2]>0
    write(fileID,xdg[:]);
end
if nsize[3]>0
    write(fileID,udg[:]);
end
if nsize[4]>0
    write(fileID,odg[:]);
end
if nsize[5]>0
    write(fileID,wdg[:]);
end

close(fileID);

end
