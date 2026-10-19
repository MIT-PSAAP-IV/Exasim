from numpy import *

def writesol(filename,ifile,xdg, udg, vdg, wdg, app, master, dmd):

    
    ncx = xdg.shape[1];

    if isinstance(udg, list) and udg == []:
        ncu = 0;
    else:
        ncu = udg.shape[1];
    
    if isinstance(vdg, list) and vdg == []:
        nco = 0;
    else:
        nco = vdg.shape[1];
    
    if isinstance(wdg, list) and wdg == []:
        ncw = 0;
    else:
        ncw = wdg.shape[1];

    ndims = zeros((12,1));
    ndims[1-1] = size(dmd['elempart']);
    # MATLAB leaves this entry at zero when the backend constructs the face
    # partition.  In that mode facepartpts is intentionally unavailable here.
    if app['Cxxpreprocessing'] == 0:
        ndims[2-1] = sum(dmd['facepartpts']);
    ndims[3-1] = master['perm'].shape[1];
    ndims[4-1] = master['npe'];
    ndims[5-1] = master['npf'];
    ndims[6-1] = app['nc'];
    ndims[7-1] = app['ncu'];
    ndims[8-1] = app['ncq'];
    ndims[9-1] = ncw;
    ndims[10-1] = nco;
    ndims[11-1] = app['nch'];
    ndims[12-1] = ncx;

    nsize = zeros((20,1));
    nsize[1-1] = size(ndims);
    nsize[2-1] = size(xdg);
    nsize[3-1] = size(udg);
    nsize[4-1] = size(vdg);
    nsize[5-1] = size(wdg);

    if (ifile>0):
        print("Writing solution into file " + str(ifile));
        fileID = open(filename + str(ifile) + ".bin","wb");
    else:
        print("Writing solution into file");
        fileID = open(filename + ".bin","wb");

    array(size(nsize), dtype=float64).tofile(fileID);
    nsize.astype('float64').tofile(fileID);
    ndims.astype('float64').tofile(fileID);
    if nsize[1]>0:
        xdg = array(xdg).flatten(order = 'F');
        xdg.astype('float64').tofile(fileID);
    if nsize[2]>0:
        udg = array(udg).flatten(order = 'F');
        udg.astype('float64').tofile(fileID);
    if nsize[3]>0:
        vdg = array(vdg).flatten(order = 'F');
        vdg.astype('float64').tofile(fileID);
    if nsize[4]>0:
        wdg = array(wdg).flatten(order = 'F');
        wdg.astype('float64').tofile(fileID);

    fileID.close();
