function strkk = kkgencodeface(filename, f, xdg, udg, odg, wdg, uhg, nlg, tau, uinf, param, time, foldername, genjac)

if nargin < 14
    genjac = false;
end

strkk = "";
nbc = size(f,2);
for k = 1:nbc
    str1 = kkgencodebou(filename + string(k), f(:,k), xdg, udg, odg, wdg, uhg, nlg, tau, uinf, param, time);
    strkk = strkk + str1;    
end
if genjac
    jacfilename = filename + "Jac";
    for k = 1:nbc
        str1 = kkgencodeboujac(jacfilename + string(k), f(:,k), xdg, udg, odg, wdg, uhg, nlg, tau, uinf, param, time);
        strkk = strkk + str1;
    end
end

cpufile = "Kokkos" + filename;
outputfile = cpufile;
tmp = "(dstype* f, const dstype* xdg, const dstype* udg, const dstype* odg, const dstype* wdg, const dstype* uhg, const dstype* nlg, const dstype* tau, const dstype* uinf, const dstype* param, const dstype time, const int modelnumber, const int ib, const int ng, const int nc, const int ncu, const int nd, const int ncx, const int nco, const int ncw)\n";
tmp = "void " + cpufile + tmp;
tmp = tmp + "{\n";
for k = 1:nbc
    if k == 1
        tmp = tmp + "\tif (ib == " + string(k) + ")\n";    
    else            
        tmp = tmp + "\telse if (ib == " + string(k) + ")\n";    
    end 
    tmp = tmp + "\t\t" + cpufile + string(k) + "(f, xdg, udg, odg, wdg, uhg, nlg, tau, uinf, param, time, modelnumber, ng, nc, ncu, nd, ncx, nco, ncw);\n";
end
tmp = tmp + "}\n\n";

strkk = strkk + tmp;
if genjac
    cpufile = "Kokkos" + filename + "Jac";
    tmp = "(dstype* f, dstype* f_udg, dstype* f_wdg, dstype* f_uhg, const dstype* xdg, const dstype* udg, const dstype* odg, const dstype* wdg, const dstype* uhg, const dstype* nlg, const dstype* tau, const dstype* uinf, const dstype* param, const dstype time, const int modelnumber, const int ib, const int ng, const int nc, const int ncu, const int nd, const int ncx, const int nco, const int ncw)\n";
    tmp = "void " + cpufile + tmp;
    tmp = tmp + "{\n";
    for k = 1:nbc
        if k == 1
            tmp = tmp + "\tif (ib == " + string(k) + ")\n";
        else
            tmp = tmp + "\telse if (ib == " + string(k) + ")\n";
        end
        tmp = tmp + "\t\t" + cpufile + string(k) + "(f, f_udg, f_wdg, f_uhg, xdg, udg, odg, wdg, uhg, nlg, tau, uinf, param, time, modelnumber, ng, nc, ncu, nd, ncx, nco, ncw);\n";
    end
    tmp = tmp + "}\n\n";
    strkk = strkk + tmp;
end
fid = fopen(foldername + "/" + outputfile + ".cpp", 'w');
fprintf(fid, char(strkk));
fclose(fid);

end
