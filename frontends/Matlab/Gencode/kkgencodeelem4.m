function strkk = kkgencodeelem4(filename, f, xdg, uinf, param, foldername)
filename = string(filename);
foldername = string(foldername);

cpufile = "cpu" + filename;
tmp = "(dstype* f, const dstype* xdg, const dstype* uinf, const dstype* param, const int modelnumber, const int ng, const int ncx, const int nce, const int npe, const int ne)\n";
str = "\tfor (int i = 0; i <ng; i++) {\n";
strkk = "void " + cpufile;
strkk = strkk + tmp + "{\n";
strkk = strkk + str;

str = string("\t\tint j = i%%npe;\n");
str = str + "\t\tint k = i/npe;\n";

fstr = string(f(:));
str = varsassign(str, "param", length(param), 0, fstr);
str = varsassign(str, "uinf", length(uinf), 0, fstr);
varname = "xdg";
for i = 1:length(xdg)
    str1 = varname + string(i);
    if any(contains(fstr, str1))
      str2 = varname + "[j+npe*" + string(i-1) + "+npe*ncx*k" + "]";
      str = str + "\t\tT " + str1 + " = " + str2 + ";\n";
    end
end

n = length(f(:));
fv = f(:);
for j = 0:(n-1)
    val = fv(j+1);
    if isequal(val, 0) || (isa(val, 'sym') && isequal(val, sym(0)))
        expr = '0.0';
    elseif isa(val, 'sym')
        expr = char(ccode(val));
        ieq = strfind(expr, '=');
        if ~isempty(ieq)
            expr = expr((ieq(1)+1):end);
        end
        expr = strrep(expr, ';', '');
        expr = strtrim(expr);
    else
        expr = num2str(val, 17);
    end
    str = str + "\t\t" + string(['f[j+npe*' num2str(j) '+npe*nce*k] = ' expr ';']) + "\n";
end
strkk = strkk + str + "\t}\n" + "}\n\n";
strkk = strrep(strkk, "T ", "dstype ");
fid = fopen(fullfile(char(foldername), [char(cpufile) '.cpp']), 'w');
fprintf(fid, char(strkk));
fclose(fid);
return;
ccode(f(:),'file','tmp.c');
tmpfile = 'tmp.c';
if exist(tmpfile, 'file') ~= 2 && exist('tmp.c.c', 'file') == 2
    tmpfile = 'tmp.c.c';
end

fid  = fopen(tmpfile,'r');
f=fread(fid,'*char')';
fclose(fid);
f = strrep(f, 't0', 'A0[0][0]');
fid  = fopen(tmpfile,'w');
fprintf(fid,'%s',f);
fclose(fid);

mystr = str;
fid = fopen(tmpfile,'r');
tline = fgetl(fid);
i=1; a1 = 0;
while ischar(tline)
    str = tline;

    i1 = strfind(str,'[');
    i2 = strfind(str,']');
    if isempty(i1)==0
        a2 = str2num(str((i1(1)+1):(i2(1)-1)));
        for j = a1:(a2-1)
            strj = ['f[j+npe*' num2str(j) '+npe*nce*k] = 0.0;'];
            mystr = mystr + "\t\t" + string(strj) + "\n";
        end
        a1 = a2+1;
    end

    str = strrep(str, '  ', '');
    str = strrep(str, 'A0[', 'f[j+npe*');
    str = strrep(str, '][0]', '+npe*nce*k]');
    if isempty(i1)==1
        str = "T " + string(str);
    end

    mystr = mystr + "\t\t" + string(str) + "\n";
    tline = fgetl(fid);
    i=i+1;
end
if a1<n
    for j = a1:(n-1)
        strj = ['f[j+npe*' num2str(j) '+npe*nce*k] = 0.0;'];
        mystr = mystr + "\t\t" + string(strj) + "\n";
    end
end
fclose(fid);
str = mystr;

strkk = strkk + str + "\t}\n" + "}\n\n";
strkk = strrep(strkk, "T ", "dstype ");
fid = fopen(fullfile(char(foldername), [char(cpufile) '.cpp']), 'w');
fprintf(fid, char(strkk));
fclose(fid);

if exist(tmpfile, 'file') == 2
    delete(tmpfile);
end
if exist('tmp.c.c', 'file') == 2
    delete('tmp.c.c');
end

end
