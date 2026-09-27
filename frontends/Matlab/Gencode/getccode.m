function mystr = getccode(f, varstr)

mystr = string("");
n = length(f(:));

if ~isa(f, 'sym')
    mystr = getccodeelementwise(f, varstr);
    return;
end

% fv = f(:);
% for j = 0:(n-1)
%     val = fv(j+1);
%     if isequal(val, 0) || (isa(val, 'sym') && isequal(val, sym(0)))
%         expr = '0.0';
%     elseif isa(val, 'sym')
%         expr = char(ccode(val));
%         ieq = strfind(expr, '=');
%         if ~isempty(ieq)
%             expr = expr((ieq(1)+1):end);
%         end
%         expr = strrep(expr, ';', '');
%         expr = strtrim(expr);
%     else
%         expr = num2str(val, 17);
%     end
%     mystr = mystr + "\t\t" + string([varstr num2str(j) '*ng+i] = ' expr ';']) + "\n";
% end
% return;

tmpfile = [tempname '.c'];
cleanup = onCleanup(@() deleteifexists(tmpfile));
ccode(f(:),'file',tmpfile);

fid = fopen(tmpfile,'r');
if fid < 0
    mystr = getccodeelementwise(f, varstr);
    return;
end
f=fread(fid,'*char')';
fclose(fid);
f = strrep(f, 't0', 'A0[0][0]');    
fid  = fopen(tmpfile,'w');
fprintf(fid,'%s',f);
fclose(fid);

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
            strj = [varstr num2str(j) '*ng+i] = 0.0;'];
            mystr = mystr + "\t\t" + string(strj) + "\n";
        end
        a1 = a2+1;              
    end

    str = strrep(str, '  ', '');
    str = strrep(str, 'A0[', varstr);
    str = strrep(str, '][0]', '*ng+i]');                          
    if isempty(i1)==1
        str = "T " + string(str);
    end

    mystr = mystr + "\t\t" + string(str) + "\n";
    tline = fgetl(fid);        
    i=i+1;   
end
if a1<n
    for j = a1:(n-1)                
        strj = [varstr num2str(j) '*ng+i] = 0.0;'];
        mystr = mystr + "\t\t" + string(strj) + "\n";
    end
end
fclose(fid);

deleteifexists(tmpfile);
clear cleanup;

end

function mystr = getccodeelementwise(f, varstr)
mystr = string("");
fv = f(:);
for j = 0:(numel(fv)-1)
    val = fv(j+1);
    if isequal(val, 0) || (isa(val, 'sym') && isequal(val, sym(0)))
        expr = '0.0';
    elseif isa(val, 'sym')
        expr = char(ccode(val));
        ieq = strfind(expr, '=');
        if ~isempty(ieq)
            expr = expr((ieq(1)+1):end);
        end
        expr = strtrim(strrep(expr, ';', ''));
    else
        expr = num2str(val, 17);
    end
    assignment = [varstr num2str(j) '*ng+i] = ' expr ';'];
    mystr = mystr + "\t\t" + string(assignment) + "\n";
end
end

function deleteifexists(filename)
if exist(filename, 'file') == 2
    delete(filename);
end
end
