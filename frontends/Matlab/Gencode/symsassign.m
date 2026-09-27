function mystr = symsassign(mystr, f)
mystr = string(mystr);
mystr = mystr + getccode(f, 'f[');
end
