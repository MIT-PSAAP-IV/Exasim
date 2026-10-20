function read_msh3(filename)
 lines=strip.(readlines(filename)); start=findfirst(==("\$Nodes"),lines); count=parse(Int,lines[start+1]); points=zeros(2,count)
 for i in 1:count; record=parse.(Float64,split(lines[start+1+i])); points[:,i]=record[2:3]; end
 start=findfirst(==("\$Elements"),lines); count=parse(Int,lines[start+1]); cells=Vector{Vector{Int}}()
 for i in 1:count; record=parse.(Int,split(lines[start+1+i])); record[2]==2 && push!(cells,record[5:7]); end
 points,hcat(cells...)
end
