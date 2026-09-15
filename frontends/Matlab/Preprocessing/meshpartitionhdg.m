function dmd = meshpartitionhdg(t,f,t2t,bcm,dim,elemtype,porder,coupledinterface,nproc,metis,Cxxpreprocessing,elem2cpu)

if nargin < 12
    elem2cpu = [];
end

disp('run elementpartition...');  
dmd = elementpartitionhdg(t,t2t,f,coupledinterface,nproc,metis,elem2cpu);

if Cxxpreprocessing == 0
    disp('run facepartition...');  
    dmd = facepartitionhdg(dmd,t,f,bcm,dim,elemtype,porder,nproc);
else
    for i = 1:nproc   
        fi = f(:,dmd{i}.elempart); 
        dmd{i}.bf = 0*f(:,dmd{i}.elempart);
        for j=1:length(bcm)
          ind = fi==j;
          dmd{i}.bf(ind) = bcm(j);
        end      
    end    
end


