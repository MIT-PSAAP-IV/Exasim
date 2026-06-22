/*
================================================================================
File: setsysstruct.cpp

Description:
------------
This file contains functions for initializing and setting up system structures
used in numerical solvers, particularly for finite element or DG methods.
It includes routines for generating random fields, allocating memory for
solution vectors, and handling parallel communication (MPI, CUDA, HIP).

Functions:
----------

1. rand_normal(dstype mean, dstype stddev)
    -------------------------------------------------
    Generates a random number from a normal (Gaussian) distribution with
    specified mean and standard deviation using the Box-Muller method.
    Uses static variables to cache one of the generated values for efficiency.

2. randomfield(dstype *randvect, commonstruct &common, resstruct res,
                    meshstruct mesh, tempstruct tmp, Int backend)
    -------------------------------------------------
    Fills the provided vector with random values (normally distributed).
    Handles parallel communication for distributed memory systems (MPI).
    Synchronizes with GPU devices if applicable (CUDA/HIP).
    Performs data exchange between subdomains and updates the random field
    accordingly.

3. setsysstruct(sysstruct &sys, commonstruct &common, resstruct res,
                     meshstruct mesh, tempstruct tmp, Int backend)
    -------------------------------------------------
    Initializes the main system structure for the solver.
    Allocates memory for solution vectors, temporary arrays, and previous
    solution states depending on the temporal and spatial schemes.
    Handles allocation for GPU/CPU backends.
    Generates and normalizes a random vector for use in the solver.
    Sets up additional structures if polynomial degree (ppdegree) > 1.

Notes:
------
- The code is designed to be portable across CPU and GPU backends.
- MPI is used for parallel communication between subdomains.
- CUDA/HIP synchronization is included for GPU execution.
- Memory allocation is handled via TemplateMalloc and standard malloc.
- The code assumes the existence of several utility functions for array
  operations and device management.

================================================================================
*/
#ifndef __SETSYSSTRUCT
#define __SETSYSSTRUCT

dstype rand_normal(dstype mean, dstype stddev)
{   //Box muller method
    static dstype n2 = 0.0;
    static int n2_cached = 0;
    if (!n2_cached)
    {
        dstype x, y, r;
        do
        {
            x = 2.0*rand()/RAND_MAX - 1;
            y = 2.0*rand()/RAND_MAX - 1;
            r = x*x + y*y;
        }
        while (r == 0.0 || r > 1.0);
        {
            dstype d = sqrt(-2.0*log(r)/r);
            dstype n1 = x*d;
            n2 = y*d;
            dstype result = n1*stddev + mean;
            n2_cached = 1;
            return result;
        }
    }
    else
    {
        n2_cached = 0;
        return n2*stddev + mean;
    }
}

void randomfield(dstype *randvect, commonstruct &common, resstruct res, meshstruct mesh, tempstruct tmp, Int backend)
{
    int N = common.grid.npe*common.components.ncu*common.ne;          
    
    dstype *rvec = (dstype *) malloc((N)*sizeof(dstype));
    for (int i=0; i<N; i++) rvec[i] = rand_normal(0.0, 1.0);   
      
    //TemplateMalloc(&randvect, N, backend);   
    TemplateCopytoDevice(randvect, rvec, N, common.backend );   
            
    free(rvec);

#ifdef HAVE_MPI         
    int bsz = common.grid.npe*common.components.ncu;
    
    for (int n=0; n<common.nelemsend; n++)  {       
      ArrayCopy(&tmp.tempn[bsz*n], &randvect[bsz*common.elemsend[n]], bsz);     
    }
    
#ifdef HAVE_CUDA
    cudaDeviceSynchronize();
#endif
    
#ifdef HAVE_HIP
    hipDeviceSynchronize();
#endif
    
    /* non-blocking send */
    Int neighbor, nsend, psend = 0, request_counter = 0;
    for (int n=0; n<common.nnbsd; n++) {
        neighbor = common.nbsd[n];
        nsend = common.elemsendpts[n]*bsz;
        if (nsend>0) {
            MPI_Isend(&tmp.tempn[psend], nsend, MPI_DOUBLE, neighbor, 0,
                  EXASIM_COMM_LOCAL, &common.requests[request_counter]);
            psend += nsend;
            request_counter += 1;
        }
    }

    /* non-blocking receive */
    Int nrecv, precv = 0;
    for (int n=0; n<common.nnbsd; n++) {
        neighbor = common.nbsd[n];
        nrecv = common.elemrecvpts[n]*bsz;
        if (nrecv>0) {
            MPI_Irecv(&tmp.tempg[precv], nrecv, MPI_DOUBLE, neighbor, 0,
                  EXASIM_COMM_LOCAL, &common.requests[request_counter]);
            precv += nrecv;
            request_counter += 1;
        }
    }
    
    MPI_Waitall(request_counter, common.requests, common.statuses);    
    for (int n=0; n<common.nelemrecv; n++) {        
      ArrayCopy(&randvect[bsz*common.elemrecv[n]], &tmp.tempg[bsz*n], bsz);       
    }    
#endif    
    
    Int ncu = common.components.ncu;
    for (Int i=0; i<ncu; i++) {
        // extract the ith component of udg and store it in res.Rq
        ArrayExtract(res.Rq, randvect, common.grid.npe, ncu, common.ne, 0, common.grid.npe, i, i+1, 0, common.ne);
        
        // make it a CG field and store in res.Ru
        ArrayDG2CG(res.Ru, res.Rq, mesh.cgent2dgent, mesh.rowent2elem, common.sizes.ndofucg);
        
        // convert CG field to DG field
        GetArrayAtIndex(res.Rq, res.Ru, mesh.cgelcon, common.grid.npe*common.ne1);
        
        // insert utm into ucg
        ArrayInsert(randvect, res.Rq, common.grid.npe, ncu, common.ne, 0, common.grid.npe, i, i+1, 0, common.ne);
    }             
    
#ifdef HAVE_MPI             
    for (int n=0; n<common.nelemsend; n++)  {       
      ArrayCopy(&tmp.tempn[bsz*n], &randvect[bsz*common.elemsend[n]], bsz);     
    }
    
#ifdef HAVE_CUDA
    cudaDeviceSynchronize();
#endif
    
#ifdef HAVE_HIP
    hipDeviceSynchronize();
#endif
    
    /* non-blocking send */
    psend = 0;
    request_counter = 0;
    for (int n=0; n<common.nnbsd; n++) {
        neighbor = common.nbsd[n];
        nsend = common.elemsendpts[n]*bsz;
        if (nsend>0) {
            MPI_Isend(&tmp.tempn[psend], nsend, MPI_DOUBLE, neighbor, 0,
                  EXASIM_COMM_LOCAL, &common.requests[request_counter]);
            psend += nsend;
            request_counter += 1;
        }
    }

    /* non-blocking receive */
    precv = 0;
    for (int n=0; n<common.nnbsd; n++) {
        neighbor = common.nbsd[n];
        nrecv = common.elemrecvpts[n]*bsz;
        if (nrecv>0) {
            MPI_Irecv(&tmp.tempg[precv], nrecv, MPI_DOUBLE, neighbor, 0,
                  EXASIM_COMM_LOCAL, &common.requests[request_counter]);
            precv += nrecv;
            request_counter += 1;
        }
    }
    
    MPI_Waitall(request_counter, common.requests, common.statuses);    
    for (int n=0; n<common.nelemrecv; n++) {        
      ArrayCopy(&randvect[bsz*common.elemrecv[n]], &tmp.tempg[bsz*n], bsz);       
    }    
#endif        
}

void setsysstruct(sysstruct &sys, commonstruct &common, resstruct res, meshstruct mesh, tempstruct tmp, Int backend)
{
    Int ncu = common.components.ncu;// number of compoments of (u)    
    Int npe = common.grid.npe; // number of nodes on master element    
    Int ne = common.ne1; // number of elements in this subdomain 
    Int N = npe*ncu*ne;    
        
    Int M = common.solverparams.gmresRestart+1;    
    M = max(M, common.solverparams.RBdim);    
    
    // fix bug here
    Int ndof = (common.spatialScheme==0) ? N : common.sizes.ndofuhat;              
    TemplateMalloc(&sys.u, ndof, backend); 
    TemplateMalloc(&sys.x, ndof, backend); 
    TemplateMalloc(&sys.b, ndof, backend); 
    TemplateMalloc(&sys.r, ndof, backend); 
    //TemplateMalloc(&sys.v, ndof*M, backend);      
    
    if (common.spatialScheme==0) {
      //TemplateMalloc(&sys.v, ndof*M, backend);      
      //sys.szv = ndof * M;
        sys.v = &res.K[res.szP];
        sys.szv = 0;
    }
    else {
      sys.v = &res.K[res.szP];
      sys.szv = 0;
    }
    
    sys.backend = backend;  
    sys.szu = ndof;
    sys.szx = ndof;
    sys.szb = ndof;
    sys.szr = ndof;
    //sys.szv = ndof * M;

    ArraySetValue(sys.u, 0.0, ndof);
    ArraySetValue(sys.x, 0.0, ndof);
    ArraySetValue(sys.b, 0.0, ndof);
    ArraySetValue(sys.r, 0.0, ndof);
    ArraySetValue(sys.v, 0.0, ndof*M);
        
    if (common.components.ncs>0) {        
        TemplateMalloc(&sys.utmp, npe*common.components.nc*common.ne2, backend); 
        sys.szutmp = npe*common.components.nc*common.ne2;
        
        if (common.components.ncw>0) {
            //TemplateMalloc(&sys.w, N, backend); 
            TemplateMalloc(&sys.wtmp, npe*common.components.ncw*common.ne2, backend); 
            //TemplateMalloc(&sys.wsrc, N, backend);               
            sys.szwtmp = npe*common.components.ncw*common.ne2; 
        }                
        
        // allocate memory for the previous solutions
        if (common.timeparams.temporalScheme==1) // BDF schemes 
        {
            N = common.grid.npe*common.components.ncs*common.ne2;
            if (common.timeparams.torder==1) {
                TemplateMalloc(&sys.udgprev1, N, backend);        
                sys.szudgprev1 = N;
            }
            else if (common.timeparams.torder==2) {
                TemplateMalloc(&sys.udgprev, N, backend);      
                TemplateMalloc(&sys.udgprev1, N, backend);      
                TemplateMalloc(&sys.udgprev2, N, backend);     
                sys.szudgprev = N;
                sys.szudgprev1 = N;
                sys.szudgprev2 = N;                 
            }
            else if (common.timeparams.torder==3) {
                TemplateMalloc(&sys.udgprev, N, backend);      
                TemplateMalloc(&sys.udgprev1, N, backend);      
                TemplateMalloc(&sys.udgprev2, N, backend);    
                TemplateMalloc(&sys.udgprev3, N, backend);   
                sys.szudgprev = N;
                sys.szudgprev1 = N;
                sys.szudgprev2 = N;                 
                sys.szudgprev3 = N;                  
            }      
            if (common.timeparams.wave==1) {
                N = common.grid.npe*common.components.ncu*common.ne1;
                if (common.timeparams.torder==1) {
                    TemplateMalloc(&sys.wprev1, N, backend);   
                    sys.szwprev1 = N;
                }
                else if (common.timeparams.torder==2) {
                    TemplateMalloc(&sys.wprev, N, backend);      
                    TemplateMalloc(&sys.wprev1, N, backend);      
                    TemplateMalloc(&sys.wprev2, N, backend);      
                    sys.szwprev = N;
                    sys.szwprev1 = N;
                    sys.szwprev2 = N;                
                }
                else if (common.timeparams.torder==3) {
                    TemplateMalloc(&sys.wprev, N, backend);      
                    TemplateMalloc(&sys.wprev1, N, backend);      
                    TemplateMalloc(&sys.wprev2, N, backend);    
                    TemplateMalloc(&sys.wprev3, N, backend);    
                    sys.szwprev = N;
                    sys.szwprev1 = N;
                    sys.szwprev2 = N;                
                    sys.szwprev3 = N;                
                }                  
            }
        }    
        else // DIRK schemes
        {
            TemplateMalloc(&sys.udgprev, npe*common.components.ncs*common.ne2, backend);      
            sys.szudgprev = npe*common.components.ncs*common.ne2;
            if (common.components.ncw>0) {
                TemplateMalloc(&sys.wprev, npe*common.components.ncw*common.ne2, backend);                
                sys.szwprev = npe*common.components.ncw*common.ne2;
            }
        }        
    }    
    
    if (backend==2) { // GPU
#ifdef HAVE_CUDA        
        cudaTemplateHostAlloc(&sys.tempmem, (5*M + M*M), cudaHostAllocMapped); // zero copy
        //TemplateMalloc(&sys.tempmem, (5*M + M*M), backend);            
#endif                  
    }
    else if (backend==3) { // GPU
#ifdef HAVE_HIP        
        hipTemplateHostMalloc(&sys.tempmem, (5*M + M*M), hipHostMallocMapped); // zero copy
        //TemplateMalloc(&sys.tempmem, (5*M + M*M), backend);            
#endif                  
    }    
    else { // CPU
        sys.tempmem = (dstype *) malloc((5*M + M*M)*sizeof(dstype));
    }
    sys.ipiv = (Int *) malloc(max(common.ppdegree, M*M)*sizeof(Int));             
    
    sys.szipiv = max(common.ppdegree, M*M);
    sys.sztempmem = (5*M + M*M);
                 
    if (common.spatialScheme==0) {
      TemplateMalloc(&sys.randvect, common.grid.npe*common.components.ncu*common.ne, backend);     
      randomfield(sys.randvect, common, res, mesh, tmp, backend);
    }
    else {
      dstype *randvectu;
      TemplateMalloc(&randvectu, common.grid.npe*common.components.ncu*common.ne, backend);            
      randomfield(randvectu, common, res, mesh, tmp, backend);
      TemplateMalloc(&sys.randvect, ndof, backend);     
      GetFaceNodes(sys.randvect, randvectu, mesh.f2e, mesh.perm, common.grid.npf, ncu, npe, ncu, common.nf);
      TemplateFree(randvectu, backend);  
    }    
    
    dstype normr = PNORM(common.cublasHandle, ndof, common.couplingparams.ndofuhatinterface, sys.randvect, backend);    
    //cout<<"sys.randvect: "<<common.mpiRank<<" "<<normr<<" "<<ndof<<endl;
    ArrayMultiplyScalar(common.cublasHandle, sys.randvect, 1.0/normr, ndof, backend);              
    sys.szrandvect = ndof;

    if (common.ppdegree > 1) {
        sys.lam = (dstype *) malloc((6*common.ppdegree + 2*common.ppdegree*common.ppdegree)*sizeof(dstype));        
        TemplateMalloc(&sys.q, ndof, backend);     
        TemplateMalloc(&sys.p, ndof, backend);                       
        sys.szq = ndof;
        sys.szp = ndof;
        sys.szlam = (6*common.ppdegree + 2*common.ppdegree*common.ppdegree);
    }
}

#endif

