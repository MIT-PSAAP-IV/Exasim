// Included in namespace fieldsplit after Context. Same pivoted HIP BLAS LU
// and inverse as native Inverse, with per-rank reusable pointer/pivot buffers.
// Sequential assembly owns these views; the next call uses the same stream.
inline void inverse(Context& ctx,cublasHandle_t handle,double* a,double* c,int* ipiv,int n,int batch,int backend){
 EXASIM_PROFILE_SCOPE("hdg.local_lu_inverse");
#ifdef HAVE_HIP
 if(ctx.cached_inverse && backend==3){
  static_assert(sizeof(uintptr_t)==sizeof(double*));
  if(size_t(n)*batch>ctx.pivots.extent(0) || size_t(batch)>ctx.info.extent(0))throw std::runtime_error("FS inverse workspace capacity exceeded");
  auto ap=ctx.aPointers.data(),cp=ctx.cPointers.data();
  Kokkos::parallel_for("FS cached inverse pointer tables",batch,KOKKOS_LAMBDA(int k){ap[k]=reinterpret_cast<uintptr_t>(a+size_t(k)*n*n);cp[k]=reinterpret_cast<uintptr_t>(c+size_t(k)*n*n);});
  CHECK_HIPBLAS(hipblasDgetrfBatched(handle,n,reinterpret_cast<double**>(ap),n,ctx.pivots.data(),ctx.info.data(),batch));
  if(ctx.validate){std::vector<int> status(batch);TemplateCopytoHost(status.data(),ctx.info.data(),batch,backend);for(int v:status)if(v)throw std::runtime_error("3D device LU failed with info="+std::to_string(v));}
  CHECK_HIPBLAS(hipblasDgetriBatched(handle,n,reinterpret_cast<double**>(ap),n,ctx.pivots.data(),reinterpret_cast<double**>(cp),n,ctx.info.data(),batch));
  if(ctx.validate){std::vector<int> status(batch);TemplateCopytoHost(status.data(),ctx.info.data(),batch,backend);for(int v:status)if(v)throw std::runtime_error("3D device inverse failed with info="+std::to_string(v));}
  ArrayCopy(a,c,n*n*batch);
  return;
 }
#endif
 Inverse(handle,a,c,ipiv,n,batch,backend);
}
