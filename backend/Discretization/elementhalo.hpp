#ifndef EXASIM_ELEMENTHALO_HPP
#define EXASIM_ELEMENTHALO_HPP

namespace exasim_meshadapt {

template <class D>
void exchangeElementField(D& disc, dstype* field, const Int* sendIndices,
                          const Int* receiveIndices, Int blockSize)
{
#ifdef HAVE_MPI
    if (disc.common.mpiProcs <= 1) return;

    GetArrayAtIndex(disc.tmp.buffsend, field, sendIndices,
                    blockSize*disc.common.nelemsend);
    Kokkos::fence();

    Int sendOffset = 0, receiveOffset = 0, requestCount = 0;
    for (Int n = 0; n < disc.common.nnbsd; ++n) {
        const Int count = disc.common.elemsendpts[n]*blockSize;
        if (count > 0) {
            MPI_Isend(&disc.tmp.buffsend[sendOffset], count, mpi_type<dstype>(),
                      disc.common.nbsd[n], 0, EXASIM_COMM_LOCAL,
                      &disc.common.requests[requestCount++]);
            sendOffset += count;
        }
    }
    for (Int n = 0; n < disc.common.nnbsd; ++n) {
        const Int count = disc.common.elemrecvpts[n]*blockSize;
        if (count > 0) {
            MPI_Irecv(&disc.tmp.buffrecv[receiveOffset], count, mpi_type<dstype>(),
                      disc.common.nbsd[n], 0, EXASIM_COMM_LOCAL,
                      &disc.common.requests[requestCount++]);
            receiveOffset += count;
        }
    }
    MPI_Waitall(requestCount, disc.common.requests, disc.common.statuses);
    PutArrayAtIndex(field, disc.tmp.buffrecv, receiveIndices,
                    blockSize*disc.common.nelemrecv);
#else
    (void)disc;
    (void)field;
    (void)sendIndices;
    (void)receiveIndices;
    (void)blockSize;
#endif
}

} // namespace exasim_meshadapt

#endif
