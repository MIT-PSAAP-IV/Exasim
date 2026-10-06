add_executable(mesh_transfer_api_test api_test.cpp)
target_link_libraries(mesh_transfer_api_test PRIVATE Exasim::MeshTransfer)
add_test(NAME mesh_transfer_api COMMAND mesh_transfer_api_test)
add_executable(mesh_transfer_basis_test basis_test.cpp)
target_link_libraries(mesh_transfer_basis_test PRIVATE Exasim::MeshTransfer)
add_test(NAME mesh_transfer_basis COMMAND mesh_transfer_basis_test)
add_executable(mesh_transfer_geometry_test geometry_test.cpp)
target_link_libraries(mesh_transfer_geometry_test PRIVATE Exasim::MeshTransfer)
add_test(NAME mesh_transfer_geometry COMMAND mesh_transfer_geometry_test)
add_executable(mesh_transfer_local_walk_test local_walk_test.cpp)
target_link_libraries(mesh_transfer_local_walk_test PRIVATE Exasim::MeshTransfer)
add_test(NAME mesh_transfer_local_walk COMMAND mesh_transfer_local_walk_test)
add_executable(mesh_transfer_validation_test validation_test.cpp)
target_link_libraries(mesh_transfer_validation_test PRIVATE Exasim::MeshTransfer)
add_test(NAME mesh_transfer_validation COMMAND mesh_transfer_validation_test)
add_executable(mesh_transfer_neighbors_test neighbors_test.cpp)
target_link_libraries(mesh_transfer_neighbors_test PRIVATE Exasim::MeshTransfer)
add_test(NAME mesh_transfer_neighbors COMMAND mesh_transfer_neighbors_test)
find_package(LAPACK QUIET)
if(LAPACK_FOUND)
  add_executable(mesh_transfer_mkshape_test mkshape_test.cpp)
  target_link_libraries(mesh_transfer_mkshape_test
    PRIVATE Exasim::MeshTransfer LAPACK::LAPACK)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(mesh_transfer_mkshape_test PRIVATE -Wno-unused-variable)
  endif()
  add_test(NAME mesh_transfer_mkshape COMMAND mesh_transfer_mkshape_test)
endif()
if(MESHTRANSFER_ENABLE_MPI)
  add_test(NAME mesh_transfer_api_mpi
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:mesh_transfer_api_test>
            ${MPIEXEC_POSTFLAGS})
  add_executable(mesh_transfer_mpi_locate_test mpi_locate_test.cpp)
  target_link_libraries(mesh_transfer_mpi_locate_test PRIVATE Exasim::MeshTransfer)
  add_test(NAME mesh_transfer_mpi_locate_2
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:mesh_transfer_mpi_locate_test>
            ${MPIEXEC_POSTFLAGS})
  add_test(NAME mesh_transfer_mpi_locate_3
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 3
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:mesh_transfer_mpi_locate_test>
            ${MPIEXEC_POSTFLAGS})
  add_executable(mesh_transfer_mpi_neighbors_test mpi_neighbors_test.cpp)
  target_link_libraries(mesh_transfer_mpi_neighbors_test PRIVATE Exasim::MeshTransfer)
  add_test(NAME mesh_transfer_mpi_neighbors
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            ${MPIEXEC_PREFLAGS} $<TARGET_FILE:mesh_transfer_mpi_neighbors_test>
            ${MPIEXEC_POSTFLAGS})
endif()
