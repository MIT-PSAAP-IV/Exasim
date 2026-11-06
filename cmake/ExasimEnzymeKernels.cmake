# Generate JVPs for built-in and frontend models. Use this only with Enzyme.
include_guard(GLOBAL)
set_property(GLOBAL PROPERTY EXASIM_ENZYME_GENERATOR
  "${CMAKE_CURRENT_LIST_DIR}/generate_enzyme_kernels.py")

function(exasim_generate_enzyme_kernels)
  cmake_parse_arguments(AD "" "OUTPUT;KERNELS;ABI;NAMESPACE" "DEPENDS" ${ARGN})
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  get_property(_generator GLOBAL PROPERTY EXASIM_ENZYME_GENERATOR)
  # Track existing kernel files. Wait for the Text2Code stamp when supplied.
  file(GLOB _kernels CONFIGURE_DEPENDS "${AD_KERNELS}/Kokkos*.cpp")
  add_custom_command(OUTPUT "${AD_OUTPUT}"
    COMMAND "${Python3_EXECUTABLE}" "${_generator}"
      --kernels "${AD_KERNELS}" --abi "${AD_ABI}"
      --namespace "${AD_NAMESPACE}" --output "${AD_OUTPUT}"
    DEPENDS "${_generator}" "${AD_ABI}" ${_kernels} ${AD_DEPENDS}
    COMMENT "Enzyme: emitting pointwise JVPs for ${AD_NAMESPACE}"
    VERBATIM)
endfunction()
