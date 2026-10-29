# Preparation and execution are deliberately separate native programs. This
# keeps the ordinary application free of preprocessing/backend definitions.
string(RANDOM LENGTH 12 ALPHABET 0123456789abcdef nonce)
set(input "${ROOT}/input")
set(output "${ROOT}/output-${nonce}")
execute_process(COMMAND "${MPIEXEC}" "${NP_FLAG}" "${RANKS}" "${PREPARE}" "${input}" --prepare-only
  RESULT_VARIABLE prepared OUTPUT_VARIABLE preparation ERROR_VARIABLE preparation_error TIMEOUT 60)
if(NOT prepared EQUAL 0)
  message(FATAL_ERROR "Public fixture preparation failed: ${prepared}\n${preparation}\n${preparation_error}")
endif()
execute_process(COMMAND "${MPIEXEC}" "${NP_FLAG}" "${RANKS}" "${APP}" --hdgldg "${input}" "${output}"
  RESULT_VARIABLE solved OUTPUT_VARIABLE solution ERROR_VARIABLE solution_error TIMEOUT 60)
file(WRITE "${ROOT}/public-${nonce}.log" "${solution}\n${solution_error}")
if(NOT solved EQUAL 0 OR NOT solution MATCHES "HDGLDG Newton 1")
  message(FATAL_ERROR "Public coupled solve failed: ${solved}\n${solution}\n${solution_error}")
endif()
message(STATUS "Public coupled launch passed at np=${RANKS}: ${output}")
