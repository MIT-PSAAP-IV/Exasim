# Check both the failure and its diagnostic, including an aborted process.
execute_process(
  COMMAND "${EXECUTABLE}" "${MODE}"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE output
  ERROR_VARIABLE error
  TIMEOUT 30)
string(FIND "${output}${error}" "${EXPECTED}" found)
if(result STREQUAL "0" OR found EQUAL -1)
  message(FATAL_ERROR "Expected failure: ${EXPECTED}\nResult: ${result}\n${output}${error}")
endif()
