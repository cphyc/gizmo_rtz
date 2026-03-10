# CMake script to run a test and compare output against golden reference
# Variables: EXE, TEST_NAME, GOLDEN, COMPARE, PYTHON

set(OUTPUT_FILE "${CMAKE_CURRENT_BINARY_DIR}/test_${TEST_NAME}_output.txt")

# Run the test binary
execute_process(
    COMMAND ${EXE} ${TEST_NAME}
    OUTPUT_FILE ${OUTPUT_FILE}
    ERROR_VARIABLE stderr_output
    RESULT_VARIABLE rc
)

if(NOT rc EQUAL 0)
    message(FATAL_ERROR "Test binary failed with exit code ${rc}:\n${stderr_output}")
endif()

# Compare against golden reference
execute_process(
    COMMAND ${PYTHON} ${COMPARE} ${GOLDEN} ${OUTPUT_FILE}
    OUTPUT_VARIABLE compare_output
    ERROR_VARIABLE compare_error
    RESULT_VARIABLE compare_rc
)

message("${compare_output}")
if(NOT compare_rc EQUAL 0)
    message(FATAL_ERROR "Comparison failed:\n${compare_error}")
endif()
