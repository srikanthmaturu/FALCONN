include_directories("${CMAKE_CURRENT_BINARY_DIR}/../include")

file(GLOB libFiles RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
#message("Inside lib: ${CMAKE_CURRENT_SOURCE_DIR}")

set(min_hash_idx_SRCS ${libFiles})

#[[add_library(min_hash_idx ${min_hash_idx_SRCS})

install(TARGETS min_hash_idx RUNTIME DESTINATION bin
                            LIBRARY DESTINATION lib
                            ARCHIVE DESTINATION lib)

set_target_properties(min_hash_idx PROPERTIES
                        VERSION "1.0.0"
                        SOVERSION "1")]]
