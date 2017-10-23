cmake_minimum_required(VERSION 3.4)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules")
include(AppendCompilerFlags)

enable_testing()

project(tf_idf_falconn_index C CXX)

append_cxx_compiler_flags("-msse4.2 -std=c++14 -Wall -DNDEBUG -fopenmp" "GCC" CMAKE_CXX_FLAGS)
append_cxx_compiler_flags("-O3 -ffast-math -funroll-loops" "GCC" CMAKE_CXX_FLAGS)


set(CMAKE_CXX_STANDARD 14)

INCLUDE_DIRECTORIES(${CMAKE_HOME_DIRECTORY}/external/eigen
        ${CMAKE_HOME_DIRECTORY}/external/edlib/edlib/include
        ${CMAKE_HOME_DIRECTORY}/src/include)

LINK_DIRECTORIES(${CMAKE_HOME_DIRECTORY}/external/edlib/build/lib)

set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)

set(BOOST_LIBRARYDIR "/util/opt/boost/1.59/gcc/5.4/lib")
set(BOOST_INCLUDEDIR "/util/opt/boost/1.59/gcc/5.4/include")

#[[
find_package(Boost COMPONENTS serialization filesystem REQUIRED)

if(Boost_FOUND)
    INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
    message("Boost Version: ${Boost_VERSION}")
    message("Boost found")
else()
    message("Boost not found")
endif()
]]

add_library(xxhash src/tf_idf_index/xxhash.c)

ADD_SUBDIRECTORY(lib)

FILE(STRINGS ${CMAKE_HOME_DIRECTORY}/exp0.config exp0.lines REGEX "^[^#].*")
FOREACH(line ${exp0.lines})
    MESSAGE("line = ${line}")
    LIST(GET line 0 index_name)
    LIST(GET line 1 index_type)
    MESSAGE("Exp 0 ${index_name}; ${index_type};")
    STRING(REGEX REPLACE "tf_idf_falconn_idx<(.*)>" "\\1" t_part ${index_type})
    STRING(REGEX REPLACE "," ";" t_params ${t_part})
    LIST(GET t_params 0 NGRAM_LENGTH)
    LIST(GET t_params 1 USE_TDFS)
    LIST(GET t_params 2 USE_IIDF)
    LIST(GET t_params 3 REMAP)
    LIST(GET t_params 4 LSH_HASH_TYPE)
    LIST(GET t_params 5 NUMBER_OF_HASH_TABLES)
    LIST(GET t_params 6 NUMBER_OF_HASH_BITS)
    LIST(GET t_params 7 NUMBER_OF_PROBES)
    LIST(GET t_params 8 THRESHOLD)
    LIST(LENGTH t_params p_size)
    LIST(GET t_params 9 POINT_TYPE)
    STRING(REGEX REPLACE "(.*)<(.*)>" "\\1" point_type ${POINT_TYPE})
    SET(exec ${index_name}_NGL_${NGRAM_LENGTH}_UTD_${USE_TDFS}_UIF_${USE_IIDF}_RMP_${REMAP}_LT_${LSH_HASH_TYPE}_NHT_${NUMBER_OF_HASH_TABLES}_NHB_${NUMBER_OF_HASH_BITS}_NP_${NUMBER_OF_PROBES}_TH_${THRESHOLD}_PT_${point_type})
    ADD_EXECUTABLE(${exec} src/tf_idf_index/index.cpp)
    TARGET_LINK_LIBRARIES(${exec} ${Boost_LIBRARIES} xxhash edlib_static)
    STRING(COMPARE EQUAL ${POINT_TYPE} DenseVectorFloat IsDenseVectorFloatType)
    SET_PROPERTY(TARGET ${exec} PROPERTY COMPILE_DEFINITIONS
            INDEX_NAME="${index_name}"
            INDEX_TYPE=${index_type}
            NGRAM_LENGTH=${NGRAM_LENGTH}
            USE_TDFS=${USE_TDFS}
            USE_IIDF=${USE_IIDF}
            REMAP=${REMAP}
            LSH_HASH_TYPE=${LSH_HASH_TYPE}
            NUMBER_OF_HASH_TABLES=${NUMBER_OF_HASH_TABLES}
            NUMBER_OF_HASH_BITS=${NUMBER_OF_HASH_BITS}
            NUMBER_OF_PROBES=${NUMBER_OF_PROBES}
            THRESHOLD=${THRESHOLD}
            POINT_TYPE=${POINT_TYPE}
            CUSTOM_BOOST_ENABLED=true
            )
    SET(nb_exec ${exec}_SV_NBV)
    ADD_EXECUTABLE(${nb_exec} src/tf_idf_index/index.cpp)
    TARGET_LINK_LIBRARIES(${nb_exec} xxhash edlib_static)
    SET_PROPERTY(TARGET ${nb_exec} PROPERTY COMPILE_DEFINITIONS
            NGRAM_LENGTH=${NGRAM_LENGTH}
            USE_TDFS=${USE_TDFS}
            USE_IIDF=${USE_IIDF}
            REMAP=${REMAP}
            LSH_HASH_TYPE=${LSH_HASH_TYPE}
            NUMBER_OF_HASH_TABLES=${NUMBER_OF_HASH_TABLES}
            NUMBER_OF_HASH_BITS=${NUMBER_OF_HASH_BITS}
            NUMBER_OF_PROBES=${NUMBER_OF_PROBES}
            THRESHOLD=${THRESHOLD}
            POINT_TYPE=${POINT_TYPE}
            )
    if(IsDenseVectorFloatType)
        GET_TARGET_PROPERTY(current_properties ${exec} COMPILE_DEFINITIONS)
        SET_PROPERTY(TARGET ${exec} PROPERTY COMPILE_DEFINITIONS ${current_properties}
                VT_DVF=true)
        GET_TARGET_PROPERTY(current_properties ${nb_exec} COMPILE_DEFINITIONS)
        SET_PROPERTY(TARGET ${nb_exec} PROPERTY COMPILE_DEFINITIONS ${current_properties}
                VT_DVF=true)
        message("Set VT_DVF compile definition")
    endif(IsDenseVectorFloatType)
    message("Point type ${POINT_TYPE}")
    message("Target ${exec}")
    message("Index Type ${index_type}")
    message("NON BOOST Target ${nb_exec}")
ENDFOREACH()
