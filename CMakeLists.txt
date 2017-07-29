cmake_minimum_required(VERSION 3.4)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules")
include(AppendCompilerFlags)

enable_testing()

project(tf_idf_falconn_index C CXX)

append_cxx_compiler_flags("-msse4.2 -std=c++14 -Wall -DNDEBUG -fopenmp" "GCC" CMAKE_CXX_FLAGS)
append_cxx_compiler_flags("-O3 -ffast-math -funroll-loops" "GCC" CMAKE_CXX_FLAGS)


set(CMAKE_CXX_STANDARD 14)

INCLUDE_DIRECTORIES(${CMAKE_HOME_DIRECTORY}/external/eigen
        ${CMAKE_HOME_DIRECTORY}/src/include)

set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)

set(BOOST_LIBRARYDIR "/util/opt/boost/1.59/gcc/5.4/lib")
set(BOOST_INCLUDEDIR "/util/opt/boost/1.59/gcc/5.4/include")

find_package(Boost COMPONENTS serialization filesystem REQUIRED)

if(Boost_FOUND)
    INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
    message("Boost Version: ${Boost_VERSION}")
    message("Boost found")
else()
    message("Boost not found")
endif()

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
    LIST(GET t_params 1 THRESHOLD)
    LIST(LENGTH t_params p_size)
    LIST(GET t_params 2 POINT_TYPE)
    STRING(REGEX REPLACE "(.*)<(.*)>" "\\1" point_type ${POINT_TYPE})
    SET(exec ${index_name}_NGL_${NGRAM_LENGTH}_TH_${THRESHOLD}_PT_${point_type})

    ADD_EXECUTABLE(${exec} src/tf_idf_index/index.cpp)
    TARGET_LINK_LIBRARIES(${exec} ${Boost_LIBRARIES})
    SET_PROPERTY(TARGET ${exec} PROPERTY COMPILE_DEFINITIONS
            INDEX_NAME="${index_name}"
            INDEX_TYPE=${index_type}
            NGRAM_LENGTH=${NGRAM_LENGTH}
            THRESHOLD=${THRESHOLD}
            POINT_TYPE=${POINT_TYPE}
            CUSTOM_BOOST_ENABLED=true
            )
    SET(nb_exec ${exec}_SV_NONBOOSTVERSION)
    ADD_EXECUTABLE(${nb_exec} src/tf_idf_index/index.cpp)
    SET_PROPERTY(TARGET ${nb_exec} PROPERTY COMPILE_DEFINITIONS
            NGRAM_LENGTH=${NGRAM_LENGTH}
            THRESHOLD=${THRESHOLD}
            )
    message("Point type ${POINT_TYPE}")
    message("Target ${exec}")
    message("Index Type ${index_type}")
    message("NON BOOST Target ${nb_exec}")
ENDFOREACH()
