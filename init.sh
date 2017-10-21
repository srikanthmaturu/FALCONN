git submodule init external/edlib
git submodule update external/edlib
cd external/edlib/build/
cmake -D CMAKE_BUILD_TYPE=Release ..
make

