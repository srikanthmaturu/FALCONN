xxhash: xxhash.c
	g++ -c xxhash.c -o xxhash.o
	ar -rv libxxhash.a  xxhash.o

all: xxhash
	g++ -std=c++11 -fopenmp -Wall -O3 -march=native index.cpp -o tf_idf_falconn_index -I ../include -I ../../external/eigen -I ../../external/edlib/edlib/include -L /external/edlib/build/lib/edlib_static -L ./ -l xxhash -pthread -D USE_TDFS=false -D \
	USE_IIDF=false -D NGRAM_LENGTH=5 -D REMAP=false -D LSH_HASH_TYPE=2 -D NUMBER_OF_HASH_TABLES=32 -D POINT_TYPE=DenseVectorFloat -D NUMBER_OF_HASH_BITS=14 -D NUMBER_OF_PROBES=2032 -D THRESHOLD=150
