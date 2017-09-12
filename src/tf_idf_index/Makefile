
xxhash: xxhash.c
    g++ -std=c++11 -Wall -O3 -march=native xxhash.c -o xxhash

all: xxhash
	g++ -std=c++11 -Wall -O3 -march=native index.cpp -o tf_idf_falconn_index -I ../include -I ../../external/eigen -l xxhash -pthread -D USE_TDFS=false -D \
	USE_IIDF=false -D NGRAM_LENGTH=3 -D POINT_TYPE=DenseVectorFloat -D NUMBER_OF_HASH_BITS=14 -D NUMBER_OF_PROBES=7000 -D THRESHOLD=50