edlib: ../../external/edlib/edlib/src/edlib.cpp
	g++ -c ../../external/edlib/edlib/src/edlib.cpp -I ../../external/edlib/edlib/include -o edlib.o
	ar -rv libedlib.a edlib.o

xxhash: xxhash.c
	g++ -c xxhash.c -o xxhash.o
	ar -rv libxxhash.a  xxhash.o

all: xxhash edlib
	g++ -std=c++11 -fpermissive -fopenmp -Wall -O3 -march=native index.cpp -o tf_idf_falconn_index -I ../include -I ../../external/eigen -I ../../external/edlib/edlib/include -l edlib -L ./ -l xxhash -pthread -D USE_TDFS=false -D \
	USE_IIDF=false -D NGRAM_LENGTH=5 -D REMAP=false -D LSH_HASH_TYPE=2 -D NUMBER_OF_HASH_TABLES=32 -D POINT_TYPE=DenseVectorFloat -D NUMBER_OF_HASH_BITS=14 -D NUMBER_OF_PROBES=2032 -D THRESHOLD=150 -D DATASET_TYPE=0
