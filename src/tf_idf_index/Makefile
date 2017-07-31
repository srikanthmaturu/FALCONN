all:
	g++ -std=c++11 -Wall -O3 -march=native index.cpp -o tf_idf_falconn_index -I ../include -I ../../external/eigen -pthread -D USE_TDFS=true -D USE_IIDF=true -D THRESHOLD=90 -D NGRAM_LENGTH=3 -D POINT_TYPE=SparseVectorFloat
