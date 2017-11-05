
#include <vector>
#include <string>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <random>
#include <chrono>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <math.h>
#include <tuple>

#include "tf_idf_falconn_idx_helper.hpp"

template<class T>
typename std::enable_if<std::is_same<T, DenseVectorFloat>::value, T>::type
get_point(VectorFloat tf_idf_vector) {
    DenseVectorFloat point;
    point.resize(tf_idf_vector.size());
    for (uint64_t i = 0; i < tf_idf_vector.size(); i++) {
        point[i] = tf_idf_vector[i];
    }
    return point;
}

template<class T>
typename std::enable_if<std::is_same<T, SparseVectorFloat>::value, T>::type
get_point(VectorFloat tf_idf_vector) {
    SparseVectorFloat point;
    int32_t tf_idf_vector_size = tf_idf_vector.size();
    for (int32_t i = 0; i < tf_idf_vector_size; i++) {
        if(tf_idf_vector[i] != 0){
            point.push_back(std::make_pair(i, tf_idf_vector[i]));
        }
    }
    return point;
}

VectorFloat compute_dmk_vector(std::string sequence, uint8_t ngram_length){
    //std::cout << "New Sequence " << std::endl;
    uint64_t dmk_vec_size = pow(4, ngram_length);
    VectorFloat dmkVector(dmk_vec_size,0);
    std::vector<std::vector<double>> v(dmk_vec_size,std::vector<double>());
    //Calculating p values
    for (uint64_t i = 0; i < sequence.size() - ngram_length + 1; i++){
        std::string ngram = sequence.substr(i, ngram_length);
        uint64_t d_num = 0;
        for (uint64_t j = 0; j < ngram_length; j++) {
            d_num += a_map[ngram[j]] * pow(4, (ngram_length - j - 1));
        }
        //std::cout << ngram << " " << d_num << std::endl;
        v[d_num].push_back(i+1);
    }
    //std::cout << dmk_vec_size << " " << dmkVector.size() << " " << v.size() << std::endl;
    double vec_sq_sum = 0.0;
    for(uint64_t i = 0; i < v.size(); i++){
        if(v[i].size() == 0){
            continue;
        }
//        for(uint64_t j = 0; j < v[i].size() ; j++){
//            std::cout << v[i][j] << "\t";
//        }
//        std::cout << std::endl;
        //calculating alpha values
        for(int64_t j = (v[i].size() - 1); j > 0 ; j--){
            v[i][j] = 1.0 / (v[i][j] - v[i][j-1]);
        }
        v[i][0] = 1.0 / v[i][0];
//        for(uint64_t j = 0; j < v[i].size() ; j++){
//            std::cout << v[i][j] << "\t";
//        }
//        std::cout << std::endl;

        //calculating beta values
        double bSum = v[i][0];
        for(uint64_t j = 1; j < v[i].size(); j++) {
            v[i][j] += v[i][j-1];
            bSum += v[i][j];
        }
//        for(uint64_t j = 0; j < v[i].size() ; j++){
//            std::cout << v[i][j] << "\t";
//        }
//        std::cout << std::endl;
        //std::cout << bSum << std::endl;
        //calculatinb shanon's entropy
        double entropy = 0.0;
        for(uint64_t j = 0; j < v[i].size(); j++){
            v[i][j] /= bSum;
            entropy += -1 * v[i][j] * log2(v[i][j]);
        }
//        for(uint64_t j = 0; j < v[i].size() ; j++){
//            std::cout << v[i][j] << "\t";
//        }
//        std::cout << std::endl;
        dmkVector[i] = entropy;
        vec_sq_sum += pow(entropy, 2);
        //std::cout << "Entropy: " << entropy <<  "  " << vec_sq_sum << std::endl;
    }
    vec_sq_sum = pow(vec_sq_sum, 0.5);
    //std::cout << vec_sq_sum << std::endl;
    for(uint64_t i = 0; i < dmkVector.size(); i++){
        dmkVector[i] /= vec_sq_sum;
        //std::cout << dmkVector[i] << "\t" ;
    }
    //std::cout << std::endl;
    return dmkVector;
}

template<class T>
void construct_dmk_dataset(std::vector<std::string> &data, std::vector<T>& dmk_dataset, uint8_t ngram_length) {
    uint64_t data_size = data.size();
    dmk_dataset.resize(data_size);
    //#pragma omp parallel for
    for(uint64_t i = 0; i < data_size; i++){
        dmk_dataset[i] = get_point<T>(compute_dmk_vector(data[i], ngram_length));
    }
}

template<class T>
T get_query_dmk_vector(std::string sequence, uint64_t ngram_length){
    return get_point<T>(compute_dmk_vector(sequence, ngram_length));
}
