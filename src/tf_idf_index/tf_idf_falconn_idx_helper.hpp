//
// Created by srikanth on 7/17/17.
//

#pragma once

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <map>
#include <fstream>
#include <string>
#include <math.h>
#include <unordered_map>
#include <vector>
#include <utility>
#include <tuple>
#include <regex>
#include "xxhash.h"

#include <falconn/lsh_nn_table.h>
typedef falconn::DenseVector<float> DenseVectorFloat;
typedef falconn::SparseVector<float, int32_t> SparseVectorFloat;
typedef falconn::DenseVector<double> DenseVectorDouble;
typedef falconn::SparseVector<double> SparseVectorDouble;

typedef std::vector<float> VectorFloat;
typedef std::vector<double> VectorDouble;

using namespace std;

#define MIN3(a, b, c) ((a) < (b) ? ((a) < (c) ? (a) : (c)) : ((b) < (c) ? (b) : (c)))

std::map<char, int> Mapp = {{'A', 0}, {'C', 1}, {'G', 2}, {'T', 3}, {'N', 4}};
std::map<int, char> Mapp_r = {{0, 'A'}, {1, 'C'}, {2, 'G'}, {3, 'T'}, {4, 'N'}};

std::map<char, int> d_map = {{'A', 0},
                             {'C', 1},
                             {'G', 2},
                             {'T', 3},
                             {'N', 4}};

std::map<char, int> p_map = {{'A',0},{'R',1},{'N',2},{'D',3},{'C',4},
                             {'Q',5},{'E',6},{'G',7},{'H',8},{'I',9},
                             {'L',10},{'K',11},{'M',12},{'F',13},{'P',14},
                             {'S',15},{'T',16},{'W',17},{'Y',18},{'V',19}};

uint8_t getAlphabetMapSize(uint8_t data_type){
    switch(data_type){
        case 0:
            return d_map.size();
        case 1:
            return p_map.size();
        default:
            std::cerr << "Invalid data type." << std::endl;
            exit(1);
    }
}

uint8_t getAlphabetMapValue(uint8_t data_type, char c){
    switch(data_type){
        case 0:
            if(d_map.find(c) == d_map.end()){
                std::cerr << "Invalid alphabet." << std::endl;
                exit(1);
            }
            return d_map[c];
        case 1:
            if(p_map.find(c) == p_map.end()){
                std::cerr << "Invalid alphabet." << std::endl;
                exit(1);
            }
            return p_map[c];
        default:
            std::cerr << "Invalid data type." << std::endl;
            exit(1);
    }
}

uint8_t hamming_distance(std::string& fs, std::string& ss){
    uint8_t hm_distance = 0;

    if((fs.length() == ss.length())){

        for(uint8_t i = 0; i < fs.length(); i++){
            if(!(fs[i] == ss[i])){
                hm_distance++;
            }
        }

    }else{
        hm_distance = -1;
    }
    return hm_distance;
}

size_t uiLevenshteinDistance(const std::string &s1, const std::string &s2)
{
    const size_t m(s1.size());
    const size_t n(s2.size());

    if( m==0 ) return n;
    if( n==0 ) return m;

    size_t *costs = new size_t[n + 1];

    for( size_t k=0; k<=n; k++ ) costs[k] = k;

    size_t i = 0;
    for ( std::string::const_iterator it1 = s1.begin(); it1 != s1.end(); ++it1, ++i )
    {
        costs[0] = i+1;
        size_t corner = i;

        size_t j = 0;
        for ( std::string::const_iterator it2 = s2.begin(); it2 != s2.end(); ++it2, ++j )
        {
            size_t upper = costs[j+1];
            if( *it1 == *it2 )
            {
                costs[j+1] = corner;
            }
            else
            {
                size_t t(upper<corner?upper:corner);
                costs[j+1] = (costs[j]<t?costs[j]:t)+1;
            }

            corner = upper;
        }
    }

    size_t result = costs[n];
    delete [] costs;

    return result;
}

vector<uint64_t>& hashify_vector(vector<string>& sequences){
    vector<uint64_t> * hashes = new vector<uint64_t>();
    for(string s: sequences){
        hashes->push_back(XXH64(s.c_str(), 100, 0xcc9e2d51));
    }
    return *hashes;
}

tuple<uint64_t, uint64_t, uint64_t> get_comparison(vector<string> linear_result, vector<string> falconn_result){
    uint64_t fp = 0, fn = 0;
    vector<uint64_t> * linear_result_hashes = &(hashify_vector(linear_result));
    vector<uint64_t> * falconn_result_hashes = &(hashify_vector(falconn_result));

    for(uint64_t hash: *falconn_result_hashes){
        if(find(linear_result_hashes->begin(), linear_result_hashes->end(), hash) == linear_result_hashes->end()){
            fp++;
        }
    }
    for(uint64_t hash: *linear_result_hashes){
        if(find(falconn_result_hashes->begin(), falconn_result_hashes->end(), hash) == falconn_result_hashes->end()){
            fn++;
        }
    }
    return make_tuple(falconn_result_hashes->size(),fp, fn);
};

void getKmers(std::string fastaFileName, std::vector<std::string>& sequences, uint64_t kmerSize){
    ifstream fastaFile(fastaFileName, ifstream::in);

    regex e("^>");
    smatch m;

    while(!fastaFile.eof()){
        std::string line;
        std::getline(fastaFile, line);

        if(!regex_search(line, e) && line.size() >= kmerSize){
            for(uint64_t i = 0; i < line.size() - kmerSize + 1; i++){
                sequences.push_back(line.substr(i, kmerSize));
            }
        }
    }

    fastaFile.close();
    return;
}

