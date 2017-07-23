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
#include <string.h>
#include <vector>

#include <falconn/lsh_nn_table.h>
typedef falconn::DenseVector<float> DenseVectorFloat;
typedef falconn::SparseVector<float> SparseVectorFloat;
typedef falconn::DenseVector<double> DenseVectorDouble;
typedef falconn::SparseVector<double> SparseVectorDouble;

using namespace std;

#define MIN3(a, b, c) ((a) < (b) ? ((a) < (c) ? (a) : (c)) : ((b) < (c) ? (b) : (c)))

std::map<char, int> Mapp = {{'A', 0}, {'C', 1}, {'G', 2}, {'T', 3}, {'N', 4}};
std::map<int, char> Mapp_r = {{0, 'A'}, {1, 'C'}, {2, 'G'}, {3, 'T'}, {4, 'N'}};

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

