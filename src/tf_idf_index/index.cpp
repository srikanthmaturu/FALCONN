//
// Created by Srikanth Maturu (srikanthmaturu@outlook.com)on 7/17/2017.
//
#include "tf_idf_falconn_idx.hpp"

#include <chrono>
#include <iostream>
#include <cstdint>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <typeinfo>
#include <omp.h>


using namespace std;
using namespace tf_idf_falconn_index;

using namespace std::chrono;
using timer = std::chrono::high_resolution_clock;

const string index_name = INDEX_NAME;

template<class duration_type=std::chrono::seconds>
struct my_timer{
    string phase;
    time_point<timer> start;

    my_timer() = delete;
    my_timer(string _phase) : phase(_phase) {
        std::cout << "Start phase ``" << phase << "''" << std::endl;
        start = timer::now();
    };
    ~my_timer(){
        auto stop = timer::now();
        std::cout << "End phase ``" << phase << "'' ";
        std::cout << " ("<< duration_cast<duration_type>(stop-start).count() << " ";
        std::cout << " ";
        std::cout << " elapsed)" << std::endl;
    }
};

template<uint64_t ngram_length_t, uint8_t threshold_t>
struct idx_file_trait{
    static std::string value(std::string hash_file){
        return hash_file + ".NGL_" + to_string(ngram_length_t)+ "_TH_" +to_string(threshold_t);
    }
};


void load_sequences(string sequences_file, vector<string>& sequences){
    ifstream input_file(sequences_file, ifstream::in);

    for(string sequence; getline(input_file, sequence);){
        uint64_t pos;
        if((pos=sequence.find('\n')) != string::npos){
            sequence.erase(pos);
        }
        if((pos=sequence.find('\r')) != string::npos){
            sequence.erase(pos);
        }
        sequences.push_back(sequence);
    }
}

int main(int argc, char* argv[]){
    constexpr uint64_t ngram_length = NGRAM_LENGTH;
    constexpr uint64_t threshold = THRESHOLD;
    typedef INDEX_TYPE tf_idf_falconn_index_type;
    typedef POINT_TYPE point_type;
    if ( argc < 2 ) {
        cout << "Usage: ./" << argv[0] << " sequences_file [query_file]" << endl;
        return 1;
    }

    string sequences_file = argv[1];
    string queries_file = argv[2];
    cout << "SF: " << sequences_file << " QF:" << queries_file << endl;
    string idx_file = idx_file_trait<ngram_length, threshold>::value(sequences_file);
    string queries_results_file = idx_file_trait<ngram_length, threshold>::value(queries_file) + "_search_results.txt";
    tf_idf_falconn_index_type tf_idf_falconn_i;

    {
        ifstream idx_ifs(idx_file);
        if ( !idx_ifs.good()){
            auto index_construction_begin_time = timer::now();
            vector<string> sequences;
            load_sequences(sequences_file, sequences);
            {
                //            my_timer<> t("index construction");
//                auto temp = index_type(keys, async);
                cout<< "Index construction begins"<< endl;
                auto temp = tf_idf_falconn_index_type(sequences);
                //            std::cout<<"temp.size()="<<temp.size()<<std::endl;
                tf_idf_falconn_i = std::move(temp);
            }
            tf_idf_falconn_i.store_to_file(idx_file);
            auto index_construction_end_time = timer::now();
            cout<< "Index construction completed." << endl;
            cout << "# total_time_to_construct_index_in_us :- " << duration_cast<chrono::microseconds>(index_construction_end_time-index_construction_begin_time).count() << endl;
        } else {
            cout << "Index already exists. Using the existing index." << endl;
        }

        tf_idf_falconn_i.load_from_file(idx_file);

        vector<string> queries;
        load_sequences(queries_file, queries);
        vector< vector< pair<string, uint64_t > > > query_results_vector(queries.size());
        ofstream results_file(queries_results_file);
        auto start = timer::now();

        #pragma omp parallel for
        for(uint64_t i=0; i<queries.size(); i++){
            auto res = tf_idf_falconn_i.match<point_type>(queries[i]);
            uint8_t minED = 100;
            for(size_t j=0; j < res.second.size(); ++j){
                uint64_t edit_distance = uiLevenshteinDistance(queries[i], res.second[j]);
                if(edit_distance < minED){
                    minED = edit_distance;
                    query_results_vector[i].clear();
                }
                else if(edit_distance > minED){
                    continue;
                }
                query_results_vector[i].push_back(make_pair(res.second[j], edit_distance));
            }
            cout << "processed query " << i << endl;
        }

        auto stop = timer::now();
        cout << "# time_per_search_query_in_us = " << duration_cast<chrono::microseconds>(stop-start).count()/(double)queries.size() << endl;
        cout << "# total_time_for_entire_queries_in_us = " << duration_cast<chrono::microseconds>(stop-start).count() << endl;
        cout << "saving results in the results file: " << queries_results_file << endl;

        for(uint64_t i=0; i < queries.size(); i++){
            results_file << ">" << queries[i] << endl;
            //cout << "Stored results of " << i << endl;
            for(size_t j=0; j<query_results_vector[i].size(); j++){
                results_file << "" << query_results_vector[i][j].first.c_str() << "  " << query_results_vector[i][j].second << endl;
            }
        }
        results_file.close();
    }
}


