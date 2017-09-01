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
#include "tf_idf_falconn_idx_helper.hpp"



using namespace std;
using namespace tf_idf_falconn_index;

using namespace std::chrono;
using timer = std::chrono::high_resolution_clock;

#ifdef CUSTOM_BOOST_ENABLED
const string index_name = INDEX_NAME;
#else
const string index_name = "TF_IDF_FALCONN_IDX";
#endif

string pt_name = "";

#define getindextype(ngl, utd, uiidf, nhb, np, th, pt) tf_idf_falconn_idx<ngl,utd,uiidf,nhb,np,th,pt>
#define STR(x)    #x
#define STRING(x) STR(x)

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

template<uint64_t ngram_length_t, bool use_tdfs_t, bool use_iidf_t, uint64_t number_of_hash_bits, uint64_t number_of_probes, uint8_t threshold_t>
struct idx_file_trait{
    static std::string value(std::string hash_file){
        return hash_file + ".NGL_" + to_string(ngram_length_t)+ "_UTD_" + ((use_tdfs_t)?"true":"false") + "_UIIDF_" + ((use_iidf_t)?"true":"false")+"_NHB_"+to_string(number_of_hash_bits)+"_NP_" +to_string(number_of_probes)+"_TH_" +to_string(threshold_t)
                + "_PT_" + pt_name;
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
    constexpr bool use_tdfs = USE_TDFS;
    constexpr bool use_iidf = USE_IIDF;
    constexpr uint64_t number_of_hash_bits = NUMBER_OF_HASH_BITS;
    constexpr uint64_t number_of_probes = NUMBER_OF_PROBES;
    constexpr uint64_t threshold = THRESHOLD;
    pt_name = STRING(POINT_TYPE);
#ifdef CUSTOM_BOOST_ENABLED
    typedef INDEX_TYPE tf_idf_falconn_index_type;
#else
    typedef POINT_TYPE point_type;
    typedef getindextype(ngram_length, use_tdfs, use_iidf, number_of_hash_bits, number_of_probes, threshold, point_type) tf_idf_falconn_index_type;
#endif

    if ( argc < 4 ) {
        cout << "Usage: ./" << argv[0] << " sequences_file query_file filter_enabled" << endl;
        return 1;
    }

    if(use_tdfs)
    {
        cout << "Usage of tdfs is enabled." << endl;
    }
    else{
        cout << "Usage of tdfs is disabled." << endl;
    }

    string sequences_file = argv[1];
    string queries_file = argv[2];
    string filter_enabled = argv[3];
    cout << "SF: " << sequences_file << " QF:" << queries_file << endl;
    string idx_file = idx_file_trait<ngram_length, use_tdfs, use_iidf, number_of_hash_bits, number_of_probes, threshold>::value(sequences_file);
    string queries_results_file = idx_file_trait<ngram_length, use_tdfs, use_iidf, number_of_hash_bits, number_of_probes, threshold>::value(queries_file) + "_search_results.txt";
    tf_idf_falconn_index_type tf_idf_falconn_i;

    {
#ifdef CUSTOM_BOOST_ENABLED
        ifstream idx_ifs(idx_file);
        if ( !idx_ifs.good()){
            auto index_construction_begin_time = timer::now();
            vector<string> sequences;
            load_sequences(sequences_file, sequences);
            {
                cout<< "Index construction begins"<< endl;
                auto temp = tf_idf_falconn_index_type(sequences);
                tf_idf_falconn_i = std::move(temp);
            }
            tf_idf_falconn_i.store_to_file(idx_file);
            auto index_construction_end_time = timer::now();
            cout<< "Index construction completed." << endl;
            cout << "# total_time_to_construct_index_in_us :- " << duration_cast<chrono::microseconds>(index_construction_end_time-index_construction_begin_time).count() << endl;
        } else {
            cout << "Index already exists. Using the existing index." << endl;
            tf_idf_falconn_i.load_from_file(idx_file);
            std::cout << "Loaded from file. " << std::endl;
        }
#else
        vector<string> sequences;
        load_sequences(sequences_file, sequences);
        tf_idf_falconn_i = tf_idf_falconn_index_type(sequences);
#endif
        tf_idf_falconn_i.printLSHConstructionParameters();
        tf_idf_falconn_i.construct_table();
        vector<string> queries;
        load_sequences(queries_file, queries);
        ofstream results_file(queries_results_file), box_test_results_file("box_test_results");
        if(filter_enabled == "1"){
            cout << "Filter enabled. Filtering based on edit-distance. Only kmers with least edit-distance to query is outputted." << endl;
            auto start = timer::now();

            uint64_t block_size = 100000;
            uint64_t queries_size = queries.size();
            std::cout << queries_size << std::endl;
            if(queries_size < block_size){
                block_size = queries_size;
            }

            for(uint64_t l = 32; l <= 64; l += 32){
                for(uint64_t nhb = 7; nhb <= 21; nhb += 7){
                    for(uint64_t np = l; np < l * 3; np +=5){
                        tf_idf_falconn_i.updateParmeters(l, nhb, np);
                        vector< vector< pair<string, uint64_t > > > query_results_vector;
                        tf_idf_falconn_i.printLSHConstructionParameters();
                        tf_idf_falconn_i.construct_table();
                        cout << "Current LSH Parameters: " << endl;
                        tf_idf_falconn_i.printLSHConstructionParameters();
                        uint64_t extra_block = queries_size % block_size;
                        uint64_t number_of_blocks =  queries_size / block_size;

                        if(extra_block > 0) {
                            number_of_blocks++;
                        }
                        uint64_t realMatchesCount = 0, actualMatchesCount = 0;
                        for(uint64_t bi = 0; bi < number_of_blocks; bi++){
                            uint64_t block_end = (bi == (number_of_blocks-1))? queries_size : (bi + 1)*block_size;
                            query_results_vector.resize(block_size);
                            //#pragma omp parallel for
                            for(uint64_t i= bi * block_size, j = 0; i< block_end; i++, j++){
                                auto res = tf_idf_falconn_i.match(queries[i]);
                                //tf_idf_falconn_i.linear_test(queries[i], linear_test_results_file);
                                auto nnPair = tf_idf_falconn_i.count_nearest_neighbours(queries[i]);
                                if(nnPair.first <= 40){
                                    realMatchesCount += nnPair.second;
                                }
                                uint8_t minED = 100;
                                for(size_t k=0; k < res.second.size(); ++k){
                                    uint64_t edit_distance = uiLevenshteinDistance(queries[i], res.second[k]);
                                    if(edit_distance == 0){
                                        continue;
                                    }
                                    if(edit_distance < minED){
                                        minED = edit_distance;
                                        query_results_vector[j].clear();
                                    }
                                    else if(edit_distance > minED){
                                        continue;
                                    }
                                    query_results_vector[j].push_back(make_pair(res.second[k], edit_distance));
                                }
                                uint64_t actual_matches = 0;
                                if(nnPair.first == minED && nnPair.first <= 40){
                                    actual_matches = query_results_vector[j].size();
                                    actualMatchesCount += actual_matches;
                                }
                                //cout << "Processed query: " << i << " Candidates: " << res.second.size() << " Actual Matches: " << actual_matches << " Real Matches: " << nnPair.second << endl;
                            }

                            for(uint64_t i=bi * block_size, j = 0; i < block_end; i++, j++){
                                results_file << ">" << queries[i] << endl;
                                //cout << "Stored results of " << i << endl;
                                for(size_t k=0; k<query_results_vector[j].size(); k++){
                                    results_file << "" << query_results_vector[j][k].first.c_str() << "  " << query_results_vector[j][k].second << endl;
                                }
                            }
                            query_results_vector.clear();
                        }

                        auto stop = timer::now();
                        box_test_results_file << (actualMatchesCount * 1.0) /(realMatchesCount * 1.0) << "," << duration_cast<chrono::microseconds>(stop-start).count()/(double)queries.size() << "," << l << "," << nhb << "," << np << endl;
                    }
                }
            }

            /*cout << "# time_per_search_query_in_us = " << duration_cast<chrono::microseconds>(stop-start).count()/(double)queries.size() << endl;
            cout << "# total_time_for_entire_queries_in_us = " << duration_cast<chrono::microseconds>(stop-start).count() << endl;
            cout << "Saved results in the results file: " << queries_results_file << endl;
            cout << "Actual Matches: " << actualMatchesCount << " Real Matches: " << realMatchesCount << " Recall: " << (actualMatchesCount * 1.0) /(realMatchesCount * 1.0) <<endl;*/
        }else{
            cout << "Filter disabled. So, outputting all similar kmers to query based on threshold given to falconn." << endl;
            auto start = timer::now();
            //#pragma omp parallel for
            for(uint64_t i=0; i<queries.size(); i++){
                results_file << ">" << queries[i] << endl;
                auto res = tf_idf_falconn_i.match(queries[i]);
                for(size_t j=0; j < res.second.size(); ++j){
                    results_file << res.second[j].c_str()  << endl;
                }
                cout << "Processed query: " << i << " Matches:" << res.first <<endl;
            }

            auto stop = timer::now();
            cout << "# time_per_search_query_in_us = " << duration_cast<chrono::microseconds>(stop-start).count()/(double)queries.size() << endl;
            cout << "# total_time_for_entire_queries_in_us = " << duration_cast<chrono::microseconds>(stop-start).count() << endl;
            cout << "saved results in the results file: " << queries_results_file << endl;
        }
        results_file.close();
    }
}


