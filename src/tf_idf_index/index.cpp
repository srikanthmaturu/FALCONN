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

#define getindextype(ngl, utd, uiidf, remap, nht, nhb, np, th, pt) tf_idf_falconn_idx<ngl,utd,uiidf,remap,nht,nhb,np,th,pt>
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

template<uint64_t ngram_length_t, bool use_tdfs_t, bool use_iidf_t, bool remap_t, uint64_t number_of_hash_tables_t, uint64_t number_of_hash_bits, uint64_t number_of_probes, uint8_t threshold_t>
struct idx_file_trait{
    static std::string value(std::string hash_file){
        return hash_file + ".NGL_" + to_string(ngram_length_t)+ "_UTD_" + ((use_tdfs_t)?"true":"false") + "_UIIDF_" + ((use_iidf_t)?"true":"false")+"_remap_" + ((remap_t)?"true":"false")+"_NHT_"+to_string(number_of_hash_tables_t)+"_NHB_"+to_string(number_of_hash_bits)+"_NP_" +to_string(number_of_probes)+"_TH_" +to_string(threshold_t)
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

template<class index_type>
double process_queries_parallely(index_type& tf_idf_falconn_i, vector<string>& queries, uint64_t number_of_blocks, uint64_t block_size, vector<vector<string>> linear_results){
    uint64_t realMatchesCount = 0, actualMatchesCount = 0, queries_size = queries.size();
//#pragma omp parallel for
    for(uint64_t bi = 0; bi < number_of_blocks; bi++){
        uint64_t block_end = (bi == (number_of_blocks-1))? queries_size : (bi + 1)*block_size;
        for(uint64_t i= bi * block_size, j = 0; i< block_end; i++, j++){
            vector<string>& linear_res = linear_results[i];
            auto res = tf_idf_falconn_i.match(queries[i]);
            auto cs_fp_fn_pair = get_comparison(linear_res, res.second);
            realMatchesCount += (linear_res.size());
            actualMatchesCount += (get<0>(cs_fp_fn_pair) - get<1>(cs_fp_fn_pair));
        }
    }
    double recall = (actualMatchesCount * 1.0) /(realMatchesCount * 1.0);
    return recall;
}

template<class index_type>
void process_queries_box_test(index_type& tf_idf_falconn_i, vector<string>& queries){
    ofstream box_test_results_file("box_test_results_NGL_" + to_string(NGRAM_LENGTH) + ".csv");

    uint64_t block_size = 100000;
    uint64_t queries_size = queries.size();
    std::cout << queries_size << std::endl;
    if(queries_size < block_size){
        block_size = queries_size;
    }
    uint64_t extra_block = queries_size % block_size;
    uint64_t number_of_blocks =  queries_size / block_size;
    auto start = timer::now();
    if(extra_block > 0) {
        number_of_blocks++;
    }

    vector<vector<uint64_t>> queries_linear_results(queries.size(),vector<uint64_t>(3,0));
    vector<vector<string>> linear_results;

    for(uint64_t bi = 0; bi < number_of_blocks; bi++){
        uint64_t block_end = (bi == (number_of_blocks-1))? queries_size : (bi + 1)*block_size;
        for(uint64_t i= bi * block_size, j = 0; i< block_end; i++, j++){
            linear_results.push_back(tf_idf_falconn_i.get_nearest_neighbours_by_linear_method(queries[i], 30));
        }
    }

    uint64_t polytope_vertices = NGRAM_LENGTH * 2 + 1;

    for(uint64_t l = 32; l <= 64; l += 32){
        for(uint64_t nhb = polytope_vertices; nhb <= polytope_vertices * 3; nhb += polytope_vertices){
            uint64_t np_max = 0;
            if(nhb == 7){
                np_max = l * 3;
            }
            else {
                np_max = l * 12;
            }
            uint64_t step = 1000;
            for(uint64_t np = l; np < np_max; np = np + step){
                tf_idf_falconn_i.updateParmeters(l, nhb, np);
                cout << "Current LSH Parameters: " << endl;
                tf_idf_falconn_i.printLSHConstructionParameters();
                tf_idf_falconn_i.construct_table();

                double recall = process_queries_parallely(tf_idf_falconn_i, queries, number_of_blocks, block_size, linear_results);
                auto stop = timer::now();
                box_test_results_file << recall << "," << (duration_cast<chrono::microseconds>(stop-start).count()/1000000.0)/(double)queries.size() << "," << l << "," << nhb << "," << np << endl;
                if(np_max <= (np + step) && (recall < 0.95)){
                    step *= 2;
                    np_max = np + step * 2;
                }
                if(recall >= 0.95 ||  np_max > 1000000){
                    break;
                }
            }
        }
    }
}

template<class index_type>
void process_queries_thresholds_test(index_type& tf_idf_falconn_i, vector<string>& queries){
    ofstream thresholds_test_results_file("thresholds_test_results");
    vector< vector< pair<string, uint64_t > > > query_results_vector;
    uint64_t block_size = 100000;
    uint64_t queries_size = queries.size();
    std::cout << queries_size << std::endl;
    if(queries_size < block_size){
        block_size = queries_size;
    }

    cout << "Current LSH Parameters: " << endl;
    tf_idf_falconn_i.printLSHConstructionParameters();
    tf_idf_falconn_i.construct_table();

    vector<vector<uint64_t>> queries_linear_results(queries.size(),vector<uint64_t>(3,0));
    uint64_t extra_block = queries_size % block_size;
    uint64_t number_of_blocks =  queries_size / block_size;

    if(extra_block > 0) {
        number_of_blocks++;
    }
    thresholds_test_results_file << "Query Index, tp, (ED <= 15), (15 < ED <= 20), (20 < ED <= 25), (ED > 25), ";
    for(double th = 10; th <= 150; th += 10) {
        thresholds_test_results_file << "Threshold-" << th / 100.0 << " candidates, (ED <= 15), (15 < ED <= 20), (20 < ED <= 25), (ED > 25) , fp, fn" ;
        if(th < 150){
            thresholds_test_results_file << ",";
        }
    }
    thresholds_test_results_file << endl;

    for(uint64_t bi = 0; bi < number_of_blocks; bi++){
        uint64_t block_end = (bi == (number_of_blocks-1))? queries_size : (bi + 1)*block_size;
        query_results_vector.resize(block_size);
        //#pragma omp parallel for
        for(uint64_t i= bi * block_size, j = 0; i< block_end; i++, j++){
            auto linear_res = tf_idf_falconn_i.get_nearest_neighbours_by_linear_method(queries[i], 30);
            auto edCategoryCounts = tf_idf_falconn_i.getCategoryCounts(queries[i], linear_res);
            for(auto it : edCategoryCounts){
                thresholds_test_results_file << "," << it.second;
            }
            thresholds_test_results_file << ",";
            thresholds_test_results_file << i << "," << linear_res.size() << ",";
            for(double th = 10; th <= 150; th += 10) {
                tf_idf_falconn_i.setThreshold(th/100.0);
                auto res = tf_idf_falconn_i.match(queries[i]);
                auto categoryCounts = tf_idf_falconn_i.getCategoryCounts(queries[i], res.second);
                auto cs_fp_fn_pair = get_comparison(linear_res, res.second);
                thresholds_test_results_file << std::get<0>(cs_fp_fn_pair);
                for(auto it : categoryCounts){
                    thresholds_test_results_file << "," << it.second;
                }
                thresholds_test_results_file << "," << std::get<1>(cs_fp_fn_pair) << "," << std::get<2>(cs_fp_fn_pair);
                if(th < 150){
                    thresholds_test_results_file << ",";
                }
            }
            thresholds_test_results_file << endl;
        }
        query_results_vector.clear();
    }
}

template<class index_type>
void process_queries_linear_test(index_type& tf_idf_falconn_i, vector<string>& queries){
    ofstream results_file("linear_test_results");
    for(string query: queries){
        tf_idf_falconn_i.linear_test(query, results_file);
    }
}

template<class index_type>
void process_queries_with_multiple_methods(index_type& tf_idf_falconn_i, vector<string>& queries){
    ofstream results_file("multiple_methods_results.txt");
    for(string query: queries){
        results_file << ">" << query << endl;
        tf_idf_falconn_i.get_nearest_neighbours_by_linear_method_using_multiple_methods(results_file, query, 30, 0.9);
    }

}

template<class index_type>
void process_queries(index_type& tf_idf_falconn_i, vector<string>& queries, ofstream& results_file){
    vector< vector< pair<string, uint64_t > > > query_results_vector;
    uint64_t block_size = 100000;
    uint64_t queries_size = queries.size();
    std::cout << queries_size << std::endl;
    if(queries_size < block_size){
        block_size = queries_size;
    }

    uint64_t extra_block = queries_size % block_size;
    uint64_t number_of_blocks =  queries_size / block_size;

    if(extra_block > 0) {
        number_of_blocks++;
    }
    auto start = timer::now();
    //#pragma omp parallel
    for(uint64_t bi = 0; bi < number_of_blocks; bi++){
        uint64_t block_end = (bi == (number_of_blocks-1))? queries_size : (bi + 1)*block_size;
        query_results_vector.resize(block_size);
        //#pragma omp for
        for(uint64_t i= bi * block_size, j = 0; i< block_end; i++, j++){
            auto res = tf_idf_falconn_i.match(queries[i]);

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
            cout << "Processed query: " << i << " Candidates: " << res.second.size() << endl;
        }
        //#pragma omp barrier

        for(uint64_t i=bi * block_size, j = 0; i < block_end; i++, j++){
            results_file << ">" << queries[i] << endl;
            cout << "Stored results of " << i << endl;
            for(size_t k=0; k<query_results_vector[j].size(); k++){
                results_file << "" << query_results_vector[j][k].first.c_str() << "  " << query_results_vector[j][k].second << endl;
            }
        }
        query_results_vector.clear();
    }
    auto stop = timer::now();
    cout << "# time_per_search_query_in_seconds = " << (duration_cast<chrono::microseconds>(stop-start).count()/1000000.0)/(double)queries_size << endl;
    cout << "# total_time_for_entire_queries_in_seconds= " << duration_cast<chrono::microseconds>(stop-start).count() << endl;
}

int main(int argc, char* argv[]){
    constexpr uint64_t ngram_length = NGRAM_LENGTH;
    constexpr bool use_tdfs = USE_TDFS;
    constexpr bool use_iidf = USE_IIDF;
    constexpr bool remap = REMAP;
    constexpr uint64_t number_of_hash_tables = NUMBER_OF_HASH_TABLES;
    constexpr uint64_t number_of_hash_bits = NUMBER_OF_HASH_BITS;
    constexpr uint64_t number_of_probes = NUMBER_OF_PROBES;
    constexpr uint64_t threshold = THRESHOLD;
    pt_name = STRING(POINT_TYPE);
#ifdef CUSTOM_BOOST_ENABLED
    typedef INDEX_TYPE tf_idf_falconn_index_type;
#else
    typedef POINT_TYPE point_type;
    typedef getindextype(ngram_length, use_tdfs, use_iidf, remap, number_of_hash_tables, number_of_hash_bits, number_of_probes, threshold, point_type) tf_idf_falconn_index_type;
#endif

    if ( argc < 5 ) {
        cout << "Usage: ./" << argv[0] << " sequences_file query_file type_of_query_file(fasta|kmers) filter_enabled [type_of_test]" << endl;
        return 1;
    }

    if(use_tdfs)
    {
        cout << "Usage of tdfs is enabled." << endl;
    }
    else{
        cout << "Usage of tdfs is disabled." << endl;
    }

    if(remap){
        cout << "Remap enabled." << endl;
    }

    string sequences_file = argv[1];
    string queries_file = argv[2];
    string type_of_query_file = argv[3];
    string filter_enabled = argv[4];
    uint64_t database_kmer_size = 0;
    cout << "SF: " << sequences_file << " QF:" << queries_file << endl;
    string idx_file = idx_file_trait<ngram_length, use_tdfs, use_iidf, remap, number_of_hash_tables, number_of_hash_bits, number_of_probes, threshold>::value(sequences_file);
    string queries_results_file = idx_file_trait<ngram_length, use_tdfs, use_iidf, remap, number_of_hash_tables, number_of_hash_bits, number_of_probes, threshold>::value(queries_file) + "_search_results.txt";
    tf_idf_falconn_index_type tf_idf_falconn_i;

    {
#ifdef CUSTOM_BOOST_ENABLED
        ifstream idx_ifs(idx_file);
        if ( !idx_ifs.good()){
            auto index_construction_begin_time = timer::now();
            vector<string> sequences;
            load_sequences(sequences_file, sequences);
            database_kmer_size = sequences[0].size();
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
        cout << "Kmer size: " << sequences.size() << endl;
        database_kmer_size = sequences[0].size();
        tf_idf_falconn_i = tf_idf_falconn_index_type(sequences);
#endif
        //tf_idf_falconn_i.printLSHConstructionParameters();
        tf_idf_falconn_i.construct_table();
        vector<string> queries;
        if(type_of_query_file == "fasta"){
            cout << "Input Query File Type: fasta" << endl;
            getKmers(queries_file, queries, database_kmer_size);
        }else{
            load_sequences(queries_file, queries);
        }
        ofstream results_file(queries_results_file);
        if(filter_enabled == "1"){
            cout << "Filter enabled. Filtering based on edit-distance. Only kmers with least edit-distance to query is outputted." << endl;
            if(argc == 6){
                switch(stoi(argv[5])){
                    case 0:
                        process_queries_box_test(tf_idf_falconn_i, queries);
                        break;
                    case 1:
                        process_queries_thresholds_test(tf_idf_falconn_i, queries);
                        break;
                    case 2:
                        process_queries_linear_test(tf_idf_falconn_i, queries);
                        break;
                    case 3:
                        process_queries_with_multiple_methods(tf_idf_falconn_i, queries);
                        break;
                }
            }
            else{
                process_queries(tf_idf_falconn_i, queries, results_file);
                cout << "Saved results in the results file: " << queries_results_file << endl;
            }
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


