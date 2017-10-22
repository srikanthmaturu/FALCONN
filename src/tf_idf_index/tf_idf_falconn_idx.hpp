//
// Created by srikanth on 7/17/17.
//

#pragma once

#include <cstdint>
#include <vector>
#include <iostream>
#include <random>
#include <chrono>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <string>
#include <math.h>
#include <tuple>
#include "tf_idf_falconn_idx_helper.hpp"

#ifdef CUSTOM_BOOST_ENABLED
#include "eigen_boost_serialization.hpp"
#include <boost/serialization/vector.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#endif

#include <stdlib.h>
#include <falconn/lsh_nn_table.h>

namespace tf_idf_falconn_index {
    template<uint64_t ngram_length_t, bool use_tdfs_t, bool use_iidf_t, bool remap_t, uint64_t lsh_type, uint64_t number_of_hash_tables_t, uint64_t number_of_hash_bits_t, uint64_t number_of_probes_t, uint8_t threshold_t, class point_type_t=SparseVectorFloat>
    class tf_idf_falconn_idx {
    public:
        tf_idf_falconn_idx() {
            switch(lsh_type){
                case 1:
                    params.lsh_family = falconn::LSHFamily::Hyperplane;
                    break;
                case 2:
                    params.lsh_family = falconn::LSHFamily::CrossPolytope;
                    break;
            }

            params.l = number_of_hash_tables;

            if (std::is_same<point_type, DenseVectorFloat>::value) {
                params.num_rotations = 1;
                params.feature_hashing_dimension = pow(4, ngram_length);
            } else {
                params.num_rotations = 2;
                params.feature_hashing_dimension = pow(4, ngram_length) / 2;
            }
            threshold = (double_t) threshold_t / (double_t) 100;
        }

        tf_idf_falconn_idx(const tf_idf_falconn_idx &) = default;

        tf_idf_falconn_idx(tf_idf_falconn_idx &&) = default;

        tf_idf_falconn_idx &operator=(const tf_idf_falconn_idx &) = default;

        tf_idf_falconn_idx &operator=(tf_idf_falconn_idx &&) = default;

        tf_idf_falconn_idx(falconn::LSHConstructionParameters lshParams): params(lshParams) {

        }

        void initialize(std::vector<std::string> &data){
            original_data = data;
            construct_dataset(data);
            params.dimension = dataset[0].size();
            params.distance_function = falconn::DistanceFunction::EuclideanSquared;

            // we want to use all the available threads to set up
            params.num_setup_threads = 1;
            params.storage_hash_table = falconn::StorageHashTable::BitPackedFlatHashTable;
            falconn::compute_number_of_hash_functions<point_type>(number_of_hash_bits, &params);
            std::cout << "FALCONN threshold set to " << threshold << std::endl;
        }

        void printLSHConstructionParameters(){
            std::cout << "Number of Hash Tables: " << params.l << std::endl;
            std::cout << "Feature Hashing Dimension: " << params.feature_hashing_dimension << std::endl;
            std::cout << "Number of Hash bits: " << number_of_hash_bits << std::endl;
            std::cout << "Number of Hash Functions Per Hash Table: " << params.k << std::endl;
            std::cout << "Number of Probes: " << num_probes << std::endl;
            std::cout << "Last CP dimension: " << params.last_cp_dimension << std::endl;
        }

        void updateParmeters(uint64_t l, uint64_t nhb, uint64_t np){
            number_of_hash_bits = nhb;
            params.l = l;
            falconn::compute_number_of_hash_functions<point_type>(nhb, &params);
            num_probes = np;
        }

        void setThreshold(double_t threshold){
            this->threshold = threshold;
        }

        void setNGL(uint64_t ngram_length){
            this->ngram_length = ngram_length;
        }

        void setNumberOfProbes(uint64_t numberOfProbes){
            this->num_probes = numberOfProbes;
        }

        typedef point_type_t point_type;
        typedef vector<point_type> Dataset;

        Dataset dataset;
        point_type center;
        falconn::LSHConstructionParameters params;
        uint64_t number_of_hash_bits = number_of_hash_bits_t;
        uint64_t num_probes = number_of_probes_t;
        uint64_t number_of_hash_tables = number_of_hash_tables_t;
        bool use_tdfs = use_tdfs_t;
        bool use_iidf = use_iidf_t;
        bool remap = remap_t;
        double_t threshold;

        std::vector<std::string> original_data;
        uint64_t ngram_length = ngram_length_t;
        std::vector<uint64_t> tdfs;
        std::vector<double_t> tf_vec_maximums;
        std::map<char, int> a_map = {{'A', 0},
                                     {'C', 1},
                                     {'G', 2},
                                     {'T', 3},
                                     {'N', 4}};

        unique_ptr<falconn::LSHNearestNeighborTable<point_type>> table;
        unique_ptr<falconn::LSHNearestNeighborQuery<point_type>> query_object;

        void construct_table() {
            std::cout << "Dataset size: " << dataset.size() << std::endl;
            table = std::move(falconn::construct_table<point_type>(dataset, params));
            //table->reset_query_statistics();
            query_object = table->construct_query_object(num_probes);
        }

        unique_ptr<falconn::LSHNearestNeighborQuery<point_type>> createQueryObject(){
            return table->construct_query_object(num_probes);
        }


#ifdef CUSTOM_BOOST_ENABLED

        void store_to_file(std::string idx_file) {
            std::ofstream idx_file_ofs(idx_file);
            boost::archive::binary_oarchive oa(idx_file_ofs);
            oa << original_data;
            oa << tdfs;
            oa << tf_vec_maximums;
            oa << dataset;
            oa << center;
            oa << params;
            oa << threshold;
        }

        void load_from_file(std::string idx_file) {
            original_data.clear();
            dataset.clear();
            tdfs.clear();
            std::ifstream idx_file_ifs(idx_file);
            boost::archive::binary_iarchive ia(idx_file_ifs);
            ia >> original_data;
            ia >> tdfs;
            ia >> tf_vec_maximums;
            ia >> dataset;
            ia >> center;
            ia >> params;
            ia >> threshold;
        }

#endif

        std::pair<uint64_t, std::vector<std::string>> match(std::string query) {
            auto query_tf_idf_vector = getQuery_tf_idf_vector(query);
            //std::cout << std::endl;
            std::vector<int32_t> nearestNeighbours;
            query_object->find_near_neighbors(query_tf_idf_vector, threshold, &nearestNeighbours);
            std::vector<std::string> matches;
            for (auto i:nearestNeighbours) {
                matches.push_back(original_data[i]);
            }
            return std::make_pair(matches.size(), matches);
        }

        std::pair<uint64_t, std::vector<std::string>> match(unique_ptr<falconn::LSHNearestNeighborQuery<point_type>>& thread_query_object, std::string query) {
            auto query_tf_idf_vector = getQuery_tf_idf_vector(query);
            //std::cout << std::endl;
            std::vector<int32_t> nearestNeighbours;
            thread_query_object->find_near_neighbors(query_tf_idf_vector, threshold, &nearestNeighbours);
            std::vector<std::string> matches;
            for (auto i:nearestNeighbours) {
                matches.push_back(original_data[i]);
            }
            return std::make_pair(matches.size(), matches);
        }

        void getNearestNeighboursByEditDistance(unique_ptr<falconn::LSHNearestNeighborQuery<point_type>>& thread_query_object, std::string query, std::vector<int32_t> nearestNeighbours, uint64_t maxEditDistance){
            auto query_tf_idf_vector = getQuery_tf_idf_vector(query);
            thread_query_object->find_near_neighbors(query_tf_idf_vector, threshold, &nearestNeighbours);
            for (auto i:nearestNeighbours) {
                if(uiLevenshteinDistance(query, original_data[i]) <= maxEditDistance){
                    nearestNeighbours.push_back(i);
                }
            }
        }

        void getQueryBucketIds(std::string query){
            auto query_tf_idf_vector = getQuery_tf_idf_vector(query);
            return query_object->getQueryBucketIds(query_tf_idf_vector);
        }

        point_type getQuery_tf_idf_vector(std::string query) {
            uint64_t tf_vec_size = pow(4, ngram_length);
            uint64_t string_size = query.size();
            uint64_t data_size = dataset.size();
            vector<float> tf_idf_vector(tf_vec_size, 0.0);
            for (uint64_t i = 0; i < string_size - ngram_length + 1; i++) {
                std::string ngram = query.substr(i, ngram_length);
                uint64_t d_num = 0;
                for (uint64_t j = 0; j < ngram_length; j++) {
                    d_num += a_map[ngram[j]] * pow(4, (ngram_length - j - 1));
                }
                tf_idf_vector[d_num]++;
            }
            if(remap){
                for(uint64_t i = 0; i < tf_vec_maximums.size(); i += 2){
                    tf_idf_vector[i] = tf_vec_maximums[i] - tf_idf_vector[i];
                }
            }
            double vec_sq_sum = 0.0;
            for (uint64_t i = 0; i < tf_vec_size; i++) {
                if (tf_idf_vector[i] > 0) {
                    if (!use_tdfs) {
                        tf_idf_vector[i] = (1 + log10(tf_idf_vector[i]));
                    } else if (!use_iidf) {
                        if (tdfs[i] > 0) {
                            tf_idf_vector[i] *= ((log10(1 + (data_size / tdfs[i]))));
                        }
                    } else {
                        tf_idf_vector[i] *= ((log10(1 + ((double) tdfs[i] / (double) data_size))));
                    }
                    vec_sq_sum += pow(tf_idf_vector[i], 2);
                }
            }
            vec_sq_sum = pow(vec_sq_sum, 0.5);

            for (uint64_t i = 0; i < tf_vec_size; i++) {
                tf_idf_vector[i] /= vec_sq_sum;
                //std::cout << tf_idf_vector[i] << " \t";
            }
            //std::cout << std::endl;
            point_type point = get_point<point_type>(tf_idf_vector);
            if (std::is_same<point_type, DenseVectorFloat>::value) {
                subtract_center(point);
                //point.normalize();
            }
            return point;
        }

        point_type get_pure_tf_idf_vector(std::string query) {
            if(remap){
                return get_pure_tf_idf_vector_by_remap(query);
            }
            uint64_t tf_vec_size = pow(4, ngram_length);
            uint64_t string_size = query.size();
            uint64_t data_size = dataset.size();
            vector<float> tf_idf_vector(tf_vec_size, 0.0);
            for (uint64_t i = 0; i < string_size - ngram_length + 1; i++) {
                std::string ngram = query.substr(i, ngram_length);
                uint64_t d_num = 0;
                for (uint64_t j = 0; j < ngram_length; j++) {
                    d_num += a_map[ngram[j]] * pow(4, (ngram_length - j - 1));
                }
                tf_idf_vector[d_num]++;
            }
            for (uint64_t i = 0; i < tf_vec_size; i++) {
                if (tf_idf_vector[i] > 0) {
                    if (!use_tdfs) {
                        tf_idf_vector[i] = (1 + log10(tf_idf_vector[i]));
                    } else if (!use_iidf) {
                        if (tdfs[i] > 0) {
                            tf_idf_vector[i] *= ((log10(1 + (data_size / tdfs[i]))));
                        }
                    } else {
                        tf_idf_vector[i] *= ((log10(1 + ((double) tdfs[i] / (double) data_size))));
                    }
                }
            }
            point_type point = get_point<point_type>(tf_idf_vector);
            return point;
        }

        point_type get_pure_tf_idf_vector_by_remap(std::string query) {
            uint64_t tf_vec_size = pow(4, ngram_length);
            uint64_t string_size = query.size();
            vector<float> tf_idf_vector(tf_vec_size, 0.0);
            for (uint64_t i = 0; i < string_size - ngram_length + 1; i++) {
                std::string ngram = query.substr(i, ngram_length);
                uint64_t d_num = 0;
                for (uint64_t j = 0; j < ngram_length; j++) {
                    d_num += a_map[ngram[j]] * pow(4, (ngram_length - j - 1));
                }
                tf_idf_vector[d_num]++;
            }
            for (uint64_t i = 0; i < tf_vec_size; i += 2) {
                tf_idf_vector[i] = tf_vec_maximums[i] - tf_idf_vector[i];
            }
            for (uint64_t i = 0; i < tf_vec_size; i++) {
                if (tf_idf_vector[i] > 0) {
                    tf_idf_vector[i] = (1 + log10(tf_idf_vector[i]));
                }
            }
            point_type point = get_point<point_type>(tf_idf_vector);
            return point;
        }

        void construct_dataset(std::vector<std::string> &data) {
            if(remap){
                return construct_dataset_by_remap(data);
            }
            uint64_t tf_vec_size = pow(4, ngram_length);
            uint64_t data_size = data.size();
            uint64_t string_size = data[0].size();
            tdfs.resize(tf_vec_size);
            vector<VectorFloat> tf_idf_vectors(data_size, VectorFloat(tf_vec_size, 0));
            vector<double> vec_sq_sums(data_size);
            for (uint64_t i = 0; i < data_size; i++) {
                for (uint64_t j = 0; j < string_size - ngram_length + 1; j++) {
                    std::string ngram = data[i].substr(j, ngram_length);
                    uint64_t d_num = 0;
                    for (uint64_t k = 0; k < ngram_length; k++) {
                        d_num += a_map[ngram[k]] * pow(4, (ngram_length - k - 1));
                    }
                    tf_idf_vectors[i][d_num]++;
                }
                for (uint64_t j = 0; j < tf_vec_size; j++) {
                    if (tf_idf_vectors[i][j] > 0) {
                        tf_idf_vectors[i][j] = (1 + log10(tf_idf_vectors[i][j]));
                        if (use_tdfs) {
                            tdfs[j]++;
                        }
                    }
                }
            }
            for (uint64_t i = 0; i < data_size; i++) {
                double_t vec_sq_sum = 0.0;
                for (uint64_t j = 0; j < tf_vec_size; j++) {
                    if (use_tdfs) {
                        if (!use_iidf) {
                            if (tdfs[j] > 0) {
                                tf_idf_vectors[i][j] *= (log10(1 + (data_size / tdfs[j])));
                            }
                        } else {
                            tf_idf_vectors[i][j] *= (log10(1 + ((double) tdfs[j] / (double) data_size)));
                        }
                    }
                    vec_sq_sum += pow(tf_idf_vectors[i][j], 2);
                }
                vec_sq_sum = pow(vec_sq_sum, 0.5);
                for (uint64_t j = 0; j < tf_vec_size; j++) {
                    tf_idf_vectors[i][j] /= vec_sq_sum;
                    //std::cout << tf_idf_vectors[i][j] << " \t";
                }
                //std::cout << std::endl;
                dataset.push_back(get_point<point_type>(tf_idf_vectors[i]));
            }

            if (std::is_same<point_type, DenseVectorFloat>::value) {
                re_center_dataset<point_type>();
            }
        }

        void construct_dataset_by_remap(std::vector<std::string> &data) {
            uint64_t tf_vec_size = pow(4, ngram_length);
            uint64_t data_size = data.size();
            uint64_t string_size = data[0].size();
            tf_vec_maximums.resize(tf_vec_size, 0.0);
            vector<VectorFloat> tf_idf_vectors(data_size, VectorFloat(tf_vec_size, 0));
            vector<double> vec_sq_sums(data_size);
            for (uint64_t i = 0; i < data_size; i++) {
                for (uint64_t j = 0; j < string_size - ngram_length + 1; j++) {
                    std::string ngram = data[i].substr(j, ngram_length);
                    uint64_t d_num = 0;
                    for (uint64_t k = 0; k < ngram_length; k++) {
                        d_num += a_map[ngram[k]] * pow(4, (ngram_length - k - 1));
                    }
                    tf_idf_vectors[i][d_num]++;
                }
                for (uint64_t j = 0; j < tf_vec_size; j++) {
                    if (tf_idf_vectors[i][j] > tf_vec_maximums[j]) {
                        tf_vec_maximums[j] = tf_idf_vectors[i][j];
                    }
                }
            }
            for (uint64_t i = 0; i < data_size; i++) {
                for (uint64_t j = 0; j < tf_vec_size; j += 2) {
                    tf_idf_vectors[i][j] = tf_vec_maximums[j] - tf_idf_vectors[i][j];
                }
                for (uint64_t j = 0; j < tf_vec_size; j++) {
                    if (tf_idf_vectors[i][j] > 0) {
                        tf_idf_vectors[i][j] = (1 + log10(tf_idf_vectors[i][j]));
                    }
                }
                double_t vec_sq_sum = 0.0;
                for (uint64_t j = 0; j < tf_vec_size; j++) {
                    vec_sq_sum += pow(tf_idf_vectors[i][j], 2);
                }
                vec_sq_sum = pow(vec_sq_sum, 0.5);
                for (uint64_t j = 0; j < tf_vec_size; j++) {
                    tf_idf_vectors[i][j] /= vec_sq_sum;
                    //std::cout << tf_idf_vectors[i][j] << " \t";
                }
                //std::cout << std::endl;
                dataset.push_back(get_point<point_type>(tf_idf_vectors[i]));
            }
            if (std::is_same<point_type, DenseVectorFloat>::value) {
                re_center_dataset<point_type>();
            }
        }

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
//                if(tf_idf_vector[i] != 0){}
                point.push_back(std::make_pair(i, tf_idf_vector[i]));
            }
            return point;
        }

        template<class T>
        typename std::enable_if<std::is_same<T, DenseVectorFloat>::value, void>::type re_center_dataset() {
            // find the center of mass
            uint64_t data_size = dataset.size();
            center = dataset[rand() % data_size];
            for (uint64_t i = 1; i < data_size; ++i) {
                //dataset[i].normalize();
                center += dataset[i];
            }
            center /= dataset.size();

            std::cout << "Re-centering dataset points." << std::endl;
            for (auto &datapoint : dataset) {
                datapoint -= center;
            }
            std::cout << "Done." << std::endl;
        }

        template<class T>
        typename std::enable_if<std::is_same<T, SparseVectorFloat>::value, void>::type re_center_dataset() {
        }

        template<class T>
        typename std::enable_if<std::is_same<T, DenseVectorFloat>::value, void>::type subtract_center(T &point) {
            point -= center;
        }

        template<class T>
        typename std::enable_if<std::is_same<T, SparseVectorFloat>::value, void>::type subtract_center(T &point) {
        }

        template<class T>
        typename std::enable_if<std::is_same<T, DenseVectorFloat>::value, void>::type remap_dataset() {
            uint64_t tf_vec_size = pow(4, ngram_length);
            tf_vec_maximums.resize(tf_vec_size, 0.0);
            for (uint64_t i = 0; i < dataset.size(); ++i) {
                for(uint64_t j = 0; j < tf_vec_size; ++j){
                    if(dataset[i][j] > tf_vec_maximums[j]){
                        tf_vec_maximums[j] = dataset[i][j];
                    }
                }
            }
            for (uint64_t i = 0; i <dataset.size(); ++i){
                for(uint64_t j = 0; j < tf_vec_size; j += 2){
                    dataset[i][j] = tf_vec_maximums[j] - dataset[i][j];
                }
            }
        }


        template<class T>
        typename std::enable_if<std::is_same<T, SparseVectorFloat>::value, void>::type remap_dataset(){

        };

        template<class T>
        typename std::enable_if<std::is_same<T, DenseVectorFloat>::value, void>::type remap_query_tf_idf_vector(T& point) {
            for(uint64_t i = 0; i < tf_vec_maximums.size(); i += 2){
                point[i] = tf_vec_maximums[i] - point[i];
            }
        }

        template<class T>
        typename std::enable_if<std::is_same<T, SparseVectorFloat>::value, void>::type remap_query_tf_idf_vector(T& map){

        };

        void linear_test(std::string query, std::ofstream &results_file) {
            auto query_tf_idf_vector = getQuery_tf_idf_vector(query);
            auto query_pure_tf_idf_vector = get_pure_tf_idf_vector(query);
            std::vector<std::string> results(dataset.size());
            #pragma omp parallel for
            for (uint64_t i = 0; i < original_data.size(); i++) {
                auto data_item_pure_tf_idf_vector = get_pure_tf_idf_vector(original_data[i]);
                auto cosine_angle = acos(dataset[i].dot(query_tf_idf_vector))/ 3.14;
                auto cosine_distance = dataset[i].dot(query_tf_idf_vector);
                auto euclidean_distance = (dataset[i] - query_tf_idf_vector).squaredNorm();
                auto pure_cosine_distance = data_item_pure_tf_idf_vector.dot(query_pure_tf_idf_vector)/(data_item_pure_tf_idf_vector.norm() * query_pure_tf_idf_vector.norm());
                auto pure_cosine_angle = acos(pure_cosine_distance)/3.14;
                auto pure_euclidean_distance = (data_item_pure_tf_idf_vector - query_pure_tf_idf_vector).squaredNorm();

                string result =  to_string(uiLevenshteinDistance(query, original_data[i])) + "," + to_string(pure_cosine_distance) + "," + to_string(cosine_distance) + "," + to_string(pure_cosine_angle) + "," + to_string(cosine_angle) + "," + to_string(pure_euclidean_distance) + "," + to_string(euclidean_distance) + "\n";
                results[i] = result;
            }

            for (uint64_t i = 0; i < results.size(); i++) {
                results_file << results[i];
            }
        }

        std::vector<string>& get_nearest_neighbours_by_linear_method(std::string query, uint64_t edit_distance_threshold) {
            auto query_tf_idf_vector = getQuery_tf_idf_vector(query);
            std::vector<std::string> * nearest_neighbours = new std::vector<std::string>();
            #pragma omp parallel for
            for (uint64_t i = 0; i < original_data.size(); i++) {
                auto edit_distance = uiLevenshteinDistance(query, original_data[i]);
                if(edit_distance == 0){
                    continue;
                }
                else if(edit_distance <= edit_distance_threshold){
                    nearest_neighbours->push_back(original_data[i]);
                    //std::cout << original_data[i] << std::endl;
                }else {
                    continue;
                }
            }
            return *nearest_neighbours;
        }

        uint8_t getCategoryIndex(uint16_t editDistance){
            if(editDistance <= 15){
                return 1;
            }
            else if(editDistance <= 20){
                return 2;
            }
            else if(editDistance <= 25) {
                return 3;
            }
            else {
                return 4;
            }
        }

        std::string getEditDistanceCategory(uint8_t category){
            if(category == 1){
                return "ED <= 15";
            }
            else if(category == 2){
                return "15 < ED <= 20";
            }
            else if(category == 3){
                return "20 < ED <= 25";
            }
            else {
                return "ED > 25";
            }
        }

        void get_nearest_neighbours_by_linear_method_using_multiple_methods(ofstream& results_file, std::string query, uint64_t edit_distance_threshold, double_t cosine_distance_threshold) {
            auto query_tf_idf_vector = getQuery_tf_idf_vector(query);
            auto query_pure_tf_idf_vector = get_pure_tf_idf_vector(query);
            std::map<uint64_t, std::vector<std::pair<uint16_t, std::string>>> nearest_neighbours;

            struct ComparePairs {
                bool operator()(std::pair<uint16_t, std::string> i, std::pair<uint16_t, std::string> j) {
                    return (i.first < j.first);
                }
            };


            for(uint8_t i = 1; i <= 4; i++){
                nearest_neighbours[i] = std::vector<std::pair<uint16_t, std::string>>();
            }
            uint64_t editDistanceMatches = 0;
            #pragma omp parallel for
            for (uint64_t i = 0; i < original_data.size(); i++) {
                auto data_item_pure_tf_idf_vector = get_pure_tf_idf_vector(original_data[i]);
                auto edit_distance = uiLevenshteinDistance(query, original_data[i]);
                auto cosine_distance = dataset[i].dot(query_tf_idf_vector);
                auto euclidean_distance = (dataset[i] - query_tf_idf_vector).squaredNorm();
                auto pure_cosine_distance = data_item_pure_tf_idf_vector.dot(query_pure_tf_idf_vector)/(data_item_pure_tf_idf_vector.norm() * query_pure_tf_idf_vector.norm());
                auto pure_euclidean_distance = (data_item_pure_tf_idf_vector - query_pure_tf_idf_vector).squaredNorm();

                if(edit_distance <= edit_distance_threshold){
                    std::string t = original_data[i] + " " + to_string(pure_cosine_distance) + "(" + to_string(cosine_distance) + ") " + to_string(edit_distance) + " " + to_string(pure_euclidean_distance) + "(" + to_string(euclidean_distance) + ")";
                    uint8_t category = getCategoryIndex(edit_distance);
                    editDistanceMatches++;
                    nearest_neighbours[category].push_back(std::make_pair(edit_distance,t));
                }
            }

            results_file << "Edit_distance based matches(count:"+to_string(editDistanceMatches)+"):" << std::endl;
            for(auto category:nearest_neighbours){
                results_file << "Matches with " << getEditDistanceCategory(category.first) << " (count:"+to_string(category.second.size())+"):"<< endl;
                std::sort(category.second.begin(), category.second.end(), ComparePairs());
                for(auto p: category.second){
                    results_file << p.second << endl;
                }
            }

            for(uint8_t i = 1; i <= 4; i++){
                nearest_neighbours[i].clear();
            }

            uint64_t cosineDistanceMatches = 0;
            #pragma omp parallel for ordered
            for (uint64_t i = 0; i < original_data.size(); i++) {
                auto data_item_pure_tf_idf_vector = get_pure_tf_idf_vector(original_data[i]);
                auto edit_distance = uiLevenshteinDistance(query, original_data[i]);
                auto cosine_distance = dataset[i].dot(query_tf_idf_vector);
                auto euclidean_distance = (dataset[i] - query_tf_idf_vector).squaredNorm();
                auto pure_cosine_distance = data_item_pure_tf_idf_vector.dot(query_pure_tf_idf_vector)/(data_item_pure_tf_idf_vector.norm() * query_pure_tf_idf_vector.norm());
                auto pure_euclidean_distance = (data_item_pure_tf_idf_vector - query_pure_tf_idf_vector).squaredNorm();

                if(pure_cosine_distance >= cosine_distance_threshold){
                    std::string t = original_data[i] + " " + to_string(pure_cosine_distance) + "(" + to_string(cosine_distance) + ") " + to_string(edit_distance) + " " + to_string(pure_euclidean_distance) + "(" + to_string(euclidean_distance) + ")";
                    #pragma omp ordered
                    {
                        cosineDistanceMatches++;
                        uint8_t category = getCategoryIndex(edit_distance);
                        nearest_neighbours[category].push_back(std::make_pair(edit_distance,t));
                    }
                }
            }

            results_file << "Cosine_Similarity(Th:0.9) based matches(count:"+to_string(cosineDistanceMatches)+"):" << std::endl;
            for(auto category:nearest_neighbours){
                results_file << "Matches with " << getEditDistanceCategory(category.first) << " (count:"+to_string(category.second.size())+"):"<< endl;
                std::sort(category.second.begin(), category.second.end(), ComparePairs());
                for(auto p: category.second){
                    results_file << p.second << endl;
                }
            }

            for(uint8_t i = 1; i <= 4; i++){
                nearest_neighbours[i].clear();
            }

            for(uint8_t threshold = 10; threshold <= 150; threshold += 10){
                setThreshold(threshold/100.0);
                uint64_t falconn_matches = 0;
                for(auto falconn_match: match(query).second){
                    auto data_item_pure_tf_idf_vector = get_pure_tf_idf_vector(falconn_match);
                    auto edit_distance = uiLevenshteinDistance(query, falconn_match);
                    auto cosine_distance = query_tf_idf_vector.dot(getQuery_tf_idf_vector(falconn_match));
                    auto euclidean_distance = (getQuery_tf_idf_vector(falconn_match) - query_tf_idf_vector).squaredNorm();
                    auto pure_cosine_distance = data_item_pure_tf_idf_vector.dot(query_pure_tf_idf_vector)/(data_item_pure_tf_idf_vector.norm() * query_pure_tf_idf_vector.norm());
                    auto pure_euclidean_distance = (data_item_pure_tf_idf_vector - query_pure_tf_idf_vector).squaredNorm();
                    std::string t = falconn_match + " " + to_string(pure_cosine_distance) + "(" + to_string(cosine_distance) + ") " + to_string(edit_distance) + " " + to_string(pure_euclidean_distance) + "(" + to_string(euclidean_distance) + ")";
                    falconn_matches++;
                    uint8_t category = getCategoryIndex(edit_distance);
                    nearest_neighbours[category].push_back(std::make_pair(edit_distance,t));
                }

                results_file << "Falconn(Th:"<< threshold/100.0 <<") based matches(count:"+to_string(falconn_matches)+"):" << std::endl;
                for(auto category:nearest_neighbours){
                    results_file << "Matches with " << getEditDistanceCategory(category.first) << " (count:"+to_string(category.second.size())+"):"<< endl;
                    std::sort(category.second.begin(), category.second.end(), ComparePairs());
                    for(auto p: category.second){
                        results_file << p.second << endl;
                    }
                }
                for(uint8_t i = 1; i < 4; i++){
                    nearest_neighbours[i].clear();
                }
            }
        }

        std::map<std::string,uint64_t> getCategoryCounts(std::string query, std::vector<std::string> candidates){
            std::map<std::string,uint64_t> categoryCounts;
            for(uint8_t i = 1; i <= 4; i++){
                categoryCounts[getEditDistanceCategory(i)] = 0;
            }
            //std::cout << "Q: " << query << std::endl;
            //std::cout << "Candidates: " << (int)candidates.size() << std::endl;
            //bool first = true;
            for(uint64_t i = 0; i < candidates.size() ; i++){
                /*if(i == 0 && !first){
                    std::cout << "Problem!" << std::endl;
                }else{
                    first = false;
                }*/

                //std::cout << "C" << (int)i;
                auto edit_distance = uiLevenshteinDistance(query, candidates[i]);
                //std::cout << " " << (int)edit_distance << "\t";
                categoryCounts[getEditDistanceCategory(getCategoryIndex(edit_distance))]++;
            }
            //std::cout << std::endl;
            //std::cout << "Complete" << std::endl;
            return std::move(categoryCounts);
        }

        std::pair<uint64_t, uint64_t> count_nearest_neighbours(std::string query) {
            auto query_tf_idf_vector = getQuery_tf_idf_vector(query);
            std::pair<uint64_t, uint64_t> nnPair;
            uint8_t minEd = 100;
            #pragma omp parallel for
            for (uint64_t i = 0; i < original_data.size(); i++) {
                auto edit_distance = uiLevenshteinDistance(query, original_data[i]);
                if(edit_distance == 0){
                    continue;
                }
                if(edit_distance < minEd){
                    minEd = edit_distance;
                    nnPair.first = edit_distance;
                    nnPair.second = 0;
                }else if(edit_distance > minEd){
                    continue;
                }
                nnPair.second += 1;
            }
            return nnPair;
        }
    };
}

