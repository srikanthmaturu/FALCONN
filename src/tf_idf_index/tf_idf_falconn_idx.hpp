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
#include "tf_idf_falconn_idx_helper.hpp"

#if CUSTOM_BOOST_ENABLED
#include "eigen_boost_serialization.hpp"
#include <boost/serialization/vector.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#endif

#include <stdlib.h>
#include <falconn/lsh_nn_table.h>

namespace tf_idf_falconn_index {
    template<uint64_t ngram_length_t, uint8_t threshold_t, class point_type_t=DenseVectorFloat>
    class tf_idf_falconn_idx {
    public:
        tf_idf_falconn_idx() = default;

        tf_idf_falconn_idx(const tf_idf_falconn_idx &) = default;

        tf_idf_falconn_idx(tf_idf_falconn_idx &&) = default;

        tf_idf_falconn_idx &operator=(const tf_idf_falconn_idx &) = default;

        tf_idf_falconn_idx &operator=(tf_idf_falconn_idx &&) = default;

        tf_idf_falconn_idx(std::vector<std::string> &data) {
            original_data = data;
            construct_dataset(data);
            params.dimension = dataset[0].size();
            params.lsh_family = falconn::LSHFamily::CrossPolytope;
            params.l = 50;
            params.distance_function = falconn::DistanceFunction::EuclideanSquared;
            falconn::compute_number_of_hash_functions<point_type>(18, &params);
            params.num_rotations = 1;
            // we want to use all the available threads to set up
            params.num_setup_threads = 0;
            params.storage_hash_table = falconn::StorageHashTable::BitPackedFlatHashTable;
            falconn::compute_number_of_hash_functions<point_type>(18, &params);
            threshold = (double_t)threshold_t / (double_t)100;
            std::cout << "FALCONN threshold set to " << threshold << std::endl;
        }

        void setThreshold(){

        }


        typedef point_type_t point_type;
        typedef vector<point_type> Dataset;

        Dataset dataset;
        point_type center;
        falconn::LSHConstructionParameters params;
        uint8_t num_probes = 50;
        double_t threshold;

        std::vector<std::string> original_data;
        uint64_t ngram_length = ngram_length_t;
        std::vector<uint64_t> tdfs;
        std::map<char, int> a_map = {{'A', 0},
                                     {'C', 1},
                                     {'G', 2},
                                     {'T', 3},
                                     {'N', 4}};

        unique_ptr<falconn::LSHNearestNeighborTable<point_type>> table;

        void construct_table() {
            std::cout << "Dataset size: " << dataset.size() << std::endl;
            table = std::move(falconn::construct_table<point_type>(dataset, params));
            table->reset_query_statistics();
        }

        #if CUSTOM_BOOST_ENABLED
        void store_to_file(std::string idx_file) {
            std::ofstream idx_file_ofs(idx_file);
            boost::archive::binary_oarchive oa(idx_file_ofs);
            oa << original_data;
            oa << tdfs;
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
            ia >> dataset;
            ia >> center;
            ia >> params;
            ia >> threshold;
        }
        #endif

        typename std::enable_if<std::is_same<point_type, DenseVectorFloat>::value,std::pair<uint64_t, std::vector<std::string>>>::type match(std::string query) {
            auto query_tf_idf_vector = getQuery_tf_idf_vector(query);
            std::cout << std::endl;
            std::vector<int32_t > nearestNeighbours;
            table->find_near_neighbors(query_tf_idf_vector, threshold, &nearestNeighbours);
            //nearestNeighbours.push_back(table->find_nearest_neighbor(query_tf_idf_vector));
            std::vector<std::string> matches;
            for (auto i:nearestNeighbours) {
                matches.push_back(original_data[i]);
            }
            return std::make_pair(matches.size(), matches);
        }


        typename std::enable_if<std::is_same<point_type, DenseVectorFloat>::value, DenseVectorFloat>::type getQuery_tf_idf_vector(std::string query) {
            uint64_t tf_vec_size = pow(4, ngram_length);
            uint64_t string_size = query.size();
            uint64_t data_size = dataset.size();
            DenseVectorFloat point;
            vector<float> tf_idf_vector(tf_vec_size, 0.0);
            for (uint64_t i = 0; i < string_size - ngram_length + 1; i++) {
                std::string ngram = query.substr(i, ngram_length);
                uint64_t d_num = 0;
                for (uint64_t j = 0; j < ngram_length; j++) {
                    d_num += a_map[ngram[j]] * pow(4, (ngram_length - j - 1));
                }
                tf_idf_vector[d_num]++;
            }
            double vec_sq_sum = 0.0;
            for (uint64_t i = 0; i < tf_vec_size; i++) {
                if (tf_idf_vector[i] > 0 && tdfs[i] > 0) {
                    tf_idf_vector[i] = (1 + log10(tf_idf_vector[i])) * ((log10(1 + (data_size / tdfs[i]))));
                    vec_sq_sum += pow(tf_idf_vector[i], 2);
                }
            }
            vec_sq_sum = pow(vec_sq_sum, 0.5);
            point.resize(tf_vec_size);
            for (uint64_t i = 0; i < tf_vec_size; i++) {
                tf_idf_vector[i] /= vec_sq_sum;
                point[i] = tf_idf_vector[i];
                //std::cout << tf_idf_vector[i] << " \t";
            }
            //std::cout << std::endl;
            point -= center;
            return point;
        }

        typename std::enable_if<std::is_same<point_type, DenseVectorFloat>::value,void>::type construct_dataset(std::vector<std::string> &data) {
            uint64_t tf_vec_size = pow(4, ngram_length);
            uint64_t data_size = data.size();
            uint64_t string_size = data[0].size();
            tdfs.resize(tf_vec_size);
            vector<vector<float>> tf_idf_vectors(data_size, vector<float>(tf_vec_size, 0));
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
                        tdfs[j]++;
                    }
                }
            }
            for (uint64_t i = 0; i < data_size; i++) {
                double_t vec_sq_sum = 0.0;
                for (uint64_t j = 0; j < tf_vec_size; j++) {
                    tf_idf_vectors[i][j] *= (log10(1 + (data_size / tdfs[j])));
                    vec_sq_sum += pow(tf_idf_vectors[i][j], 2);
                }
                vec_sq_sum = pow(vec_sq_sum, 0.5);
                DenseVectorFloat point;
                point.resize(tf_vec_size);
                for (uint64_t j = 0; j < tf_vec_size; j++) {
                    tf_idf_vectors[i][j] /= vec_sq_sum;
                    point[j] = tf_idf_vectors[i][j];
                    //std::cout << tf_idf_vectors[i][j] << " \t";
                }
                //std::cout << std::endl;
                dataset.push_back(point);
            }

            std::cout << "Normalizing dataset points." << std::endl;

            for (auto &p : dataset) {
                p.normalize();
            }

            std::cout << "Done."<< std::endl;
            // find the center of mass
            center = dataset[rand()%data_size];
            for (size_t i = 1; i < data_size; ++i) {
                center += dataset[i];
            }
            center /= dataset.size();

            std::cout << "Re-centering dataset points." << std::endl;
            for (auto &datapoint : dataset) {
                datapoint -= center;
            }
            std::cout << "Done."<<std::endl;
        }

    };

}