#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <cmath>
#include <map>
#include <algorithm>

namespace py = pybind11;
using namespace Eigen;

MatrixXf calculate_covariance_matrix(const Eigen::Ref<const Eigen::MatrixXf>& P, const Eigen::Ref<const Eigen::MatrixXf>& Q, int max_displacement) {
    if (P.rows() != Q.rows() || P.cols() != Q.cols()) {
        throw std::invalid_argument("P and Q must have the same dimensions.");
    }

    int N_rows = P.rows();
    int N_cols = P.cols();
    int D = 2 * max_displacement + 1;
    float normalization_factor = 1.0f / (N_rows * N_cols); 

    float P_sum = P.sum();
    float P_mean = P_sum * normalization_factor;

    std::vector<std::pair<int, int>> displacements;
    for (int dy = -max_displacement; dy <= max_displacement; ++dy) {
        for (int dx = -max_displacement; dx <= max_displacement; ++dx) {
            displacements.emplace_back(dy, dx);
        }
    }

    MatrixXf covariance_matrix = MatrixXf::Zero(D, D);

    for (size_t idx = 0; idx < displacements.size(); ++idx) {
        int dy = displacements[idx].first;
        int dx = displacements[idx].second;

        MatrixXf Q_shifted = MatrixXf::Zero(N_rows, N_cols);
        for (int y = 0; y < N_rows; ++y) {
            for (int x = 0; x < N_cols; ++x) {
                int new_y = y + dy;
                int new_x = x + dx;
                if (new_y >= 0 && new_y < Q.rows() && new_x >= 0 && new_x < Q.cols()) {
                    Q_shifted(y, x) = Q(new_y, new_x);
                }
            }
        }

        float shifted_sum = Q_shifted.sum();
        float product_sum = (P.array() * Q_shifted.array()).sum();
        float term1 = product_sum * normalization_factor;
        float term2 = P_mean * (shifted_sum * normalization_factor);

        covariance_matrix(idx / D, idx % D) = term1 - term2;
    }

    float min_cov = covariance_matrix.minCoeff();
    float max_cov = covariance_matrix.maxCoeff();
    if (max_cov > min_cov) {
        covariance_matrix = 255 * (covariance_matrix.array() - min_cov) / (max_cov - min_cov);
    }

    return covariance_matrix;
}

Eigen::MatrixXf laplacian_operator(const Eigen::Ref<const Eigen::MatrixXf>& image, int s = 1) {
    int rows = image.rows(), cols = image.cols();
    if (rows <= 2 * s || cols <= 2 * s) {
        throw std::invalid_argument("Image dimensions must be larger than 2 * s.");
    }

    Eigen::MatrixXf laplacian = Eigen::MatrixXf::Zero(rows, cols);

    #pragma omp parallel for
    for (int y = s; y < rows - s; ++y) {
        for (int x = s; x < cols - s; ++x) {
            laplacian(y, x) = (image(y, x - s) - 2 * image(y, x) + image(y, x + s) +
                               image(y - s, x) - 2 * image(y, x) + image(y + s, x)) / (s * s);
        }
    }
    return laplacian;
}

std::pair<int, int> calculate_covariance_displacement(const std::vector<std::pair<int, int>>& swath_coords, 
                                                      const MatrixXf& image, 
                                                      const MatrixXf& reference_image, 
                                                      int N = 48, 
                                                      int max_displacement = 24) {
    std::vector<std::pair<int, int>> displacements;
    int half_N = N / 2;

    MatrixXf lap_image = laplacian_operator(image, 1);
    MatrixXf lap_ref_image = laplacian_operator(reference_image, 1);

    #pragma omp parallel
    {
        std::vector<std::pair<int, int>> local_displacements;

        #pragma omp for nowait
        for (size_t i = 0; i < swath_coords.size(); ++i) {
            const auto& coord = swath_coords[i];
            int y = coord.first, x = coord.second;

            if (y - half_N < 0 || y + half_N > lap_image.rows() || 
                x - half_N < 0 || x + half_N > lap_image.cols()) {
                continue;
            }

            MatrixXf image_block = lap_image.block(y - half_N, x - half_N, N, N);
            MatrixXf ref_image_block = lap_ref_image.block(y - half_N, x - half_N, N, N);

            if (ref_image_block.size() == 0) {
                continue;
            }

            MatrixXf res = calculate_covariance_matrix(ref_image_block, image_block, max_displacement);

            MatrixXf::Index maxRow, maxCol;
            res.maxCoeff(&maxRow, &maxCol);
            float mean_cov = res.mean();

            int dy = maxRow, dx = maxCol, radius = 2;
            std::vector<float> circumference_values;

            for (int t = 0; t < 100; ++t) {
                double angle = 2 * M_PI * t / 100;
                int circ_y = static_cast<int>(dy + radius * sin(angle));
                int circ_x = static_cast<int>(dx + radius * cos(angle));
                if (circ_y >= 0 && circ_y < res.rows() && circ_x >= 0 && circ_x < res.cols()) {
                    circumference_values.push_back(res(circ_y, circ_x));
                }
            }

            float CL = !circumference_values.empty() ? *std::max_element(circumference_values.begin(), circumference_values.end()) : 0;

            float score = mean_cov + CL;
            if (score < 237) {
                local_displacements.emplace_back(dy - max_displacement, dx - max_displacement);
            }
        }

        #pragma omp critical
        displacements.insert(displacements.end(), local_displacements.begin(), local_displacements.end());
    }

    std::map<std::pair<int, int>, int> displacement_counts;
    for (const auto& d : displacements) {
        if (d.first != -max_displacement && d.second != -max_displacement) {
            displacement_counts[d]++;
        }
    }

    auto most_common = std::max_element(displacement_counts.begin(), displacement_counts.end(),
                                        [](const auto& a, const auto& b) {
                                            return a.second < b.second;
                                        });

    if (most_common != displacement_counts.end()) {
        return most_common->first;
    }

    return {0, 0};
}

PYBIND11_MODULE(discalc, m) {
    m.doc() = "Optimized image processing functions using Eigen";
    m.def("calculate_covariance_matrix", &calculate_covariance_matrix, "Calculate covariance matrix");
    m.def("laplacian_operator", &laplacian_operator, "Apply Laplacian operator");
    m.def("calculate_covariance_displacement", &calculate_covariance_displacement, "Calculate covariance displacement");
}
