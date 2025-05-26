#include <algorithm>
#include <cmath>
#include <Eigen/Dense>
#include <iostream>
#include <map>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

std::tuple<float, float> third_order_polynomial_fitting_4x4(const Eigen::MatrixXf &res, int dx, int dy)
{
    constexpr std::array<std::array<double, 4>, 4> weights = {{{1.0, 2.0, 2.0, 1.0},
                                                               {2.0, 4.0, 4.0, 2.0},
                                                               {2.0, 4.0, 4.0, 2.0},
                                                               {1.0, 2.0, 2.0, 1.0}}};

    Eigen::Matrix<double, 16, 10> A;
    Eigen::Vector<double, 16> b;
    int idx = 0;

    for (int j = -2; j <= 1; ++j)
    {
        for (int i = -2; i <= 1; ++i)
        {
            const int xi = dx + i, yi = dy + j;
            if (xi < 0 || xi >= res.cols() || yi < 0 || yi >= res.rows())
                continue;

            const double x = i + 0.5, y = j + 0.5;
            const double x2 = x * x, y2 = y * y;
            const double w = weights[j + 2][i + 2];

            A.row(idx) << x * x * x, y * y * y, x2 * y, x * y2, x2, y2, x * y, x, y, 1.0;
            A.row(idx) *= w;
            b[idx++] = w * res(yi, xi);
        }
    }

    if (idx < 10)
        return {dx - 0.5f, dy - 0.5f};

    const Eigen::Matrix<double, 10, 1> params = A.topRows(idx).jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b.head(idx));
    const Eigen::Matrix2d Hessian = (Eigen::Matrix2d() << 2 * params[4], params[6], params[6], 2 * params[5]).finished();
    const Eigen::Vector2d grad(params[7], params[8]);

    if (std::abs(Hessian.determinant()) < 1e-6)
        return {dx - 0.5f, dy - 0.5f};

    const Eigen::Vector2d offset = -Hessian.inverse() * grad;
    return {dx - 0.5f + static_cast<float>(offset[0]), dy - 0.5f + static_cast<float>(offset[1])};
}

std::tuple<float, float> third_order_polynomial_fitting(const Eigen::MatrixXf &res, int dy, int dx, bool expand_x)
{
    const int i_min = expand_x ? -2 : -1, i_max = 1;
    const int j_min = expand_x ? -1 : -2, j_max = 1;

    Eigen::Matrix<double, 12, 9> A;
    Eigen::Vector<double, 12> b;
    int idx = 0;

    for (int j = j_min; j <= j_max; ++j)
    {
        for (int i = i_min; i <= i_max; ++i)
        {
            const int xi = dx + i, yi = dy + j;
            if (xi < 0 || xi >= res.cols() || yi < 0 || yi >= res.rows())
                continue;

            const double x = i, y = j;
            const double x2 = x * x, y2 = y * y;
            double weight = 1.0;

            if (x == 0 && y == 0)
                weight = 4.0;
            else if ((expand_x && abs(i) == 1 && j == 0) || (!expand_x && abs(j) == 1 && i == 0))
                weight = 4.0;
            else if (abs(i) + abs(j) == 1)
                weight = 2.0;

            A.row(idx) << x * x * x, x2 * y, x * y2, x2, y2, x * y, x, y, 1.0;
            A.row(idx) *= weight;
            b[idx++] = weight * res(yi, xi);
        }
    }

    if (idx < 9)
        return {static_cast<float>(dx), static_cast<float>(dy)};

    const Eigen::Matrix<double, 9, 1> params = A.topRows(idx).jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b.head(idx));
    const Eigen::Matrix2d M = (Eigen::Matrix2d() << 2 * params[3], params[5], params[5], 2 * params[4]).finished();
    const Eigen::Vector2d rhs(-params[6], -params[7]);

    if (std::abs(M.determinant()) < 1e-6)
        return {static_cast<float>(dx), static_cast<float>(dy)};

    const Eigen::Vector2d offset = M.inverse() * rhs;
    return {static_cast<float>(dx + offset[0]), static_cast<float>(dy + offset[1])};
}
std::tuple<float, float> second_order_polynomial_fitting(const Eigen::MatrixXf &res, int dy, int dx)
{
    constexpr std::array<std::array<double, 3>, 3> weights = {{{1.0, 2.0, 1.0},
                                                               {2.0, 4.0, 2.0},
                                                               {1.0, 2.0, 1.0}}};

    Eigen::Matrix<double, 9, 6> A;
    Eigen::Vector<double, 9> b;
    int idx = 0;

    for (int j = -1; j <= 1; ++j)
    {
        for (int i = -1; i <= 1; ++i)
        {
            const double x = i, y = j;
            const double w = weights[j + 1][i + 1];

            A.row(idx) << x * x, y * y, x * y, x, y, 1.0;
            A.row(idx) *= w;
            b[idx++] = w * res(dy + j, dx + i);
        }
    }

    const Eigen::Matrix<double, 6, 1> params = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
    const Eigen::Matrix2d M = (Eigen::Matrix2d() << 2 * params[0], params[2], params[2], 2 * params[1]).finished();

    if (M.determinant() < 1e-6)
        return {static_cast<float>(dx), static_cast<float>(dy)};

    const Eigen::Vector2d offset = M.inverse() * Eigen::Vector2d(-params[3], -params[4]);
    return {static_cast<float>(dx + offset[0]), static_cast<float>(dy + offset[1])};
}

std::tuple<float, float> subpixel_maximum(const Eigen::MatrixXf &res, int dy, int dx)
{
    if (dx < 1 || dx >= res.cols() - 1 || dy < 1 || dy >= res.rows() - 1)
    {
        return {static_cast<float>(dx), static_cast<float>(dy)};
    }

    const float x_vals[3] = {res(dy, dx - 1), res(dy, dx), res(dy, dx + 1)};
    const float a_x = (x_vals[2] + x_vals[0] - 2 * x_vals[1]) / 2.0f;
    const float b_x = (x_vals[2] - x_vals[0]) / 2.0f;
    const float x0 = a_x < -1e-6f ? std::clamp(dx - b_x / (2 * a_x), 0.0f, float(res.cols() - 1)) : dx;

    const float y_vals[3] = {res(dy - 1, dx), res(dy, dx), res(dy + 1, dx)};
    const float a_y = (y_vals[2] + y_vals[0] - 2 * y_vals[1]) / 2.0f;
    const float b_y = (y_vals[2] - y_vals[0]) / 2.0f;
    const float y0 = a_y < -1e-6f ? std::clamp(dy - b_y / (2 * a_y), 0.0f, float(res.rows() - 1)) : dy;

    const float dx_diff = std::abs(x0 - dx);
    const float dy_diff = std::abs(y0 - dy);

    if (dx_diff <= 0.25f && dy_diff <= 0.25f)
        return second_order_polynomial_fitting(res, dy, dx);
    if (dx_diff > 0.25f && dy_diff <= 0.25f)
        return third_order_polynomial_fitting(res, dy, dx, true);
    if (dy_diff > 0.25f && dx_diff <= 0.25f)
        return third_order_polynomial_fitting(res, dy, dx, false);
    return third_order_polynomial_fitting_4x4(res, dy, dx);
}

Eigen::MatrixXf calculate_covariance_matrix(const Eigen::Ref<const Eigen::MatrixXf> &P,
                                            const Eigen::Ref<const Eigen::MatrixXf> &Q,
                                            int max_displacement)
{
    if (P.rows() != Q.rows() || P.cols() != Q.cols() || P.rows() != P.cols())
    {
        throw std::invalid_argument("P and Q must be square matrices of the same size.");
    }

    int N = P.rows();
    int D = 2 * max_displacement + 1;
    float normalization_factor = 1.0f / (N * N);

    float P_mean = P.sum() * normalization_factor;

    Eigen::MatrixXf covariance_matrix = Eigen::MatrixXf::Zero(D, D);

    for (int dy = -max_displacement; dy <= max_displacement; ++dy)
    {
        for (int dx = -max_displacement; dx <= max_displacement; ++dx)
        {
            int y_start = std::max(0, -dy);
            int y_end = std::min(N, N - dy);
            int x_start = std::max(0, -dx);
            int x_end = std::min(N, N - dx);

            float sum_PQ = 0.0f;
            float sum_Q_shifted = 0.0f;

            for (int y = y_start; y < y_end; ++y)
            {
                for (int x = x_start; x < x_end; ++x)
                {
                    float p_val = P(y, x);
                    float q_val = Q(y + dy, x + dx);
                    sum_PQ += p_val * q_val;
                    sum_Q_shifted += q_val;
                }
            }
            float term1 = sum_PQ * normalization_factor;
            float term2 = P_mean * (sum_Q_shifted * normalization_factor);

            int row = dy + max_displacement;
            int col = dx + max_displacement;
            covariance_matrix(row, col) = term1 - term2;
        }
    }

    float min_cov = covariance_matrix.minCoeff();
    float max_cov = covariance_matrix.maxCoeff();
    if (max_cov > min_cov)
        covariance_matrix = 255 * (covariance_matrix.array() - min_cov) / (max_cov - min_cov);
    else
        covariance_matrix.setZero();

    return covariance_matrix;
}

Eigen::MatrixXf laplacian_operator(const Eigen::Ref<const Eigen::MatrixXf> &image, int s = 1)
{
    int rows = image.rows(), cols = image.cols();

    if (rows <= 2 * s || cols <= 2 * s)
    {
        throw std::invalid_argument("Image dimensions must be larger than 2 * s.");
    }

    Eigen::MatrixXf laplacian = Eigen::MatrixXf::Zero(rows, cols);

#pragma omp parallel for
    for (int y = s; y < rows - s; ++y)
    {
        for (int x = s; x < cols - s; ++x)
        {
            float d2x = (image(y, x - s) - 2 * image(y, x) + image(y, x + s));
            float d2y = (image(y - s, x) - 2 * image(y, x) + image(y + s, x));

            laplacian(y, x) = (d2x + d2y) / (2 * s);
        }
    }

    return laplacian;
}

std::vector<std::tuple<float, float>> calculate_covariance_displacement(const std::vector<std::pair<int, int>> &swath_coords,
                                                                        const Eigen::MatrixXf &image,
                                                                        const Eigen::MatrixXf &reference_image,
                                                                        int N = 48,
                                                                        int max_displacement = 24)
{
    std::vector<std::tuple<float, float>> displacements(swath_coords.size());
    int half_N = N / 2;
    const int s{1};
    Eigen::MatrixXf lap_image = laplacian_operator(image, s);
    Eigen::MatrixXf lap_ref_image = laplacian_operator(reference_image, s);

#pragma omp parallel
    {
#pragma omp for nowait
        for (size_t i = 0; i < swath_coords.size(); ++i)
        {
            const auto &coord = swath_coords[i];
            int y = coord.first, x = coord.second;
            displacements[i] = {-100, -100};
            if (y - half_N < s || y + half_N > lap_image.rows() - s ||
                x - half_N < s || x + half_N > lap_image.cols() - s)
            {
                continue;
            }

            Eigen::MatrixXf image_block = lap_image.block(y - half_N, x - half_N, N, N);
            Eigen::MatrixXf ref_image_block = lap_ref_image.block(y - half_N, x - half_N, N, N);

            if (ref_image_block.size() == 0)
            {
                continue;
            }

            Eigen::MatrixXf res = calculate_covariance_matrix(image_block, ref_image_block, max_displacement);

            Eigen::MatrixXf::Index maxRow, maxCol;
            res.maxCoeff(&maxRow, &maxCol);
            float apex_val = res(maxRow, maxCol);
            if (apex_val < 254.5f)
            {
                continue;
            }
            int dy = maxRow, dx = maxCol;
            float mean_cov = res.mean();
            const int R = 2;
            std::vector<float> circ;
            circ.reserve(100);
            for (int t = 0; t < 100; ++t)
            {
                double theta = 2 * M_PI * t / 100.0;
                int y = static_cast<int>(std::round(maxRow + R * std::sin(theta)));
                int x = static_cast<int>(std::round(maxCol + R * std::cos(theta)));
                if (y >= 0 && y < res.rows() && x >= 0 && x < res.cols())
                    circ.push_back(res(y, x));
            }
            float CL = circ.empty() ? 0.f : *std::max_element(circ.begin(), circ.end());

            float score = mean_cov + CL;
            if (score < 237)
            {
                auto [x0, y0] = subpixel_maximum(res, dy, dx);
                x0 -= max_displacement;
                y0 -= max_displacement;
                if (std::abs(y0) < max_displacement && std::abs(x0) < max_displacement)
                    displacements[i] = {y0, x0};
            }
        }
    }
    return displacements;
}

PYBIND11_MODULE(displacement_calc, m)
{
    m.doc() = "Optimized image processing functions using Eigen";
    m.def("calculate_covariance_matrix", &calculate_covariance_matrix, "Calculate covariance matrix");
    m.def("laplacian_operator", &laplacian_operator, "Apply Laplacian operator");
    m.def("calculate_covariance_displacement", &calculate_covariance_displacement, "Calculate covariance displacement");
}
