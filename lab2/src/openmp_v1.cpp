#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <omp.h>

int main() {
    const int N = 13000;
    const double tau = 0.00001;
    const double eps = 1e-5;

    std::vector<double> A((size_t)N * N);
    std::vector<double> b(N);
    std::vector<double> x(N, 0.0);
    std::vector<double> x_next(N);

    #pragma omp parallel for schedule(runtime)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) A[i * N + j] = (i == j) ? 2.0 : 1.0;
        b[i] = (double)N + 1.0;
    }

    double b_norm_sq = 0;
    #pragma omp parallel for reduction(+:b_norm_sq) schedule(runtime)
    for (int i = 0; i < N; i++) b_norm_sq += b[i] * b[i];
    double b_norm = std::sqrt(b_norm_sq);

    auto start = std::chrono::high_resolution_clock::now();
    int iter = 0;
    bool converged = false;

    while (!converged) {
        double res_norm_sq = 0;
        #pragma omp parallel for reduction(+:res_norm_sq) schedule(runtime)
        for (int i = 0; i < N; ++i) {
            double Ax_i = 0;
            for (int j = 0; j < N; ++j) Ax_i += A[i * N + j] * x[j];
            double residual = Ax_i - b[i];
            res_norm_sq += residual * residual;
            x_next[i] = x[i] - tau * residual;
        }

        if (std::sqrt(res_norm_sq) / b_norm < eps) converged = true;
        x = x_next;
        if (++iter > 50000) break;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time: " << std::chrono::duration<double>(end-start).count() << " sec" << std::endl;

    // ПРОВЕРКА
    double diff = 0;
    for (int i = 0; i < N; i++) {
        double Ax_i = 0;
        for (int j = 0; j < N; j++) Ax_i += A[i * N + j] * x[j];
        diff += std::abs(Ax_i - b[i]);
    }
    std::cout << "Verification: " << diff << std::endl;
    return 0;
}