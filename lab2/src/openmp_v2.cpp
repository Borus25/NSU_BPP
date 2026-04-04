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

    double b_norm = 0;
    int iter = 0;
    bool converged = false;
    double current_res_norm_sq = 0;

    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel default(none) shared(A, b, x, x_next, b_norm, converged, iter, current_res_norm_sq)
    {
        #pragma omp for schedule(runtime)
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) A[i * N + j] = (i == j) ? 2.0 : 1.0;
            b[i] = (double)N + 1.0;
        }

        double local_b_sum = 0;
        #pragma omp for schedule(runtime)
        for (int i = 0; i < N; i++) local_b_sum += b[i] * b[i];

        #pragma omp atomic
        b_norm += local_b_sum;

        #pragma omp barrier

        #pragma omp single
        b_norm = std::sqrt(b_norm);

        while (!converged) {
            #pragma omp single
            current_res_norm_sq = 0;

            #pragma omp for schedule(runtime) reduction(+:current_res_norm_sq)
            for (int i = 0; i < N; ++i) {
                double Ax_i = 0;
                for (int j = 0; j < N; ++j) Ax_i += A[i * N + j] * x[j];
                double residual = Ax_i - b[i];
                current_res_norm_sq += residual * residual;
                x_next[i] = x[i] - tau * residual;
            }

            #pragma omp single
            {
                if (std::sqrt(current_res_norm_sq) / b_norm < eps) converged = true;
                x = x_next;
                iter++;
                if (iter > 50000) converged = true;
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    // --- БЛОК ПРОВЕРКИ РЕШЕНИЯ (Verification) ---
    double final_res_norm_l1 = 0;
    // Считаем L1-норму невязки: сумма абсолютных значений |Ax - b|
    for (int i = 0; i < N; ++i) {
        double Ax_i = 0;
        for (int j = 0; j < N; ++j) {
            Ax_i += A[i * N + j] * x[j];
        }
        final_res_norm_l1 += std::abs(Ax_i - b[i]);
    }

    std::cout << "Time: " << std::chrono::duration<double>(end-start).count() << " sec" << std::endl;
    std::cout << "Iterations: " << iter << std::endl;
    std::cout << "Verification (Residual Sum |Ax-b|): " << final_res_norm_l1 << std::endl;

    return 0;
}