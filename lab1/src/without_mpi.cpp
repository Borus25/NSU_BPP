#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

// --- МОДУЛЬ 1: ИНИЦИАЛИЗАЦИЯ ---
void init_system(size_t N, std::vector<double>& A, std::vector<double>& b) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            // Матрица: 2.0 на диагонали, 1.0 в остальных местах
            A[i * N + j] = (i == j) ? 2.0 : 1.0;
        }
        b[i] = static_cast<double>(N) + 1.0;
    }
}

// --- МОДУЛЬ 2: МАТЕМАТИЧЕСКОЕ ЯДРО ---
double do_iteration(size_t N, const std::vector<double>& A, const std::vector<double>& b, 
                    const std::vector<double>& x, std::vector<double>& x_next, double tau) {
    double res_norm_sq = 0;

    for (size_t i = 0; i < N; ++i) {
        double Ax_i = 0;
        for (size_t j = 0; j < N; ++j) {
            Ax_i += A[i * N + j] * x[j];
        }

        double residual = Ax_i - b[i];
        res_norm_sq += residual * residual;
        
        x_next[i] = x[i] - tau * residual;
    }
    return std::sqrt(res_norm_sq);
}

// --- МОДУЛЬ 3: ОРКЕСТРАТОР (SOLVER) ---
void solve_serial(int N, double tau, double eps) {
    // Выделение памяти (вся матрица целиком)
    std::vector<double> A((size_t)N * N);
    std::vector<double> b(N);
    std::vector<double> x(N, 0.0);
    std::vector<double> x_next(N);

    init_system(N, A, b);

    // Считаем норму b
    double b_norm_sq = 0;
    for (double val : b) b_norm_sq += val * val;
    double b_norm = std::sqrt(b_norm_sq);

    int iter = 0;
    bool converged = false;

    // Начало замера времени
    auto start = std::chrono::high_resolution_clock::now();

    while (!converged) {
        double res_norm = do_iteration(N, A, b, x, x_next, tau);

        if (res_norm / b_norm < eps) {
            converged = true;
        }

        x = x_next;
        iter++;
        
        // Защита от бесконечного цикла
        if (iter > 50000) break; 
    }

    // Конец замера времени
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    std::cout << "[Serial] Finished. Iters: " << iter 
              << ", Time: " << diff.count() << "s" << std::endl;
}

int main() {
    // Используем тот же N, что и в MPI версии для честного сравнения
    const int N = 25000;
    const double tau = 0.00001;
    const double eps = 1e-5;

    solve_serial(N, tau, eps);

    return 0;
}
