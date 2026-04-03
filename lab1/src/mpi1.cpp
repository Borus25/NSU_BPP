#include <iostream>
#include <vector>
#include <cmath>
#include <openmpi-x86_64//mpi.h>

struct Partition {
    std::vector<int> counts; // Сколько строк у каждого процесса
    std::vector<int> displs; // Смещение (начало строк) для каждого процесса
    int local_n;             // Количество строк текущего процесса
    int start_row;           // Глобальный индекс первой строки процесса

    Partition(int N, int size, int rank) {
        counts.resize(size);
        displs.resize(size);
        int sum = 0;
        for (int i = 0; i < size; i++) {
            counts[i] = (N / size) + (i < (N % size) ? 1 : 0);
            displs[i] = sum;
            sum += counts[i];
        }
        local_n = counts[rank];
        start_row = displs[rank];
    }
};

// --- МОДУЛЬ 2: ИНИЦИАЛИЗАЦИЯ ДАННЫХ ---
void init_system(size_t N, const Partition& p, std::vector<double>& A_local, std::vector<double>& b) {
    for (size_t i = 0; i < N; ++i) {
        b[i] = static_cast<double>(N) + 1.0;
    }

    // Матрица A заполняется только в части строк текущего процесса
    for (size_t i = 0; i < static_cast<size_t>(p.local_n); ++i) {
        size_t global_i = static_cast<size_t>(p.start_row) + i;
        for (size_t j = 0; j < N; ++j) {
            // Матрица без диагонального преобладания
            A_local[i * N + j] = (global_i == j) ? 2.0 : 1.0;
        }
    }
}

// Сборка обновленного вектора x из локальных частей со всех процессов
void synchronize_vector(const std::vector<double>& local_part, std::vector<double>& full_vec, const Partition& p) {
    MPI_Allgatherv(local_part.data(), p.local_n, MPI_DOUBLE,
                   full_vec.data(), p.counts.data(), p.displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
}

// Глобальное суммирование (для вычисления нормы невязки)
double reduce_sum(double local_val) {
    double global_val;
    MPI_Allreduce(&local_val, &global_val, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return global_val;
}

// Вычисляет локальную часть невязки и часть вектора для следующей итерации
double compute_iteration_step(size_t N, const Partition& p,
                              const std::vector<double>& A_local, const std::vector<double>& b,
                              const std::vector<double>& x_current, std::vector<double>& x_next_local,
                              double tau) {
    double local_res_norm_sq = 0;
    for (size_t i = 0; i < static_cast<size_t>(p.local_n); ++i) {
        size_t global_i = static_cast<size_t>(p.start_row) + i;
        double Ax_i = 0;

        // Умножение локальной строки матрицы на полный вектор x
        for (size_t j = 0; j < N; ++j) {
            Ax_i += A_local[i * N + j] * x_current[j];
        }

        double residual = Ax_i - b[global_i];
        local_res_norm_sq += residual * residual;

        // Формула: x^{n+1} = x^n - tau * (Ax^n - b)
        x_next_local[i] = x_current[global_i] - tau * residual;
    }
    return local_res_norm_sq;
}

// --- МОДУЛЬ 5: ОРКЕСТРАТОР (SOLVER) ---
void solve_variant_1(int N, double tau, double eps) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Partition p(N, size, rank);

    // Выделение памяти (используем size_t для больших N)
    std::vector<double> A_local(static_cast<size_t>(p.local_n)* N);
    std::vector<double> b(N);
    std::vector<double> x(N, 0.0);
    std::vector<double> x_next_local(p.local_n);

    init_system(N, p, A_local, b);

    // В Варианте 1 норма b считается локально (она везде одинаковая)
    double b_norm = 0;
    for (double val : b) b_norm += val * val;
    b_norm = std::sqrt(b_norm);

    int iter = 0;
    bool converged = false;
    double start_time = MPI_Wtime();

    while (!converged) {
        // 1. Математический шаг (вычисляем свою часть x_next и локальную невязку)
        double local_norm_sq = compute_iteration_step(N, p, A_local, b, x, x_next_local, tau);

        // 2. Синхронизация: узнаем глобальную невязку
        double global_res_norm = std::sqrt(reduce_sum(local_norm_sq));

        // 3. Проверка критерия остановки
        if (global_res_norm / b_norm < eps) {
            converged = true;
        }

        // 4. Синхронизация: обмениваемся вычисленными частями вектора x
        synchronize_vector(x_next_local, x, p);

        iter++;
        if (iter > 50000) break; // Защита от бесконечного цикла
    }

    double end_time = MPI_Wtime();
    if (rank == 0) {
        std::cout << "[Variant 1] Finished. Iterations: " << iter
                  << ", Time: " << end_time - start_time << "s" << std::endl;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    solve_variant_1(25000, 0.00001, 1e-5);

    MPI_Finalize();
    return 0;
}
