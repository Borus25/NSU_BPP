#include <iostream>
#include <vector>
#include <cmath>
#include <mpi.h>

// --- МОДУЛЬ 1: РАСПРЕДЕЛЕНИЕ ДАННЫХ ---
struct Partition {
    std::vector<int> counts;
    std::vector<int> displs;
    int local_n;
    int start_row;

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

// --- МОДУЛЬ 2: ИНИЦИАЛИЗАЦИЯ ---
void init_system(size_t N, const Partition& p, std::vector<double>& A_local, std::vector<double>& b_local) {
    for (size_t i = 0; i < (size_t)p.local_n; ++i) {
        size_t global_i = (size_t)p.start_row + i;
        for (size_t j = 0; j < N; ++j) {
            A_local[i * N + j] = (global_i == j) ? 2.0 : 1.0;
        }
        b_local[i] = (double)N + 1.0;
    }
}

// --- МОДУЛЬ 3: КОММУНИКАЦИИ (MPI) ---
// Сборка распределенного вектора в один полный буфер
void gather_full_vector_ptp(const std::vector<double>& x_local, std::vector<double>& x_full, const Partition& p, int rank, int size) {
    // 1. Сначала копируем свой собственный кусок в итоговый вектор
    for (int i = 0; i < p.local_n; ++i) {
        x_full[p.displs[rank] + i] = x_local[i];
    }

    int send_to = (rank + 1) % size;        // Сосед справа
    int recv_from = (rank - 1 + size) % size; // Сосед слева

    // Текущий кусок, который мы будем пересылать
    // На первом шаге это наш родной кусок
    std::vector<double> send_buffer = x_local;

    // Индекс процесса, чей кусок мы сейчас держим в send_buffer
    int current_chunk_rank = rank;

    for (int step = 1; step < size; ++step) {
        // Определяем размер куска, который будем принимать
        int prev_chunk_rank = (current_chunk_rank - 1 + size) % size;
        std::vector<double> recv_buffer(p.counts[prev_chunk_rank]);

        // Одновременно отправляем текущий кусок и принимаем следующий
        MPI_Sendrecv(send_buffer.data(), send_buffer.size(), MPI_DOUBLE, send_to, 0,
                     recv_buffer.data(), recv_buffer.size(), MPI_DOUBLE, recv_from, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Сохраняем полученный кусок в x_full
        for (int i = 0; i < p.counts[prev_chunk_rank]; ++i) {
            x_full[p.displs[prev_chunk_rank] + i] = recv_buffer[i];
        }

        // Готовимся к следующему шагу: то, что приняли, станет тем, что отправим
        send_buffer = recv_buffer;
        current_chunk_rank = prev_chunk_rank;
    }
}

// Глобальное суммирование (для норм и невязок)
double get_global_sum(double local_val) {
    double global_val;
    MPI_Allreduce(&local_val, &global_val, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return global_val;
}

// --- МОДУЛЬ 4: МАТЕМАТИЧЕСКОЕ ЯДРО ---
// Вычисление локальной части: x_next = x - tau * (Ax - b)
double compute_local_step(size_t N, const Partition& p,
                          const std::vector<double>& A_local, const std::vector<double>& b_local,
                          const std::vector<double>& x_full, std::vector<double>& x_next_local,
                          const std::vector<double>& x_local, double tau) {
    double local_res_norm_sq = 0;
    for (size_t i = 0; i < (size_t)p.local_n; ++i) {
        double Ax_i = 0;
        for (size_t j = 0; j < N; ++j) {
            Ax_i += A_local[i * N + j] * x_full[j];
        }
        double residual = Ax_i - b_local[i];
        local_res_norm_sq += residual * residual;
        x_next_local[i] = x_local[i] - tau * residual;
    }
    return local_res_norm_sq;
}

// --- МОДУЛЬ 5: ОРКЕСТРАТОР (SOLVER) ---
void solve(int N, double tau, double eps) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Partition p(N, size, rank);

    std::vector<double> A_local((size_t)p.local_n * N);
    std::vector<double> b_local(p.local_n);
    std::vector<double> x_local(p.local_n, 0.0);
    std::vector<double> x_next_local(p.local_n);
    std::vector<double> x_full(N);

    init_system(N, p, A_local, b_local);

    // Считаем норму b один раз
    double local_b_sq = 0;
    for(double val : b_local) local_b_sq += val * val;
    double b_norm = std::sqrt(get_global_sum(local_b_sq));

    int iter = 0;
    bool converged = false;
    double start_time = MPI_Wtime();

    while (!converged) {
        iter++;

        // --- ШАГ 1: Синхронизация вектора X через Point-to-Point ---
        // Вместо MPI_Allgatherv используем нашу новую функцию
        gather_full_vector_ptp(x_local, x_full, p, rank, size);

        // --- ШАГ 2: Вычисление следующего приближения ---
        // Каждый процесс считает только свои local_n строк
        double local_norm_sq = 0.0;
        for (int i = 0; i < p.local_n; ++i) {
            double Ax_i = 0.0;
            for (int j = 0; j < N; ++j) {
                Ax_i += A_local[i * N + j] * x_full[j];
            }

            double r_i = Ax_i - b_local[i]; // Компонента невязки
            local_norm_sq += r_i * r_i;

            // Вычисляем x(n+1)
            x_next_local[i] = x_local[i] - tau * r_i;
        }

        // --- ШАГ 3: Проверка сходимости ---
        double global_res_norm_sq = 0.0;
        MPI_Allreduce(&local_norm_sq, &global_res_norm_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double global_res_norm = std::sqrt(global_res_norm_sq);

        if (global_res_norm / b_norm < eps) {
            converged = true;
        } else {
            x_local = x_next_local; // Обновляем локальный x для следующей итерации
        }

        // Опционально: печать прогресса только на нулевом процессе
        if (rank == 0 && iter % 100 == 0) {
            std::cout << "Iteration " << iter << ", Residual: " << global_res_norm / b_norm << std::endl;
        }

        if (iter > 10000) break; // Защита от бесконечного цикла
    }

    double end_time = MPI_Wtime();
    if (rank == 0) {
        std::cout << "[Variant 2] finished. Iters: " << iter << ", Time: " << end_time - start_time << "s" << std::endl;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    solve(25000, 0.00001, 1e-5);
    MPI_Finalize();
    return 0;
}
