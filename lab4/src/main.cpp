#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>

using namespace std;

// Точное решение: φ(x, y, z) = x^2 + y^2 + z^2
double exact_solution(double x, double y, double z) {
    return x * x + y * y + z * z;
}

// Правая часть уравнения: ρ(x, y, z) = 6 - a * φ(x, y, z)
double right_hand_side(double x, double y, double z, double a) {
    return 6.0 - a * exact_solution(x, y, z);
}

// Функция для вычисления коэффициентов разностной схемы
struct Coefficients {
    double inv_hx2, inv_hy2, inv_hz2;
    double denominator;

    Coefficients(double hx, double hy, double hz, double a) {
        inv_hx2 = 1.0 / (hx * hx);
        inv_hy2 = 1.0 / (hy * hy);
        inv_hz2 = 1.0 / (hz * hz);
        denominator = 2.0 * (inv_hx2 + inv_hy2 + inv_hz2) + a;
    }
};

// Класс для хранения трехмерного поля с поддержкой теневых граней
class Field3D {
private:
    vector<double> data;
    int nx, ny, nz_local;  // локальные размеры с учетом теневых граней
    int nx_real, ny_real, nz_real;  // реальные размеры без теневых граней
    int ghost_layers;

public:
    Field3D(int nx, int ny, int nz, int ghost = 1)
        : nx(nx), ny(ny), nz_local(nz + 2 * ghost),
          nx_real(nx), ny_real(ny), nz_real(nz),
          ghost_layers(ghost) {
        data.resize(nx * ny * (nz + 2 * ghost), 0.0);
    }

    // Индексация: (x, y, z) с учетом теневых граней
    int index(int i, int j, int k) const {
        return ((k + ghost_layers) * ny + j) * nx + i;
    }

    // Доступ к элементу по локальным координатам (с учетом теневых граней)
    double& operator()(int i, int j, int k) {
        return data[index(i, j, k)];
    }

    const double& operator()(int i, int j, int k) const {
        return data[index(i, j, k)];
    }

    // Доступ к элементу по глобальным координатам (без теневых граней)
    double& at_global(int i, int j, int k) {
        return data[((k + ghost_layers) * ny + j) * nx + i];
    }

    const double& at_global(int i, int j, int k) const {
        return data[((k + ghost_layers) * ny + j) * nx + i];
    }

    // Получение указателя на слой по Z (для MPI коммуникаций)
    double* get_layer_ptr(int k) {
        return data.data() + ((k + ghost_layers) * ny) * nx;
    }

    const double* get_layer_ptr(int k) const {
        return data.data() + ((k + ghost_layers) * ny) * nx;
    }

    // Обнуление поля
    void zero() {
        fill(data.begin(), data.end(), 0.0);
    }

    // Заполнение граничных значений (краевые условия 1-го рода)
    void set_boundary(double x0, double y0, double z0,
                      double hx, double hy, double hz) {
        // Грани по X: i = 0 и i = nx-1
        for (int j = 0; j < ny; j++) {
            for (int k = -ghost_layers; k < nz_real + ghost_layers; k++) {
                double z = z0 + (k + ghost_layers) * hz;
                if (k >= 0 && k < nz_real) {
                    double y = y0 + j * hy;
                    at_global(0, j, k) = exact_solution(x0, y, z);
                    at_global(nx - 1, j, k) = exact_solution(x0 + (nx - 1) * hx, y, z);
                }
            }
        }

        // Грани по Y: j = 0 и j = ny-1
        for (int i = 0; i < nx; i++) {
            for (int k = -ghost_layers; k < nz_real + ghost_layers; k++) {
                double z = z0 + (k + ghost_layers) * hz;
                if (k >= 0 && k < nz_real) {
                    double x = x0 + i * hx;
                    at_global(i, 0, k) = exact_solution(x, y0, z);
                    at_global(i, ny - 1, k) = exact_solution(x, y0 + (ny - 1) * hy, z);
                }
            }
        }

        // Грани по Z: k = 0 и k = nz_real-1 (обрабатываются в каждом процессе)
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                double x = x0 + i * hx;
                double y = y0 + j * hy;
                at_global(i, j, 0) = exact_solution(x, y, z0);
                at_global(i, j, nz_real - 1) = exact_solution(x, y, z0 + (nz_real - 1) * hz);
            }
        }
    }

    // Вычисление максимальной разницы между двумя полями (для проверки сходимости)
    double max_diff(const Field3D& other) const {
        double max_diff_val = 0.0;
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                for (int k = 0; k < nz_real; k++) {
                    double diff = fabs(at_global(i, j, k) - other.at_global(i, j, k));
                    if (diff > max_diff_val) max_diff_val = diff;
                }
            }
        }
        return max_diff_val;
    }

    // Копирование значений из другого поля (только внутренняя область)
    void copy_from(const Field3D& other) {
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                for (int k = 0; k < nz_real; k++) {
                    at_global(i, j, k) = other.at_global(i, j, k);
                }
            }
        }
    }
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Параметры области моделирования
    const double x0 = -1.0, y0 = -1.0, z0 = -1.0;
    const double Dx = 2.0, Dy = 2.0, Dz = 2.0;
    const double a = 1e5;  // параметр уравнения
    const double epsilon = 1e-8;  // порог сходимости

    // Параметры сетки
    int Nx, Ny, Nz;

    // Чтение параметров из командной строки
    if (argc != 4) {
        if (world_rank == 0) {
            cerr << "Usage: " << argv[0] << " <Nx> <Ny> <Nz>" << endl;
            cerr << "Example: " << argv[0] << " 100 100 100" << endl;
        }
        MPI_Finalize();
        return 1;
    }

    Nx = atoi(argv[1]);
    Ny = atoi(argv[2]);
    Nz = atoi(argv[3]);

    if (Nz < world_size) {
        if (world_rank == 0) {
            cerr << "Error: Nz (" << Nz << ") must be >= number of processes ("
                 << world_size << ")" << endl;
        }
        MPI_Finalize();
        return 1;
    }

    // Шаги сетки
    double hx = Dx / (Nx - 1);
    double hy = Dy / (Ny - 1);
    double hz = Dz / (Nz - 1);

    // Коэффициенты разностной схемы
    Coefficients coeff(hx, hy, hz, a);

    // Распределение слоев по процессам (одномерная декомпозиция по Z)
    int layers_per_process = Nz / world_size;
    int remainder = Nz % world_size;

    int local_nz = layers_per_process + (world_rank < remainder ? 1 : 0);
    int start_z = world_rank * layers_per_process + min(world_rank, remainder);
    int end_z = start_z + local_nz - 1;

    // Создание полей с тенеными гранями (по одной с каждой стороны по Z)
    Field3D phi(Nx, Ny, local_nz, 1);
    Field3D phi_new(Nx, Ny, local_nz, 1);
    Field3D phi_prev(Nx, Ny, local_nz, 1);

    // Инициализация: начальное приближение φ = 0
    phi.zero();
    phi_new.zero();

    // Заполнение граничных значений (краевые условия 1-го рода)
    // Учитываем, что граничные слои могут быть распределены по процессам
    double global_z0 = z0;
    phi.set_boundary(x0, y0, global_z0, hx, hy, hz);

    // Сохранение начального приближения для проверки сходимости
    phi_prev.copy_from(phi);

    // Соседние процессы по оси Z
    int prev_rank = (world_rank - 1 + world_size) % world_size;
    int next_rank = (world_rank + 1) % world_size;

    // MPI запросы для асинхронных коммуникаций
    MPI_Request send_up_req, send_down_req, recv_up_req, recv_down_req;
    MPI_Status status;

    double start_time = MPI_Wtime();

    int iteration = 0;
    double max_diff_global = 1.0;

    // Основной итерационный цикл
    while (max_diff_global > epsilon && iteration < 1000000) {
        iteration++;

        // 1. Отправка и прием граничных слоев
        // Отправляем нижнюю границу предыдущему процессу (верхняя теневая грань)
        MPI_Isend(phi.get_layer_ptr(0), Nx * Ny, MPI_DOUBLE,
                  prev_rank, 0, MPI_COMM_WORLD, &send_up_req);

        // Отправляем верхнюю границу следующему процессу (нижняя теневая грань)
        MPI_Isend(phi.get_layer_ptr(local_nz - 1), Nx * Ny, MPI_DOUBLE,
                  next_rank, 1, MPI_COMM_WORLD, &send_down_req);

        // Получаем от предыдущего процесса его верхнюю границу
        MPI_Irecv(phi.get_layer_ptr(-1), Nx * Ny, MPI_DOUBLE,
                  prev_rank, 1, MPI_COMM_WORLD, &recv_up_req);

        // Получаем от следующего процесса его нижнюю границу
        MPI_Irecv(phi.get_layer_ptr(local_nz), Nx * Ny, MPI_DOUBLE,
                  next_rank, 0, MPI_COMM_WORLD, &recv_down_req);

        // 2. Вычисление внутренних слоев (не требующих теневых граней)
        // Внутренние слои: от 1 до local_nz-2 (0-индексация, исключая граничные)
        for (int k = 1; k <= local_nz - 2; k++) {
            for (int j = 0; j < Ny; j++) {
                for (int i = 0; i < Nx; i++) {
                    // Пропускаем граничные узлы по X и Y
                    if (i == 0 || i == Nx - 1 || j == 0 || j == Ny - 1) continue;

                    double x = x0 + i * hx;
                    double y = y0 + j * hy;
                    double z = global_z0 + (start_z + k) * hz;

                    double laplacian =
                        coeff.inv_hx2 * (phi(i + 1, j, k) + phi(i - 1, j, k)) +
                        coeff.inv_hy2 * (phi(i, j + 1, k) + phi(i, j - 1, k)) +
                        coeff.inv_hz2 * (phi(i, j, k + 1) + phi(i, j, k - 1));

                    phi_new(i, j, k) = (laplacian - right_hand_side(x, y, z, a)) / coeff.denominator;
                }
            }
        }

        // 3. Дожидаемся получения верхней теневой грани
        MPI_Wait(&recv_up_req, &status);

        // 4. Вычисление верхнего граничного слоя (k = 0) - требует верхнюю теневую грань
        int k = 0;
        if (start_z > 0) {  // Не глобальная граница
            for (int j = 0; j < Ny; j++) {
                for (int i = 0; i < Nx; i++) {
                    if (i == 0 || i == Nx - 1 || j == 0 || j == Ny - 1) continue;

                    double x = x0 + i * hx;
                    double y = y0 + j * hy;
                    double z = global_z0 + (start_z + k) * hz;

                    double laplacian =
                        coeff.inv_hx2 * (phi(i + 1, j, k) + phi(i - 1, j, k)) +
                        coeff.inv_hy2 * (phi(i, j + 1, k) + phi(i, j - 1, k)) +
                        coeff.inv_hz2 * (phi(i, j, k + 1) + phi.get_layer_ptr(-1)[j * Nx + i]);

                    phi_new(i, j, k) = (laplacian - right_hand_side(x, y, z, a)) / coeff.denominator;
                }
            }
        } else {
            // Глобальная граница по Z: значение фиксировано
            for (int j = 0; j < Ny; j++) {
                for (int i = 0; i < Nx; i++) {
                    if (i == 0 || i == Nx - 1 || j == 0 || j == Ny - 1) continue;
                    double x = x0 + i * hx;
                    double y = y0 + j * hy;
                    phi_new(i, j, k) = exact_solution(x, y, global_z0);
                }
            }
        }

        // 5. Дожидаемся завершения отправки верхней границы
        MPI_Wait(&send_up_req, &status);

        // 6. Дожидаемся получения нижней теневой грани
        MPI_Wait(&recv_down_req, &status);

        // 7. Вычисление нижнего граничного слоя (k = local_nz - 1) - требует нижнюю теневую грань
        k = local_nz - 1;
        if (start_z + local_nz - 1 < Nz - 1) {  // Не глобальная граница
            for (int j = 0; j < Ny; j++) {
                for (int i = 0; i < Nx; i++) {
                    if (i == 0 || i == Nx - 1 || j == 0 || j == Ny - 1) continue;

                    double x = x0 + i * hx;
                    double y = y0 + j * hy;
                    double z = global_z0 + (start_z + k) * hz;

                    double laplacian =
                        coeff.inv_hx2 * (phi(i + 1, j, k) + phi(i - 1, j, k)) +
                        coeff.inv_hy2 * (phi(i, j + 1, k) + phi(i, j - 1, k)) +
                        coeff.inv_hz2 * (phi.get_layer_ptr(local_nz)[j * Nx + i] + phi(i, j, k - 1));

                    phi_new(i, j, k) = (laplacian - right_hand_side(x, y, z, a)) / coeff.denominator;
                }
            }
        } else {
            // Глобальная граница по Z: значение фиксировано
            for (int j = 0; j < Ny; j++) {
                for (int i = 0; i < Nx; i++) {
                    if (i == 0 || i == Nx - 1 || j == 0 || j == Ny - 1) continue;
                    double x = x0 + i * hx;
                    double y = y0 + j * hy;
                    phi_new(i, j, k) = exact_solution(x, y, global_z0 + (Nz - 1) * hz);
                }
            }
        }

        // 8. Дожидаемся завершения отправки нижней границы
        MPI_Wait(&send_down_req, &status);

        // 9. Вычисление локальной максимальной разницы
        double local_max_diff = phi_prev.max_diff(phi_new);

        // 10. Глобальная редукция для получения максимальной разницы по всем процессам
        MPI_Allreduce(&local_max_diff, &max_diff_global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        // 11. Обновление полей для следующей итерации
        phi_prev.copy_from(phi_new);
        phi.copy_from(phi_new);

        // Отладочный вывод
        if (world_rank == 0 && iteration % 100 == 0) {
            cout << "Iteration " << iteration << ", max diff = " << max_diff_global << endl;
        }
    }

    double end_time = MPI_Wtime();
    double elapsed_time = end_time - start_time;

    // Вычисление ошибки решения (сравнение с точным)
    double local_error = 0.0;
    double global_error = 0.0;

    for (int k = 0; k < local_nz; k++) {
        for (int j = 0; j < Ny; j++) {
            for (int i = 0; i < Nx; i++) {
                double x = x0 + i * hx;
                double y = y0 + j * hy;
                double z = global_z0 + (start_z + k) * hz;
                double exact = exact_solution(x, y, z);
                double error = fabs(phi(i, j, k) - exact);
                if (error > local_error) local_error = error;
            }
        }
    }

    MPI_Reduce(&local_error, &global_error, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Вывод результатов
    if (world_rank == 0) {
        cout << "Grid size: " << Nx << " x " << Ny << " x " << Nz << endl;
        cout << "Number of MPI processes: " << world_size << endl;
        cout << "Iterations completed: " << iteration << endl;
        cout << "Execution time: " << elapsed_time << " seconds" << endl;
        cout << "Final max diff: " << max_diff_global << endl;
        cout << "Max error vs exact solution: " << global_error << endl;

        // Расчет эффективности (требуется время на 1 процессе)
        cout << "\nNote: For speedup calculation, run with 1 process separately." << endl;
    }

    MPI_Finalize();
    return 0;
}