#include <iostream>
#include <vector>
#include <cmath>
#include <mutex>
#include <thread>
#include <atomic>
#include <algorithm>
#include <chrono>
#include <mpi.h>

std::vector<int> local_tasks;
std::mutex task_mutex;
std::atomic<bool> terminate_listener(false);
bool balance_load = true;
const int WORK_REQUEST_TAG = 0;
const int STOP_TAG = 1;
const int WORK_REPLY_TAG = 2;

void execute_task(int weight, long long& completed_weight) {
    double res = 0.0;
    for (int i = 0; i < weight; i++) {
        res += std::sqrt(i);
    }
    completed_weight += weight;
}

void listener_thread() {
    while (!terminate_listener.load()) {
        int flag = 0;
        MPI_Status status;
        MPI_Iprobe(MPI_ANY_SOURCE, WORK_REQUEST_TAG, MPI_COMM_WORLD, &flag, &status);

        if (!flag) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
            continue;
        }

        int request = 0;
        MPI_Recv(&request, 1, MPI_INT, status.MPI_SOURCE, WORK_REQUEST_TAG, MPI_COMM_WORLD, &status);

        std::vector<int> chunk;
        {
            std::lock_guard<std::mutex> lock(task_mutex);
            if (local_tasks.size() > 2) {
                int chunk_size = static_cast<int>(local_tasks.size()) / 2;
                for (int i = 0; i < chunk_size; i++) {
                    chunk.push_back(local_tasks.back());
                    local_tasks.pop_back();
                }
            }
        }

        int send_size = static_cast<int>(chunk.size());
        MPI_Send(&send_size, 1, MPI_INT, status.MPI_SOURCE, WORK_REPLY_TAG, MPI_COMM_WORLD);

        if (send_size > 0) {
            MPI_Send(chunk.data(), send_size, MPI_INT, status.MPI_SOURCE, WORK_REPLY_TAG, MPI_COMM_WORLD);
        }
    }
}

int main(int argc, char** argv) {
    if (argc > 1) {
        balance_load = std::atoi(argv[1]) != 0;
    }

    int required = balance_load ? MPI_THREAD_MULTIPLE : MPI_THREAD_SINGLE;
    int provided = 0;
    MPI_Init_thread(&argc, &argv, required, &provided);

    if (provided < required) {
        std::cerr << "MPI threading level unsupported. Provided: " << provided << ", Required: " << required << "\n";
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        std::cout << "Requested thread level = " << required << ", provided = " << provided << "\n";
    }

    if (size == 1) {
        balance_load = false;
    }

    std::thread listener;
    if (balance_load) {
        listener = std::thread(listener_thread);
    }

    const int num_iters = 5;
    const int total_tasks = 5000;
    double total_time = 0.0;
    double total_lif = 0.0;

    for (int iter = 0; iter < num_iters; iter++) {
        std::vector<int> global_tasks(total_tasks);

        if (rank == 0) {
            int peak_index = (iter * (total_tasks / num_iters)) % total_tasks;
            for (int i = 0; i < total_tasks; i++) {
                int dist = std::abs(i - peak_index);
                dist = std::min(dist, total_tasks - dist);
                global_tasks[i] = 10000 + 1000 * ((total_tasks / 2) - dist);
            }
        }

        MPI_Bcast(global_tasks.data(), total_tasks, MPI_INT, 0, MPI_COMM_WORLD);

        {
            std::lock_guard<std::mutex> lock(task_mutex);
            local_tasks.clear();

            int start = rank * total_tasks / size;
            int end = (rank + 1) * total_tasks / size;
            for (int i = start; i < end; i++) {
                local_tasks.push_back(global_tasks[i]);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double start_time = MPI_Wtime();
        long long completed_weight = 0;

        while (true) {
            int task = 0;
            {
                std::lock_guard<std::mutex> lock(task_mutex);
                if (!local_tasks.empty()) {
                    task = local_tasks.back();
                    local_tasks.pop_back();
                }
            }

            if (task > 0) {
                execute_task(task, completed_weight);
                continue;
            }

            if (!balance_load) {
                break;
            }

            bool received_work = false;
            for (int p = 1; p < size; p++) {
                int target = (rank + p) % size;
                int req = 1;

                MPI_Send(&req, 1, MPI_INT, target, WORK_REQUEST_TAG, MPI_COMM_WORLD);

                int recv_size = 0;
                MPI_Recv(&recv_size, 1, MPI_INT, target, WORK_REPLY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                if (recv_size <= 0) {
                    continue;
                }

                std::vector<int> recv_tasks(recv_size);
                MPI_Recv(recv_tasks.data(), recv_size, MPI_INT, target, WORK_REPLY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                {
                    std::lock_guard<std::mutex> lock(task_mutex);
                    for (int t : recv_tasks) {
                        local_tasks.push_back(t);
                    }
                }
                received_work = true;
                break;
            }

            if (!received_work) {
                break;
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double end_time = MPI_Wtime();
        double iter_time = end_time - start_time;

        long long max_weight = 0, sum_weight = 0;
        double max_time = 0.0;

        MPI_Reduce(&completed_weight, &max_weight, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&completed_weight, &sum_weight, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&iter_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            double avg_weight = static_cast<double>(sum_weight) / size;
            double lif = max_weight / avg_weight;

            total_lif += lif;
            total_time += max_time;

            std::cout << "Iter " << iter << " | Max Time: " << max_time << " s | LIF: " << lif << "\n";
        }
    }

    if (balance_load) {
        terminate_listener = true;
        listener.join();
    }

    if (rank == 0) {
        std::cout << "----------------------------------\n";
        std::cout << "Balancing: " << (balance_load ? "ON" : "OFF") << "\n";
        std::cout << "Total Time: " << total_time << " s\n";
        std::cout << "Average LIF: " << (total_lif / num_iters) << "\n";
    }

    MPI_Finalize();
    return 0;
}