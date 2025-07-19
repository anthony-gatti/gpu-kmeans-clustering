#include <iostream>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;
using namespace std::chrono;

struct Point {
    int id_cluster;
    double* values;
    std::string name;
};

struct Cluster {
    double* central_values;
};

__global__ void assignClusters(
    double *point_values,
    double *cluster_values,
    int *assignments,
    int total_points,
    int K,
    int total_values,
    int *changed_flag
) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= total_points) return;

    double min_dist = INFINITY;
    int best_cluster = -1;

    for (int c = 0; c < K; c++) {
        double sum = 0.0;
        for (int j = 0; j < total_values; j++) {
            double diff = cluster_values[c * total_values + j] - point_values[idx * total_values + j];
            sum += diff * diff;
        }
        if (sum < min_dist) {
            min_dist = sum;
            best_cluster = c;
        }
    }

    if (assignments[idx] != best_cluster) {
        assignments[idx] = best_cluster;
        atomicExch(changed_flag, 1);
    }
}

__global__ void updateCentroids(
    double *point_values,
    double *cluster_values,
    int *assignments,
    int *cluster_sizes,
    int total_points,
    int K,
    int total_values
) {
    int c = blockIdx.x;
    if (c >= K) return;

    extern __shared__ double shared_sums[];
    
    for (int j = threadIdx.x; j < total_values; j += blockDim.x) {
        shared_sums[j] = 0.0;
    }
    __syncthreads();

    int local_count = 0;
    for (int i = threadIdx.x; i < total_points; i += blockDim.x) {
        if (assignments[i] == c) {
            local_count++;
            for (int j = 0; j < total_values; j++) {
                atomicAdd(&shared_sums[j], point_values[i * total_values + j]);
            }
        }
    }

    atomicAdd(&cluster_sizes[c], local_count);
    __syncthreads();

    if (threadIdx.x == 0) {
        int count = cluster_sizes[c];
        if (count > 0) {
            for (int j = 0; j < total_values; j++) {
                cluster_values[c * total_values + j] = shared_sums[j] / count;
            }
        } else {
            for (int j = 0; j < total_values; j++) {
                cluster_values[c * total_values + j] = 0.0;
            }
        }
    }
}

long long kmeansCUDA(Point *h_points, Cluster *h_clusters, int total_points, int K, int total_values, int max_iterations) {
    auto begin = high_resolution_clock::now();

    double *d_point_values, *d_cluster_values;
    int *d_assignments, *d_cluster_sizes, *d_changed_flag;

    cudaMalloc(&d_point_values, total_points * total_values * sizeof(double));
    cudaMalloc(&d_cluster_values, K * total_values * sizeof(double));
    cudaMalloc(&d_assignments, total_points * sizeof(int));
    cudaMalloc(&d_cluster_sizes, K * sizeof(int));
    cudaMalloc(&d_changed_flag, sizeof(int));

    // copies points into device memory
    for (int i = 0; i < total_points; i++) {
        cudaMemcpy(d_point_values + i * total_values,
                              h_points[i].values,
                              total_values * sizeof(double),
                              cudaMemcpyHostToDevice);
    }
    // copies initial centroids into device memory
    for (int i = 0; i < K; i++) {
        cudaMemcpy(d_cluster_values + i * total_values,
                              h_clusters[i].central_values,
                              total_values * sizeof(double),
                              cudaMemcpyHostToDevice);
    }
    // initialize assignments to -1.
    cudaMemset(d_assignments, -1, total_points * sizeof(int));

    int threads = 256;
    int blocks_points = (total_points + threads - 1) / threads;

    auto end_phase1 = high_resolution_clock::now();

    int h_changed_flag = 0;
    int iter = 0;
    do {
        iter++;
        h_changed_flag = 0;
        cudaMemset(d_changed_flag, 0, sizeof(int));

        assignClusters<<<blocks_points, threads>>>(d_point_values, d_cluster_values, d_assignments,
                                                     total_points, K, total_values, d_changed_flag);
        cudaGetLastError();
        cudaDeviceSynchronize();

        cudaMemcpy(&h_changed_flag, d_changed_flag, sizeof(int), cudaMemcpyDeviceToHost);

        cudaMemset(d_cluster_sizes, 0, K * sizeof(int));

        updateCentroids<<<K, threads, total_values * sizeof(double)>>>(d_point_values, d_cluster_values,
                                                                         d_assignments, d_cluster_sizes,
                                                                         total_points, K, total_values);
        cudaGetLastError();
        cudaDeviceSynchronize();
    } while (h_changed_flag && iter < max_iterations);

    auto end = high_resolution_clock::now();

    int *assignments_host = new int[total_points];
    cudaMemcpy(assignments_host, d_assignments, total_points * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < total_points; i++) {
        h_points[i].id_cluster = assignments_host[i];
    }
    delete[] assignments_host;

    for (int i = 0; i < K; i++) {
        cudaMemcpy(h_clusters[i].central_values,
                              d_cluster_values + i * total_values,
                              total_values * sizeof(double),
                              cudaMemcpyDeviceToHost);
    }

    long long duration = duration_cast<microseconds>(end - begin).count();

    // cout << "--------------------------------------------------" << endl;
    // for (int i = 0; i < K; i++) {
    //     cout << "Cluster " << i + 1 << endl;
    //     for (int j = 0; j < total_points; j++) {
    //         if (h_points[j].id_cluster == i) {
    //             cout << "Point " << j + 1 << ": ";
    //             for (int p = 0; p < total_values; p++) {
    //                 cout << h_points[j].values[p] << " ";
    //             }
    //             cout << endl;
    //         }
    //     }
    //     cout << "Cluster values: ";
    //     for (int j = 0; j < total_values; j++) {
    //         cout << h_clusters[i].central_values[j] << " ";
    //     }
    //     cout << "\n\n";
    // }
    // cout << "TOTAL EXECUTION TIME = " << duration << " microseconds" << endl;
    // cout << "TIME PHASE 1 = " << duration_cast<microseconds>(end_phase1 - begin).count() << " microseconds" << endl;
    // cout << "TIME PHASE 2 = " << duration_cast<microseconds>(end - end_phase1).count() << " microseconds" << endl;
    // cout << "--------------------------------------------------" << endl;

    cudaFree(d_point_values);
    cudaFree(d_cluster_values);
    cudaFree(d_assignments);
    cudaFree(d_cluster_sizes);
    cudaFree(d_changed_flag);

    return duration;
}

int main(int argc, char *argv[]) {
    srand(10);

    int total_points, total_values, K, max_iterations, has_name;
    cin >> total_points >> total_values >> K >> max_iterations >> has_name;

    Point *points = new Point[total_points];
    for (int i = 0; i < total_points; i++) {
        points[i].values = new double[total_values];
        for (int j = 0; j < total_values; j++) {
            cin >> points[i].values[j];
        }
        if (has_name) {
            cin >> points[i].name;
        }
        points[i].id_cluster = -1;
    }

    cout << "K,AverageTimeMicroseconds" << endl;

    int k_vals[] = {2, 3, 5, 10, 20};
    int num_runs = 25;
    for (int k_val : k_vals) {
        long long total_time = 0;
        for (int r = 0; r < num_runs; r++) {
            Cluster *clusters = new Cluster[k_val];
            for (int i = 0; i < k_val; i++) {
                clusters[i].central_values = new double[total_values];
            }

            int *chosen = new int[k_val];
            for (int i = 0; i < k_val; i++) {
                while (true) {
                    int idx = rand() % total_points;
                    bool duplicate = false;
                    for (int j = 0; j < i; j++) {
                        if (chosen[j] == idx) { duplicate = true; break; }
                    }
                    if (!duplicate) {
                        chosen[i] = idx;
                        break;
                    }
                }
            }

            for (int i = 0; i < k_val; i++) {
                for (int j = 0; j < total_values; j++) {
                    clusters[i].central_values[j] = points[chosen[i]].values[j];
                }
            }
            delete[] chosen;

            long long run_time = kmeansCUDA(points, clusters, total_points, k_val, total_values, max_iterations);
            total_time += run_time;

            for (int i = 0; i < k_val; i++) {
                delete[] clusters[i].central_values;
            }
            delete[] clusters;
        }
        long long avg_time = total_time / num_runs;
        cout << k_val << "," << avg_time << endl;
    }

    for (int i = 0; i < total_points; i++) {
        delete[] points[i].values;
    }
    delete[] points;

    return 0;
}