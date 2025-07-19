#include <iostream>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <stdint.h>
#include <chrono>
#include <kmcuda.h>

using namespace std;
using namespace std::chrono;

int main(int argc, char *argv[])
{
    srand(time(NULL));

    int total_points, total_values, K, max_iterations, has_name;
    cin >> total_points >> total_values >> K >> max_iterations >> has_name;

    vector<float> data(total_points * total_values);
    vector<string> names;
    string point_name;

    for (int i = 0; i < total_points; i++)
    {
        for (int j = 0; j < total_values; j++)
        {
            cin >> data[i * total_values + j];
        }

        if (has_name)
        {
            cin >> point_name;
            names.push_back(point_name);
        }
    }

    cout << "K,AverageTimeMicroseconds" << endl;
    int k_vals[] = {2, 3, 5, 10, 20};

    for (int K : k_vals)
    {
        long long total_time = 0;
        int numRuns = 25;
        
        for (int r = 0; r < numRuns; r++)
        {
            vector<float> centroids(K * total_values, 0.0f);
            vector<uint32_t> assignments(total_points);

            float average_distance;

            auto begin = high_resolution_clock::now();

            KMCUDAResult result = kmeans_cuda(
                kmcudaInitMethodPlusPlus,   // KMeans++ initialization
                NULL,                       // No predefined centroids
                0.01,                        // Convergence threshold
                0.1,                         // Yinyang refinement threshold
                kmcudaDistanceMetricL2,      // Use Euclidean distance
                total_points, total_values, K,
                0xDEADBEEF,                  // Random seed
                0,                           // Use all available CUDA devices
                -1,                          // Data is stored on host
                0,                           // No float16 mode
                0,                           // Verbose mode
                data.data(), centroids.data(), assignments.data(), &average_distance
            );

            auto end = high_resolution_clock::now();
            long long duration = duration_cast<microseconds>(end - begin).count();
            total_time += duration;

            if (result != kmcudaSuccess)
            {
                cerr << "KMCUDA failed for K=" << K << endl;
                return -1;
            }
        }

        long long avg_time = total_time / numRuns;
        cout << K << "," << avg_time << endl;
    }

    return 0;
}