// Implementation of the KMeans Algorithm
// reference: https://github.com/marcoscastro/kmeans

#include <iostream>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <chrono>
#include <sstream>
#ifdef _OPENACC
#include <openacc.h>
#endif

using namespace std;
using namespace std::chrono;

class Point
{
private:
	int id_point, id_cluster;
	vector<double> values;
	int total_values;
	string name;

public:
	Point(int id_point, vector<double> &values, string name = "")
	{
		this->id_point = id_point;
		total_values = values.size();

		for (int i = 0; i < total_values; i++)
			this->values.push_back(values[i]);

		this->name = name;
		id_cluster = -1;
	}

	int getID() const
	{
		return id_point;
	}

	void setCluster(int id_cluster)
	{
		this->id_cluster = id_cluster;
	}

	int getCluster() const
	{
		return id_cluster;
	}

	double getValue(int index) const
	{
		return values[index];
	}

	int getTotalValues() const
	{
		return total_values;
	}

	void addValue(double value)
	{
		values.push_back(value);
	}

	string getName() const
	{
		return name;
	}
};

class Cluster
{
private:
	int id_cluster;
	vector<double> central_values;

public:
	Cluster(int id_cluster, const Point *point)
	{
		this->id_cluster = id_cluster;

		int total_values = point->getTotalValues();

		for (int i = 0; i < total_values; i++)
			central_values.push_back(point->getValue(i));
	}

	double getCentralValue(int index) const
	{
		return central_values[index];
	}

	void setCentralValue(int index, double value)
	{
		central_values[index] = value;
	}

	int getID() const
	{
		return id_cluster;
	}
};

class KMeans
{
private:
	int K;
	int total_values, total_points, max_iterations;
	vector<Cluster> clusters;

	// return ID of nearest center
	int getIDNearestCenter(const Point &point)
	{
		double min_dist = INFINITY;
		int id_cluster_center = 0;

		#pragma acc parallel loop reduction(min:min_dist)
		for (int i = 0; i < K; i++) {
            double sum = 0.0;
            for (int j = 0; j < total_values; j++) {
                double diff = clusters[i].getCentralValue(j) - point.getValue(j);
                sum += diff * diff;
            }
            if (sum < min_dist) {
                min_dist = sum;
            }
        }

		for (int i = 0; i < K; i++) {
			double sum = 0.0;
			for (int j = 0; j < total_values; j++) {
				double diff = clusters[i].getCentralValue(j) - point.getValue(j);
				sum += diff * diff;
			}
			if (sum == min_dist) {
				id_cluster_center = i;
				break;
			}
		}
        return id_cluster_center;
	}

public:
	KMeans(int K, int total_points, int total_values, int max_iterations)
	{
		this->K = K;
		this->total_points = total_points;
		this->total_values = total_values;
		this->max_iterations = max_iterations;
	}

	long long run(vector<Point> &points)
	{
		auto begin = chrono::high_resolution_clock::now();

		if (K > total_points)
			return 0;

		vector<int> prohibited_indexes;

		// choose K distinct values for the centers of the clusters
		for (int i = 0; i < K; i++)
		{
			while (true)
			{
				int index_point = rand() % total_points;

				if (find(prohibited_indexes.begin(), prohibited_indexes.end(),
						 index_point) == prohibited_indexes.end())
				{
					prohibited_indexes.push_back(index_point);
					points[index_point].setCluster(i);
					Cluster cluster(i, &points[index_point]);
					clusters.push_back(cluster);
					break;
				}
			}
		}
		auto end_phase1 = chrono::high_resolution_clock::now();

		int iter = 1;

		while (true)
		{
			// associates each point to the nearest center
			int changed = 0;
			
			#pragma acc parallel loop reduction(+:changed)
			for (int i = 0; i < total_points; i++)
			{
				int id_old_cluster = points[i].getCluster();
				int id_nearest_center = getIDNearestCenter(points[i]);

				if (id_old_cluster != id_nearest_center)
				{
					points[i].setCluster(id_nearest_center);
					changed = 1;
				}
			}

			// recalculating the center of each cluster
			vector<vector<double>> cluster_values(K, vector<double>(total_values, 0.0));
			vector<int> cluster_points(K, 0);

			#pragma acc parallel loop
            for (int i = 0; i < total_points; i++) {
                int c = points[i].getCluster();
                if (c >= 0 && c < K) {
                    for (int j = 0; j < total_values; j++) {
                        #pragma acc atomic
                        cluster_values[c][j] += points[i].getValue(j);
                    }
                    #pragma acc atomic
                    cluster_points[c]++;
                }
            }

			#pragma acc parallel loop
            for (int c = 0; c < K; c++) {
                if (cluster_points[c] > 0) {
                    for (int j = 0; j < total_values; j++) {
                        clusters[c].setCentralValue(j, cluster_values[c][j] / cluster_points[c]);
                    }
                }
            }

			if (changed == 0 || iter >= max_iterations)
			{
				// cout << "Break in iteration " << iter << "\n\n";
				break;
			}

			iter++;
		}
		auto end = chrono::high_resolution_clock::now();
		long long duration = duration_cast<microseconds>(end - begin).count();

		cout << "--------------------------------------------------" << endl;
		// shows elements of clusters
		vector<vector<const Point*>> cluster_points(K);
		for (int i = 0; i < total_points; i++) {
			int c = points[i].getCluster();
			if (c >= 0 && c < K) {
				cluster_points[c].push_back(&points[i]);
			}
		}
	
		for (int c = 0; c < K; c++) {
			cout << "Cluster " << c + 1 << endl;
	
			for (const Point* pt : cluster_points[c]) {
				cout << "Point " << pt->getID() + 1 << ": ";
				for (int j = 0; j < total_values; j++) {
					cout << pt->getValue(j) << " ";
				}
				string name = pt->getName();
	
				if (!name.empty())
					cout << "- " << name;
				cout << endl;
			}
	
			cout << "Cluster values: ";
	
			for (int j = 0; j < total_values; j++) {
				cout << clusters[c].getCentralValue(j) << " ";
			}
			cout << "\n\n";
			cout << "TOTAL EXECUTION TIME = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "\n";
	
			cout << "TIME PHASE 1 = " << std::chrono::duration_cast<std::chrono::microseconds>(end_phase1 - begin).count() << "\n";
	
			cout << "TIME PHASE 2 = " << std::chrono::duration_cast<std::chrono::microseconds>(end - end_phase1).count() << "\n";
		}
		cout << "--------------------------------------------------" << endl;

		return duration;
		// return iter;
	}
};

int main(int argc, char *argv[])
{
	srand(10);

	int total_points, total_values, K, max_iterations, has_name;

	cin >> total_points >> total_values >> K >> max_iterations >> has_name;

	vector<Point> points;
	string point_name;

	for (int i = 0; i < total_points; i++)
	{
		vector<double> values;

		for (int j = 0; j < total_values; j++)
		{
			double value;
			cin >> value;
			values.push_back(value);
		}

		if (has_name)
		{
			cin >> point_name;
			Point p(i, values, point_name);
			points.push_back(p);
		}
		else
		{
			Point p(i, values);
			points.push_back(p);
		}
	}

	cout << "K,AverageTimeMicroseconds" << endl;
    int k_vals[] = {2, 3, 5, 10, 20};
    for (int K : k_vals) {
        long long total_time = 0;
        int numRuns = 25;
        for (int r = 0; r < numRuns; r++) {
            vector<Point> points_copy = points;
            KMeans kmeans(K, total_points, total_values, max_iterations);
            total_time += kmeans.run(points_copy);
        }
        long long avg_time = total_time / numRuns;
        cout << K << "," << avg_time << endl;
    }

	// cout << "K,AverageIterations" << endl;
	// int k_vals[] = {2, 3, 5, 10, 20};
	// for (int K : k_vals) {
	// 	long long total_iters = 0;
	// 	int numRuns = 100;
	// 	for (int r = 0; r < numRuns; r++) {
	// 		vector<Point> points_copy = points;
	// 		KMeans kmeans(K, total_points, total_values, max_iterations);
	// 		total_iters += kmeans.run(points_copy); // run() now returns iteration count
	// 	}
	// 	long long avg_iters = total_iters / numRuns;
	// 	cout << K << "," << avg_iters << endl;
	// }

	return 0;
}