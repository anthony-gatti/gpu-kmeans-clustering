/* Print section for all versions except original
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
*/

/* Original print section
    // shows elements of clusters
    for(int i = 0; i < K; i++)
    {
        int total_points_cluster =  clusters[i].getTotalPoints();

        cout << "Cluster " << clusters[i].getID() + 1 << endl;
        for(int j = 0; j < total_points_cluster; j++)
        {
            cout << "Point " << clusters[i].getPoint(j).getID() + 1 << ": ";
            for(int p = 0; p < total_values; p++)
                cout << clusters[i].getPoint(j).getValue(p) << " ";

            string point_name = clusters[i].getPoint(j).getName();

            if(point_name != "")
                cout << "- " << point_name;

            cout << endl;
        }

        cout << "Cluster values: ";

        for(int j = 0; j < total_values; j++)
            cout << clusters[i].getCentralValue(j) << " ";

        cout << "\n\n";
        cout << "TOTAL EXECUTION TIME = "<<std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count()<<"\n";

        cout << "TIME PHASE 1 = "<<std::chrono::duration_cast<std::chrono::microseconds>(end_phase1-begin).count()<<"\n";

        cout << "TIME PHASE 2 = "<<std::chrono::duration_cast<std::chrono::microseconds>(end-end_phase1).count()<<"\n";
    }
*/