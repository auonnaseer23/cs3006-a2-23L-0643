#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <vector>
#include <mutex>

#include "CycleTimer.h"

using namespace std;

typedef struct {
  // Control work assignments
  int start, end;

  // Shared by all functions
  double *data;
  double *clusterCentroids;
  int *clusterAssignments;
  double *currCost;
  int M, N, K;

  // Fields for parallel computeCentroids (reduction pattern)
  std::mutex *centroidMutex;
  double *globalSum;
  int *globalCount;
} WorkerArgs;


/**
 * Checks if the algorithm has converged.
 * 
 * @param prevCost Pointer to the K dimensional array containing cluster costs 
 *    from the previous iteration.
 * @param currCost Pointer to the K dimensional array containing cluster costs 
 *    from the current iteration.
 * @param epsilon Predefined hyperparameter which is used to determine when
 *    the algorithm has converged.
 * @param K The number of clusters.
 * 
 * NOTE: DO NOT MODIFY THIS FUNCTION!!!
 */
static bool stoppingConditionMet(double *prevCost, double *currCost,
                                 double epsilon, int K) {
  for (int k = 0; k < K; k++) {
    if (abs(prevCost[k] - currCost[k]) > epsilon)
      return false;
  }
  return true;
}

/**
 * Computes L2 distance between two points of dimension nDim.
 * 
 * @param x Pointer to the beginning of the array representing the first
 *     data point.
 * @param y Poitner to the beginning of the array representing the second
 *     data point.
 * @param nDim The dimensionality (number of elements) in each data point
 *     (must be the same for x and y).
 */
double dist(double *x, double *y, int nDim) {
  double accum = 0.0;
  for (int i = 0; i < nDim; i++) {
    accum += pow((x[i] - y[i]), 2);
  }
  return sqrt(accum);
}

/**
 * Assigns each data point to its "closest" cluster centroid.
 */
void computeAssignments(WorkerArgs *const args) {
  for (int m = args->start; m < args->end; m++) {
    double minDist = 1e30;
    int best = -1;
    for (int k = 0; k < args->K; k++) {
      double d = dist(&args->data[m * args->N],
                       &args->clusterCentroids[k * args->N], args->N);
      if (d < minDist) {
        minDist = d;
        best = k;
      }
    }
    args->clusterAssignments[m] = best;
  }
}


/**
 * Given the cluster assignments, computes the new centroid locations for
 * each cluster.
 */
void computeCentroids(WorkerArgs *const args) {
  int N = args->N, K = args->K;
  std::vector<double> localSum(K * N, 0.0);
  std::vector<int> localCount(K, 0);

  for (int m = args->start; m < args->end; m++) {
    int k = args->clusterAssignments[m];
    for (int n = 0; n < N; n++) {
      localSum[k * N + n] += args->data[m * N + n];
    }
    localCount[k]++;
  }

  std::lock_guard<std::mutex> lock(*args->centroidMutex);
  for (int k = 0; k < K; k++) {
    args->globalCount[k] += localCount[k];
    for (int n = 0; n < N; n++) {
      args->globalSum[k * N + n] += localSum[k * N + n];
    }
  }
}


/**
 * Computes the per-cluster cost. Used to check if the algorithm has converged.
 */
void computeCost(WorkerArgs *const args) {
  double *accum = new double[args->K];

  // Zero things out
  for (int k = 0; k < args->K; k++) {
    accum[k] = 0.0;
  }

  // Sum cost for all data points assigned to centroid
  for (int m = 0; m < args->M; m++) {
    int k = args->clusterAssignments[m];
    accum[k] += dist(&args->data[m * args->N],
                     &args->clusterCentroids[k * args->N], args->N);
  }

  // Update costs
  for (int k = args->start; k < args->end; k++) {
    args->currCost[k] = accum[k];
  }

  delete[] accum;
}

/**
 * Computes the K-Means algorithm, using std::thread to parallelize the work.
 *
 * @param data Pointer to an array of length M*N representing the M different N 
 *     dimensional data points clustered. The data is layed out in a "data point
 *     major" format, so that data[i*N] is the start of the i'th data point in 
 *     the array. The N values of the i'th datapoint are the N values in the 
 *     range data[i*N] to data[(i+1) * N].
 * @param clusterCentroids Pointer to an array of length K*N representing the K 
 *     different N dimensional cluster centroids. The data is laid out in
 *     the same way as explained above for data.
 * @param clusterAssignments Pointer to an array of length M representing the
 *     cluster assignments of each data point, where clusterAssignments[i] = j
 *     indicates that data point i is closest to cluster centroid j.
 * @param M The number of data points to cluster.
 * @param N The dimensionality of the data points.
 * @param K The number of cluster centroids.
 * @param epsilon The algorithm is said to have converged when
 *     |currCost[i] - prevCost[i]| < epsilon for all i where i = 0, 1, ..., K-1
 */
void kMeansThread(double *data, double *clusterCentroids, int *clusterAssignments,
               int M, int N, int K, double epsilon) {

  static const int NUM_THREADS = 8;

  double *prevCost = new double[K];
  double *currCost = new double[K];

  std::mutex centroidMutex;
  std::vector<double> globalSum(K * N, 0.0);
  std::vector<int> globalCount(K, 0);

  WorkerArgs args[NUM_THREADS];
  for (int t = 0; t < NUM_THREADS; t++) {
    args[t].data = data;
    args[t].clusterCentroids = clusterCentroids;
    args[t].clusterAssignments = clusterAssignments;
    args[t].currCost = currCost;
    args[t].M = M;
    args[t].N = N;
    args[t].K = K;
    args[t].centroidMutex = &centroidMutex;
    args[t].globalSum = globalSum.data();
    args[t].globalCount = globalCount.data();
  }

  for (int k = 0; k < K; k++) {
    prevCost[k] = 1e30;
    currCost[k] = 0.0;
  }

  int iter = 0;
  double assignTime = 0.0, centroidTime = 0.0, costTime = 0.0;
  while (!stoppingConditionMet(prevCost, currCost, epsilon, K)) {
    for (int k = 0; k < K; k++) {
      prevCost[k] = currCost[k];
    }

    int chunk = (M + NUM_THREADS - 1) / NUM_THREADS;
    for (int t = 0; t < NUM_THREADS; t++) {
      args[t].start = t * chunk;
      args[t].end = std::min(M, (t + 1) * chunk);
    }

    double t0 = CycleTimer::currentSeconds();
    std::thread workers[NUM_THREADS];
    for (int t = 1; t < NUM_THREADS; t++) {
      workers[t] = std::thread(computeAssignments, &args[t]);
    }
    computeAssignments(&args[0]);
    for (int t = 1; t < NUM_THREADS; t++) {
      workers[t].join();
    }
    double t1 = CycleTimer::currentSeconds();

    std::fill(globalSum.begin(), globalSum.end(), 0.0);
    std::fill(globalCount.begin(), globalCount.end(), 0);

    std::thread workers2[NUM_THREADS];
    for (int t = 1; t < NUM_THREADS; t++) {
      workers2[t] = std::thread(computeCentroids, &args[t]);
    }
    computeCentroids(&args[0]);
    for (int t = 1; t < NUM_THREADS; t++) {
      workers2[t].join();
    }

    for (int k = 0; k < K; k++) {
      int cnt = std::max(globalCount[k], 1);
      for (int n = 0; n < N; n++) {
        clusterCentroids[k * N + n] = globalSum[k * N + n] / cnt;
      }
    }
    double t2 = CycleTimer::currentSeconds();

    args[0].start = 0;
    args[0].end = K;
    computeCost(&args[0]);
    double t3 = CycleTimer::currentSeconds();

    assignTime += (t1 - t0);
    centroidTime += (t2 - t1);
    costTime += (t3 - t2);

    iter++;
  }

  double totalPhaseTime = assignTime + centroidTime + costTime;
  printf("[Profile] iterations: %d\n", iter);
  printf("[Profile] computeAssignments: %.3f ms (%.1f%%)\n", assignTime*1000, 100.0*assignTime/totalPhaseTime);
  printf("[Profile] computeCentroids:   %.3f ms (%.1f%%)\n", centroidTime*1000, 100.0*centroidTime/totalPhaseTime);
  printf("[Profile] computeCost:        %.3f ms (%.1f%%)\n", costTime*1000, 100.0*costTime/totalPhaseTime);

  delete[] currCost;
  delete[] prevCost;
}

