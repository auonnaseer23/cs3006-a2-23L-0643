#!/usr/bin/env python3
"""
generate_data.py -- Generates data.dat for CS3006 Assignment 2, Program 6 (K-Means).

FAST-NUCES CS3006 (Parallel and Distributed Computing), Fall 2026.
Instructor: Dr. Abdul Qadeer.

This replaces the ~800 MB data.dat that the Stanford handout tells you to fetch
from their AFS filesystem (which you have no access to). The generator is
deterministic: the same command produces a byte-identical file every time, on
every machine, so everybody in the class is timing the same workload.

Binary format (must match readData() in prog6_kmeans/utils.cpp):
  [int    M        ]  4 bytes      -- number of data points
  [int    N        ]  4 bytes      -- number of dimensions per point
  [int    K        ]  4 bytes      -- number of clusters
  [double epsilon  ]  8 bytes      -- convergence threshold
  [double data     ]  M*N*8 bytes  -- data points, row-major
  [double centroids]  K*N*8 bytes  -- initial cluster centroids
  [int assignments ]  M*4 bytes    -- initial cluster assignments

Usage:
    python3 generate_data.py                    # full scale, ~800 MB  <-- USE THIS ONE
    python3 generate_data.py --small            # 10k points, for a quick smoke test only
    python3 generate_data.py --M 100000 --output data_small.dat

Dependencies: numpy  (pip install numpy)

Peak RAM: about 1.1 GB at full scale. Free disk needed: about 800 MB.
"""

import argparse
import os
import struct
import numpy as np

# -- defaults matching the workload described in the Stanford handout ---------
DEFAULT_M       = 1_000_000
DEFAULT_N       = 100
DEFAULT_K       = 3
DEFAULT_EPSILON = 0.1
DEFAULT_OUTPUT  = "data.dat"
SEED            = 7        # matches #define SEED 7 in main.cpp
CHUNK           = 100_000  # do not change: the RNG draw order depends on it,
                           # and changing it changes the generated file


def generate(M: int, N: int, K: int, epsilon: float,
             output: str, verbose: bool = True) -> None:

    rng = np.random.default_rng(SEED)

    header_bytes = 4 + 4 + 4 + 8
    total_bytes  = header_bytes + M * N * 8 + K * N * 8 + M * 4
    if verbose:
        print(f"Generating {M:,} points x {N} dims, K={K} clusters -> {output}")
        print(f"Expected file size: {total_bytes / 1024**2:.1f} MB")
        print(f"Peak RAM required : about {(M * N * 8 + CHUNK * K * N * 8 * 2) / 1024**2:.0f} MB")

    # -- data points: mixture of K Gaussians ---------------------------------
    # Centres are close together and the noise is large relative to their
    # separation, so the clusters overlap heavily in any 2-D (PCA) projection.
    # That is expected -- see the warning about plot.py in the handout.
    centres = rng.uniform(0.0, 1.0, size=(K, N))
    labels  = rng.integers(0, K, size=M)

    # Preallocate and fill in place. Building a list of chunks and calling
    # np.vstack at the end would hold two full copies of an 800 MB array at
    # once, which pushes peak RAM past what a typical 8 GB laptop can give you.
    data = np.empty((M, N), dtype=np.float64)
    for start in range(0, M, CHUNK):
        end = min(start + CHUNK, M)
        noise = rng.standard_normal(size=(end - start, N)) * 3.0
        data[start:end] = centres[labels[start:end]] + noise
        if verbose:
            print(f"  points: {end:>10,} / {M:,}  ({end / M * 100:3.0f}%)", end="\r")
    if verbose:
        print()

    # -- initial centroids (mirrors initCentroids() in main.cpp) -------------
    centroids = np.empty((K, N), dtype=np.float64)
    centroids[0] = rng.uniform(0.0, 1.0, size=N)
    for k in range(1, K):
        centroids[k] = centroids[0] + (rng.uniform(0.0, 1.0, size=N) - 0.5) * 0.1

    # -- initial assignments: nearest centroid -------------------------------
    # Computed in chunks. The obvious one-liner
    #     data[:, None, :] - centroids[None, :, :]
    # materialises an (M, K, N) array -- 2.4 GB at full scale, and the squaring
    # step allocates a second one. Chunking keeps this under 250 MB.
    if verbose:
        print("  computing initial assignments ...")
    assignments = np.empty(M, dtype=np.int32)
    for start in range(0, M, CHUNK):
        end = min(start + CHUNK, M)
        block = data[start:end]                                   # (chunk, N)
        d2 = ((block[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2).sum(axis=2)
        assignments[start:end] = np.argmin(d2, axis=1)

    # -- write the binary file in the exact layout readData() expects --------
    if verbose:
        print(f"  writing {output} ...")
    with open(output, "wb") as f:
        f.write(struct.pack("i", M))        # int M
        f.write(struct.pack("i", N))        # int N
        f.write(struct.pack("i", K))        # int K
        f.write(struct.pack("d", epsilon))  # double epsilon
        f.write(data.tobytes())             # double data[M*N]
        f.write(centroids.tobytes())        # double clusterCentroids[K*N]
        f.write(assignments.tobytes())      # int clusterAssignments[M]

    if verbose:
        actual = os.path.getsize(output)
        print(f"Done. File: {output}  ({actual / 1024**2:.1f} MB)")
        print(f"Header written: M={M}, N={N}, K={K}, epsilon={epsilon}")
        if actual != total_bytes:
            print(f"WARNING: expected {total_bytes} bytes, got {actual}. "
                  f"Check free disk space.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate data.dat for CS3006 asst2 prog6_kmeans"
    )
    parser.add_argument("--M", type=int, default=DEFAULT_M,
                        help=f"Number of data points (default: {DEFAULT_M:,})")
    parser.add_argument("--N", type=int, default=DEFAULT_N,
                        help=f"Dimensions per point (default: {DEFAULT_N})")
    parser.add_argument("--K", type=int, default=DEFAULT_K,
                        help=f"Number of clusters (default: {DEFAULT_K})")
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON,
                        help=f"Convergence threshold (default: {DEFAULT_EPSILON})")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help=f"Output path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--small", action="store_true",
                        help="Shortcut: only 10,000 points (smoke test, NOT for your report)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output")
    args = parser.parse_args()

    if args.small:
        args.M = 10_000

    generate(M=args.M, N=args.N, K=args.K, epsilon=args.epsilon,
             output=args.output, verbose=not args.quiet)


if __name__ == "__main__":
    main()
