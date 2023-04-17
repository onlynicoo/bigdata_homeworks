"""
    BIG DATA COMPUTING 2022-2023 - Homework 1
    Group 17 - Members:
    - CHRISTIAN MARCHIORI 2078343
    - DANIELE MOSCHETTA 2087640
    - FABIO ZANINI 2088628
"""

from CountTriangles import CountTriangles
from pyspark import SparkContext, SparkConf, RDD
from random import randint
from time import time
import numpy as np
import os
import sys


P = 8191            # Prime number used in the node color hash function
DEFAULT_KEY = 1     # Used to make group by default key more readable


def color_nodes(edge, C, a, b):
    color_1 = ((a * edge[0] + b) % P) % C
    color_2 = ((a * edge[1] + b) % P) % C
    if color_1 == color_2:
        return [(color_1, edge)]
    else:
        return []


def MR_ApproxTCwithNodeColors(edges: RDD, C: int):
    a = randint(1, P - 1)
    b = randint(0, P - 1)
    triangle_count = (edges.flatMap(lambda edge: color_nodes(edge, C, a, b))  # MAP PHASE (R1)
                      .groupByKey()  # SHUFFLE + GROUPING
                      .map(lambda pair: (DEFAULT_KEY, CountTriangles(pair[1])))  # REDUCE PHASE (R1)
                      .reduceByKey(lambda x, y: x + y)  # REDUCE PHASE (R2)
                      )
    return (C ** 2) * triangle_count.values().first()


def MR_ApproxTCwithSparkPartitions(edges: RDD, C: int):
    triangle_count = (edges  # MAP PHASE (R1) has been done with Spark partitions via RDD.repartition(numPartitions) in the main function
                      .mapPartitions(lambda edges: [(DEFAULT_KEY, CountTriangles(edges))])  # REDUCE PHASE (R1)
                      .reduceByKey(lambda x, y: x + y)  # REDUCE PHASE (R2)
                      )
    return (C ** 2) * triangle_count.values().first()


def main():

    # Check number of given arguments
    assert len(sys.argv) == 4, "Usage: python G017HW1.py <C> <R> <file_name>"

    # Read number of partitions
    C = sys.argv[1]
    assert C.isdigit(), "C must be an integer"
    C = int(C)
    assert C > 0, "C must be a positive integer"

    # Read number of runs
    R = sys.argv[2]
    assert R.isdigit(), "R must be an integer"
    R = int(R)
    assert R >= 0, "R must be a non-negative integer"   # If R == 0 we run only the second algorithm

    # Read data path
    data_path = sys.argv[3]
    assert os.path.isfile(data_path), "File or folder not found"

    # Spark setup
    conf = SparkConf().setAppName('G017HW1')
    sc = SparkContext(conf=conf)

    # Read input file and subdivide it into C random partitions
    rawData = sc.textFile(data_path, minPartitions=C)
    edges = rawData.map(
        lambda edge: (int(edge.split(',')[0]), int(edge.split(',')[1]))
    ).repartition(numPartitions=C).cache()

    # Print input parameters
    print(f"Dataset = {data_path}\nNumber of Edges = {edges.count()}\nNumber of Colors = {C}\nNumber of Repetitions = {R}")

    # 2-Round triangle count with node colors (Alg. 1)
    results = []
    run_times = []
    for i in range(R):
        start_time = time()
        result = MR_ApproxTCwithNodeColors(edges, C)
        run_time = time() - start_time
        results.append(result)
        run_times.append(run_time)
    if R > 0:
        print("Approximation through node coloring")
        print(f"- Number of triangles (median over {R} runs) = {int(np.median(results))}")
        print(f"- Running time (average over {R} runs) = {int(np.average(run_times) * 1000)} ms")

    # 2-Round triangle count with Spark partitions (Alg. 2)
    start_time = time()
    result = MR_ApproxTCwithSparkPartitions(edges, C)
    run_time = time() - start_time
    print("Approximation through Spark partitions")
    print(f"- Number of triangles = {result}")
    print(f"- Running time = {int(run_time * 1000)} ms")


if __name__ == "__main__":
    main()
