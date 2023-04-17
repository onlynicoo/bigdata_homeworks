from CountTriangles import CountTriangles
from pyspark import SparkContext, SparkConf, RDD
from random import randint
from time import time
import numpy as np
import os
import sys

P = 8191
DEFAULT_KEY = 1


def count_triangles_partitions(pairs):
    edges = [pair[1] for pair in pairs]
    return [(DEFAULT_KEY, CountTriangles(edges))]


def MR_ApproxTCwithSparkPartitions(edges: RDD, C: int):
    triangle_count = (edges
                      .map(lambda x: (randint(0, C - 1), x))  # MAP PHASE (R1)
                      .mapPartitions(count_triangles_partitions)  # REDUCE PHASE (R1)
                      # .groupByKey()  # SHUFFLE+GROUPING
                      # .mapValues(lambda vals: sum(vals))  # REDUCE PHASE (R2)
                      .reduceByKey(lambda x, y: x + y)  # REDUCE PHASE (R2)
                      )
    return (C ** 2) * triangle_count.values().first()


def main():

    # Check number of given arguments
    assert len(sys.argv) == 4, "Usage: python G017HW1.py <C> <R> <file_name>"

    # Spark setup
    conf = SparkConf().setAppName('G017HW1')
    sc = SparkContext(conf=conf)

    # Read number of partitions
    C = sys.argv[1]
    assert C.isdigit(), "C must be an integer"
    C = int(C)

    # Read number of runs
    R = sys.argv[2]
    assert R.isdigit(), "R must be an integer"
    R = int(R)

    # Read input file and subdivide it into C random partitions
    data_path = sys.argv[3]
    assert os.path.isfile(data_path), "File or folder not found"
    rawData = sc.textFile(data_path)
    edges = rawData.map(
        lambda edge: (int(edge.split(',')[0]), int(edge.split(',')[1]))
    ).repartition(C).cache()

    # Print input parameters
    print(f"Dataset = {data_path}\nNumber of Edges = {edges.count()}\nNumber of Colors = {C}\nNumber of Repetitions = {R}")

    # 2-ROUND TRIANGLE COUNT WITH SPARK PARTITIONS (ALG. 2)
    start_time = time()
    result = MR_ApproxTCwithSparkPartitions(edges, C)
    run_time = time() - start_time
    print("Approximation through Spark partitions")
    print(f"- Number of triangles = {result}")
    print(f"- Running time = {int(run_time * 1000)} ms")


if __name__ == "__main__":
    main()
