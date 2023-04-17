#------------------------------------------------------------------
# Big Data 2022 - 2023
# Homework 1: Triangle Count
# 
# Lorenzon Nicola, Manfè Alessandro, Mickel Mauro
#------------------------------------------------------------------

from pyspark import SparkContext, SparkConf, RDD
import time
import sys
import os
import numpy as np
import random as rand
# questo serve per dopo? io lascerei la chiamata alla funzione con CountTriangles e non con ct per readability poi ditemi voi <3
import CountTriangles as ct
from collections import defaultdict


# Prime number constant for hashing
P = 8191

# Function that returns the number of triangle in a given edges list
def CountTriangles(edges):
    # Create a defaultdict to store the neighbors of each vertex
    neighbors = defaultdict(set)
    for edge in edges:
        u, v = edge
        neighbors[u].add(v)
        neighbors[v].add(u)

    # Initialize the triangle count to zero
    triangle_count = 0

    # Iterate over each vertex in the graph.
    # To avoid duplicates, we count a triangle <u, v, w> only if u<v<w
    for u in neighbors:
        # Iterate over each pair of neighbors of u
        for v in neighbors[u]:
            if v > u:
                for w in neighbors[v]:
                    # If w is also a neighbor of u, then we have a triangle
                    if w > v and w in neighbors[u]:
                        triangle_count += 1
    # Return the total number of triangles in the graph
    return triangle_count

class HashPartitioner:
    def __init__(self,p,c):
        self.p = p
        self.c = c
        self.a = rand.randint(1,self.p-1)
        self.b = rand.randint(1,self.p-1)

    def getHash(self,u):
        return ((self.a*u + self.b) % self.p) % self.c
    
    def cmpTuple(self,t: tuple):
        if len(t) != 2:
            return False
        return self.getHash(t[0]) == self.getHash(t[1])

def MR_ApproxTCwithNodeColors(docs: RDD, C: int):
    hp = HashPartitioner(P,C)
    tri_count = (docs
                 .filter(hp.cmpTuple)
                 .map(lambda x: (hp.getHash(x[0]),x))
                 .groupByKey()
                 .map(lambda x: (1,CountTriangles(x[1])))
                 .reduceByKey(lambda x, y: x + y)
                 )
    val = tri_count.collect()[0][1]
    return val

# Execute second algorithm on the RDD
def MR_ApproxTCwithSparkPartitions(docs: RDD):
    tri_count = (docs
                 .mapPartitions(lambda x: [(1,CountTriangles(x))])  # MAP Phase: M2
                 .reduceByKey(lambda x, y: x + y))                  # REDUCE Phase: R2
    val = tri_count.collect()[0][1]
    return val

def main():

    # Check the number of parameters
    if not (len(sys.argv) == 4):
        raise TypeError(f"You must pass 4 param")

    # Number of partitions
    C = int(sys.argv[1])
    # Number of rounds
    R = int(sys.argv[2])

    
    data_path = str(sys.argv[3])
    
    #print("C: " + str(C))
    #print("R: " + str(R))
    #print("data_path: " + data_path)
    

    # Spark app name and configuration
    conf = SparkConf().setAppName('TriangleCountExercise')
    sc = SparkContext(conf=conf)

    # Check if file exists
    assert os.path.isfile(data_path), "File or folder not found"
    docs = sc.textFile(data_path,minPartitions=C).cache()
    
    #print(f"num part before map: {docs.getNumPartitions()}")

    # Creates the RDD of edges 
    edges_df  = docs.map(lambda x: (int(x.split(",")[0]), int(x.split(",")[1]))).repartition(C) # MAP Phase: M1
    #print(f"num part after map: {edges_df.getNumPartitions()}")

    res_one_time = []
    res_one_res = []
    res_two_time = []
    res_two_res = []    
    
    #refRDD = edges_df.coalesce(1)
    #start_time = time.time()
    #tri_number = MR_ApproxTCwithSparkPartitions(refRDD)
    #end_time = time.time()
    #print("reference result: " + str(tri_number))
    #print("reference time: " + str(end_time-start_time))
    
    # Runs the first algorithm R times and takes the time intervall between the start and the end
    for i in range(R):
        start_time = time.time()
        tri_number = (C**2)*MR_ApproxTCwithNodeColors(edges_df,C)
        end_time = time.time()
        res_one_time.append(end_time - start_time)
        res_one_res.append(tri_number)

    # Prints data about the runs
    print("MR_ApproxTCwithNodeColors median of result: " + str(np.median(res_one_res)))
    print("MR_ApproxTCwithNodeColors average of time: " + str(np.mean(res_one_time)))

    
    # Runs the second algorithm and takes the time intervall between the start and the end
    start_time = time.time()
    tri_number = (C**2)*MR_ApproxTCwithSparkPartitions(edges_df)
    end_time = time.time()
    res_two_time.append(end_time - start_time)
    res_two_res.append(tri_number)

    # Prints data about the run
    print("MR_ApproxTCwithSparkPartitions median of result: " + str(np.median(res_two_res)))
    print("MR_ApproxTCwithSparkPartitions average of time: " + str(np.mean(res_two_time)))
    
if __name__ == "__main__":
    #findspark.init()
    main()