# Homework 1
The aim of this homework is to test two different methods to aproximate the count triangle problem.
The count triangle problem is the problem to count the number of distinct triangles in a graph.

# How to run
To run the program you need to have pyspark installed.

You need to pass to the main function three parameters:
C: parameter used to partition the data
R: times that runs the algo
file_name: name of the file containing the graph

To call the function it's enough to type:
<code> python main.py C R file_name </code>

# Graph format
The graph is saved in a *.txt file 
Each entry of the file reppresent the end point of an edge