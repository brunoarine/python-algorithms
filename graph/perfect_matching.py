"""
A perfect matching is a matching that matches all vertices of the graph. That is, a matching is perfect if every vertex
of the graph is incident to an edge of the matching.

Let G=(V, E) be an undirected simple graph, and let $T$ be its associated Tutte matrix. Then, det(T) = 0 if and only
if $G$ has a perfect matching.

To speed up the detection of perfect matchings, I used Schwartz-Zippel's Theorem: Suppose det(T) = 0; then suppose
each variable in T was set to an element in {1, . . . , n2} uniformly at random within the {1,n**2} interval.
Then, P[det(T)(x) = 0] ≤ 1/n, where the randomness is taken over the setting of each variable.

References
----------
Wikipedia contributors. (2021, March 22). Matching (graph theory). In Wikipedia, The Free
Encyclopedia. https://en.wikipedia.org/w/index.php?title=Matching_(graph_theory)&oldid=1013559292

Ivan, I., Virza, M., & Yuen, H. (2011). Algebraic Algorithms for
Matching. https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.714.671&rep=rep1&type=pdf

>>> has_perfect_match([[3, 4], [4], [1, 4], [1, 2, 3]])
True

>>> has_perfect_match([[], [3, 4, 6], [2], [4], [], [6]])
False
"""
import random
import numpy as np

def gen_tutte(graph, start=1):
    """
    Generates a Tutte matrix from an undirected, non-bipartite graph
    """
    n = len(graph)
    indices = range(n)
    mat = [([0.0] * n) for i in indices]
    for i, row in enumerate(graph, start=start):
        for j in row:
            ran = random.randint(1, n ** 2)
            mat[i - start][j - start] = float(ran) if i < j else float(-ran)  # Python computes float faster than int
    return mat

def has_perfect_match(graph, start=1):
    """
    Checke whether a graph has a perfect matching by instantiating its Tutte matrix variables with random values
    between 1 and n**2. If det(mat) = 0, then the graph has no perfect matching.
    :param graph: a 2-D nested list
    :param start: number of the first vertex in the graph (default = 1)
    :return: boolean
    """
    tutte_mat = gen_tutte(graph, start=1)
    det = np.linalg.slogdet(tutte_mat)[0]  # slogdet prevents under and overflow
    return bool(det)


if __name__ == "__main__":
    import doctest
    doctest.testmod()