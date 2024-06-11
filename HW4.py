import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score

def generate_community_graph(num_vertices, community_size, p_in, p_out):
    """
    Generate a graph with communities.
    
    Parameters:
    - num_vertices: Total number of vertices in the graph.
    - community_size: Number of vertices in each community.
    - p_in: Probability of connecting vertices within the same community.
    - p_out: Probability of connecting vertices between different communities.
    
    Returns:
    - G: Generated network graph.
    """
    G = nx.Graph()
    G.add_nodes_from(range(num_vertices))
    
    # Add intra-community edges
    for community_start in range(0, num_vertices, community_size):
        for i in range(community_start, community_start + community_size):
            for j in range(i + 1, community_start + community_size):
                if np.random.rand() < p_in:
                    G.add_edge(i, j)
    
    # Add inter-community edges
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if (i // community_size) != (j // community_size) and np.random.rand() < p_out:
                G.add_edge(i, j)

    return G

def expectation_step(A, pi, p, num_groups):
    """
    Perform the expectation step of the EM algorithm.
    
    Parameters:
    - A: Adjacency matrix of the graph.
    - pi: Mixing probabilities.
    - p: Connection probabilities between groups.
    - num_groups: Number of groups.
    
    Returns:
    - gamma: Responsibilities matrix.
    """
    N = len(A)
    gamma = np.zeros((N, num_groups))
    
    for i in range(N):
        for k in range(num_groups):
            gamma[i, k] = np.log(pi[k])
            for j in range(N):
                if i != j:
                    gamma[i, k] += A[i, j] * np.log(p[k, j // (N // num_groups)])
        
        # Normalize to avoid underflow issues
        max_gamma = np.max(gamma[i, :])
        gamma[i, :] = np.exp(gamma[i, :] - max_gamma)
        gamma[i, :] /= np.sum(gamma[i, :])
    
    return gamma

def maximization_step(A, gamma, num_groups):
    """
    Perform the maximization step of the EM algorithm.
    
    Parameters:
    - A: Adjacency matrix.
    - gamma: Responsibilities from E-step.
    - num_groups: Number of groups.
    
    Returns:
    - pi: Updated mixing probabilities.
    - p: Updated connection probabilities.
    """
    N = len(A)
    pi = np.sum(gamma, axis=0) / N
    p = np.zeros((num_groups, num_groups))
    
    for k in range(num_groups):
        for l in range(num_groups):
            numerator = sum(A[i, j] * gamma[i, k] * gamma[j, l] for i in range(N) for j in range(i + 1, N))
            denominator = sum(gamma[i, k] * gamma[j, l] for i in range(N) for j in range(i + 1, N))
            p[k, l] = numerator / denominator if denominator != 0 else 0
    
    return pi, p

def em_algorithm(A, num_groups, max_iter=100, tol=1e-4):
    """
    Execute the EM algorithm on graph data.
    
    Parameters:
    - A: Adjacency matrix of the graph.
    - num_groups: Number of groups or communities.
    - max_iter: Maximum number of iterations.
    - tol: Tolerance for convergence.
    
    Returns:
    - pi: Final mixing probabilities.
    - p: Final connection probabilities.
    - gamma: Final responsibilities matrix.
    """
    N = len(A)
    pi = np.ones(num_groups) / num_groups
    p = np.random.rand(num_groups, num_groups) * 0.5
    
    for iteration in range(max_iter):
        gamma = expectation_step(A, pi, p, num_groups)
        pi_new, p_new = maximization_step(A, gamma, num_groups)
        
        if np.linalg.norm(pi - pi_new) < tol and np.linalg.norm(p - p_new) < tol:
            break
        
        pi, p = pi_new, p_new
    
    return pi, p, gamma

if __name__ == "__main__":
    # Parameters
    community_size = 100
    num_vertices = 200
    p_in = 8 / (community_size - 1)
    p_out = 2 / (num_vertices - community_size)
    num_groups = 2

    # Generate graph
    G = generate_community_graph(num_vertices, community_size, p_in, p_out)

    # Get adjacency matrix
    A = nx.adjacency_matrix(G).todense()

    # Generate true labels
    true_labels = np.array([i // community_size for i in range(num_vertices)])

    # Apply EM algorithm
    pi, p, gamma = em_algorithm(A, num_groups)

    # Determine the detected labels from the responsibilities
    detected_labels = np.argmax(gamma, axis=1)

    # Compute NMI between true labels and detected labels
    nmi = normalized_mutual_info_score(true_labels, detected_labels)

    # Plot adjacency matrix
    plt.figure(figsize=(8, 8))
    plt.imshow(A, cmap="Blues", interpolation="none")
    plt.title("Adjacency Matrix of the Network")
    plt.colorbar()
    plt.show()

    # Print estimated parameters and NMI score
    print("Estimated mixing probabilities (pi):", pi)
    print("Estimated connection probabilities (p):\n", p)
    print("Normalized Mutual Information (NMI):", nmi)
