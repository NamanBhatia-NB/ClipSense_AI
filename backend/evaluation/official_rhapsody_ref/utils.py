import numpy as np

def graph_to_indices(
    graph: np.array,
    max_n_gt: int,
    threshold: float,
) -> list[int]:
    graph = graph.copy()

    # correct the first segment bias (appendix A.3 in the paper)
    graph[0] = graph[1]
    
    indices = np.nonzero(graph > threshold)[0]
    sorted_indices = indices[np.argsort(graph[indices])[::-1]][:max_n_gt].tolist()
    return sorted_indices
