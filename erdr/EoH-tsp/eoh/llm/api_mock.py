import random


class InterfaceMock:
    """Fallback LLM interface for offline smoke runs."""

    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode

    def get_response(self, prompt_content: str):
        if "update_edge_distance" in str(prompt_content):
            scale = random.choice([0.45, 0.55, 0.65, 0.75])
            power = random.choice([0.8, 1.0, 1.2, 1.4])
            alg = (
                "Increase distances on recently used tour edges with a smooth power penalty "
                "to help GLS escape local optima."
            )
            code = f"""import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    updated_edge_distance = np.array(edge_distance, copy=True)
    max_used = float(np.max(edge_n_used)) if edge_n_used.size > 0 else 0.0
    penalty = {scale:.2f} * np.power(np.maximum(max_used - edge_n_used, 0.0), {power:.2f})
    for i in range(len(local_opt_tour) - 1):
        a = int(local_opt_tour[i])
        b = int(local_opt_tour[i + 1])
        updated_edge_distance[a, b] += penalty[a, b]
        updated_edge_distance[b, a] = updated_edge_distance[a, b]
    return updated_edge_distance
"""
        else:
            w1 = random.choice([0.1, 0.15, 0.2, 0.25])
            w2 = round(1.0 - w1, 2)
            alg = (
                "Blend current-edge distance with destination proximity to keep greedy choice "
                "stable while improving return-to-start awareness."
            )
            code = f"""import numpy as np

def select_next_node(current_node, destination_node, univisited_nodes, distance_matrix):
    if len(univisited_nodes) == 0:
        return int(destination_node)
    local_cost = distance_matrix[current_node, univisited_nodes]
    return_cost = distance_matrix[univisited_nodes, destination_node]
    score = {w1} * local_cost + {w2} * return_cost
    idx = int(np.argmin(score))
    return int(univisited_nodes[idx])
"""
        return "{%s}\n```python\n%s\n```" % (alg, code)
