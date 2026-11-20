"""Self-contained seed used in each AHD framework's existing prompt format."""


def perturb_tour(distance, current_tour, search_state, rng):
    """Perturb the route using four cuts; preserve the fixed prefix and suffix."""
    tour = list(current_tour)
    stop = search_state["movable_stop"]
    if stop >= 8:
        cuts = sorted(rng.sample(range(1, stop - 3), 4))
        a, b, c, d = [cut + offset for offset, cut in enumerate(cuts)]
        return tour[:a] + tour[c:d] + tour[b:c] + tour[a:b] + tour[d:]
    return tour
