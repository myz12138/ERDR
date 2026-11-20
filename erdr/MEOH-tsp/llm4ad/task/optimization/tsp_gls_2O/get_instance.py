import numpy as np
import numpy.typing as npt
from scipy.spatial import distance_matrix


class GetData():
    def __init__(self,n_instance,n_cities):
        self.n_instance = n_instance
        self.n_cities = n_cities

    def generate_instances(self):
        np.random.seed(2024)
        instance_data = []
        for _ in range(self.n_instance):
            coordinates = np.random.random((self.n_cities, 2))
            instance_data.append(TSPInstance(coordinates))
        return instance_data

class TSPInstance:
    def __init__(
        self,
        positions: npt.NDArray[np.float64],
        distance_matrix_data: npt.NDArray[np.float64] | None = None,
    ) -> None:
        self.positions = np.asarray(positions, dtype=np.float64)
        self.n = self.positions.shape[0]
        if distance_matrix_data is None:
            self.distmat = distance_matrix(self.positions, self.positions) + np.eye(self.n) * 1e-5
        else:
            matrix = np.asarray(distance_matrix_data, dtype=np.float64)
            if matrix.shape != (self.n, self.n):
                raise ValueError(f"distance matrix shape must be {(self.n, self.n)}, got {matrix.shape}")
            self.distmat = matrix
