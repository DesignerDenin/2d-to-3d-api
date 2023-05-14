from skimage.measure import marching_cubes
import numpy as np
import trimesh

class MISE:
    def __init__(self, resolution, upsample_steps, threshold):
        self.resolution = resolution
        self.upsample_steps = upsample_steps
        self.threshold = threshold
        self.grid_values = {}
        self.triangles = []

    def add_grid_point(self, loc):
        self.grid_values[loc] = 0

    def evaluate_point(self, loc):
        # Your code to evaluate the value of a point goes here
        pass

    def extract_mesh(self):
        # Create an empty 3D grid
        grid = np.zeros(self.resolution)

        # Evaluate the value of each grid point
        for loc in self.grid_values.keys():
            grid[loc] = self.evaluate_point(loc)

        # Perform upsampling steps
        for _ in range(self.upsample_steps):
            grid = np.kron(grid, np.ones((2, 2, 2)))

        # Extract mesh using marching cubes
        vertices, faces, _, _ = marching_cubes(grid, level=self.threshold)

        # Create a TriMesh object
        mesh = trimesh.Trimesh(vertices, faces)

        return mesh
    
    def query(self):
        # Find all points with unknown value
        unknown_points = []
        for p in self.grid_points:
            if not p.known:
                unknown_points.append(p.loc)

        # Convert to numpy array
        points_np = np.array(unknown_points, dtype=np.int64)

        return points_np
        
    def evaluate_point(self, loc):
        # Evaluate the value at the given location using trilinear interpolation
        p0 = np.floor(loc).astype(int)
        p1 = p0 + 1

        # Get the eight grid points surrounding the location
        points = [
            (p0[0], p0[1], p0[2]),
            (p0[0], p0[1], p1[2]),
            (p0[0], p1[1], p0[2]),
            (p0[0], p1[1], p1[2]),
            (p1[0], p0[1], p0[2]),
            (p1[0], p0[1], p1[2]),
            (p1[0], p1[1], p0[2]),
            (p1[0], p1[1], p1[2])
        ]

        # Get the corresponding values at the grid points
        values = [self.grid_values.get(p, 0) for p in points]

        # Perform trilinear interpolation
        dx = loc - p0
        interpolated_value = self.trilinear_interpolation(values, dx)

        return interpolated_value

    def trilinear_interpolation(self, values, dx):
        # Perform trilinear interpolation between the grid values
        v000, v001, v010, v011, v100, v101, v110, v111 = values
        x, y, z = dx

        c00 = v000 * (1 - x) + v100 * x
        c01 = v001 * (1 - x) + v101 * x
        c10 = v010 * (1 - x) + v110 * x
        c11 = v011 * (1 - x) + v111 * x

        c0 = c00 * (1 - y) + c10 * y
        c1 = c01 * (1 - y) + c11 * y

        interpolated_value = c0 * (1 - z) + c1 * z

        return interpolated_value
    

    def to_dense(self):
        # Create a dense grid using the bounding box of the points
        min_coords = np.min(self.grid_points, axis=0)
        max_coords = np.max(self.grid_points, axis=0)
        bb_min = np.floor(min_coords).astype(int)
        bb_max = np.ceil(max_coords).astype(int)

        # Compute the shape of the dense grid
        shape = bb_max - bb_min + 1

        # Create an empty dense grid
        dense_grid = np.zeros(shape, dtype=float)

        # Convert the grid points to dense grid indices
        indices = self.grid_points - bb_min

        # Fill the dense grid with the corresponding values
        dense_grid[tuple(indices.T)] = list(self.grid_values.values())

        return dense_grid
