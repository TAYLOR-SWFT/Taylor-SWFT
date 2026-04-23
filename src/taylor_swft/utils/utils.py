import numpy as np
import torch


def wall_list_to_triangles(walls: list[list[list]] | list[np.ndarray]) -> np.ndarray:
    """Convert a list of walls (each wall is a list of vertices) to a list of triangles.

    Triangulates each polygonal wall using a fan triangulation method to produce
    a list of triangular faces suitable for geometric calculations.

    Args:
        walls: List of polygonal walls where each wall is represented as a list
            of vertices (either nested lists or numpy arrays).

    Returns:
        np.ndarray: Array of triangular faces extracted from all walls.
    """
    triangles = []
    for wall in walls:
        if len(wall) < 3:
            # Skip walls that cannot form a triangle
            continue
        # Triangulate the wall using a simple fan triangulation
        triangles.extend(polygon_to_triangles_recursive(np.array(wall)))
    return np.array(triangles)


def polygon_to_triangles_recursive(polygon: np.ndarray) -> list[np.ndarray]:
    """Recursively triangulate a polygon using ear clipping method.

    Decomposes a non-convex polygon into triangles by iteratively identifying
    and clipping ears (triangles that can be removed while maintaining convexity).

    Args:
        polygon: Array of polygon vertices in 2D or 3D space of shape (n_vertices, 2|3).

    Returns:
        list[np.ndarray]: List of triangular faces as numpy arrays.
    """
    if len(polygon) < 3:
        # Not a valid polygon
        return []
    if len(polygon) == 3:
        # Already a triangle
        return [polygon]

    triangles = []
    for i in range(len(polygon)):
        prev_index = (i - 1) % len(polygon)
        next_index = (i + 1) % len(polygon)
        ear = [polygon[prev_index], polygon[i], polygon[next_index]]

        # Check if the ear is a valid triangle
        if is_ear(ear, polygon):
            triangles.append(ear)
            # Remove the ear vertex
            new_polygon = np.concat([polygon[:i], polygon[i + 1 :]])
            triangles.extend(polygon_to_triangles_recursive(new_polygon))
            break
    return triangles


def is_ear(ear: list[np.ndarray], polygon: np.ndarray) -> bool:
    """Check if a triangle (ear) is an ear of the polygon.

    An ear is a triangle formed by three consecutive vertices where the triangle
    lies entirely outside the polygon (except for shared edges) and has the same
    orientation as the polygon.

    Args:
        ear: List of three vertices forming a potential ear triangle.
        polygon: Polygon vertices (assumed counter-clockwise order).

    Returns:
        bool: True if the triangle is a valid ear of the polygon.
    """
    ear_array = np.array(ear)
    A, B, C = ear

    shifted_polygon = np.roll(polygon, -1, axis=0)
    shifted_backward_polygon = np.roll(polygon, 1, axis=0)
    polygon_normal = np.mean(
        np.cross(shifted_polygon - polygon, shifted_backward_polygon - polygon),
        axis=0,
    )

    ear_normal = np.cross(B - A, C - A)
    if np.dot(polygon_normal, ear_normal) <= 0:
        # Ear must be oriented the same way as the polygon
        return False

    is_point_in_triangle = lambda P: (
        np.cross(B - A, P - A).dot(polygon_normal) >= 0
        and np.cross(C - B, P - B).dot(polygon_normal) >= 0
        and np.cross(A - C, P - C).dot(polygon_normal) >= 0
    )

    for point in polygon:
        not_in_ear = not np.any(np.all(point == ear_array, axis=1))
        in_triangle = is_point_in_triangle(point)
        if not_in_ear and in_triangle:
            # Ear cannot contain any other points of the polygon
            return False
    return True


def is_inside(
    triangles: np.ndarray,
    X_in: np.ndarray,
    tol: float = 1e-1,
) -> np.typing.NDArray[np.bool_]:
    """Fast implementation of point-in-mesh test using generalized winding numbers.

    Based on the paper "Robust Inside-Outside Segmentation using Generalized Winding
    Numbers" by A.Jacobson et al. Efficiently determines which test points are inside
    a closed triangular mesh.

    Original code: https://github.com/marmakoide/inside-3d-mesh?tab=readme-ov-file

    Args:
        triangles: Array of triangular faces of shape (n_triangles, 3, 3) containing
            3D vertex coordinates for each corner.
        X_in: Test points of shape (n_points, 3).
        tol: Tolerance for winding number threshold (default 0.1). Points with
            |winding_number| >= 1 - tol are considered inside.

    Returns:
        np.typing.NDArray[np.bool_]: Boolean array indicating which points are inside
            the mesh.
    """

    # Compute euclidean norm along axis 1
    def anorm2(X):
        return np.sqrt(np.sum(X**2, axis=1))

    # Compute 3x3 determinant along axis 1
    def adet(X, Y, Z):
        ret = np.multiply(np.multiply(X[:, 0], Y[:, 1]), Z[:, 2])
        ret += np.multiply(np.multiply(Y[:, 0], Z[:, 1]), X[:, 2])
        ret += np.multiply(np.multiply(Z[:, 0], X[:, 1]), Y[:, 2])
        ret -= np.multiply(np.multiply(Z[:, 0], Y[:, 1]), X[:, 2])
        ret -= np.multiply(np.multiply(Y[:, 0], X[:, 1]), Z[:, 2])
        ret -= np.multiply(np.multiply(X[:, 0], Z[:, 1]), Y[:, 2])
        return ret

    # One generalized winding number per input vertex
    ret = np.zeros(X_in.shape[0], dtype=X_in.dtype)

    # Accumulate generalized winding number for each triangle
    for U, V, W in triangles:
        A, B, C = U - X_in, V - X_in, W - X_in
        omega = adet(A, B, C)

        a, b, c = anorm2(A), anorm2(B), anorm2(C)
        k = a * b * c
        k += c * np.sum(np.multiply(A, B), axis=1)
        k += a * np.sum(np.multiply(B, C), axis=1)
        k += b * np.sum(np.multiply(C, A), axis=1)

        ret += np.arctan2(omega, k)
    return ret / (2 * np.pi) >= 1 - tol


def get_ism_order(n_faces: int, wanted_sources: int) -> int:
    """Calculate image source model reflection order for target source count.

    Computes the ISM order that approximates the desired number of image sources,
    assuming exponential growth with the number of reflections.

    Args:
        n_faces: Number of reflecting surfaces in the room.
        wanted_sources: Target number of image sources to generate.

    Returns:
        int: Reflection order (minimum 1).
    """
    order = np.floor(np.log(wanted_sources) / np.log(n_faces))
    return max(int(order), 1)


def cross_fade(size: int, method: str = "exp") -> torch.Tensor:
    """Generate a cross-fade curve for smooth transitions.

    Creates a normalized curve (values in [0, 1]) for blending between two signals
    using various windowing functions.

    Args:
        size: Length of the cross-fade curve in samples.
        method: Type of cross-fade function. Options:
            - "linear": Linear interpolation from 0 to 1.
            - "exp": Exponential decay (default).
            - "inv_exp": Inverse exponential.
            - "square": Quadratic curve.
            - "sine": Sinusoidal (half-period).

    Returns:
        torch.Tensor: Cross-fade curve of shape (size,) with values in [0, 1].

    Raises:
        ValueError: If method is not recognized.
    """
    match method:
        case "linear":
            return torch.linspace(0, 1, size)
        case "exp":
            return torch.exp(torch.linspace(-5, 0, size))
        case "inv_exp":
            return 1 - torch.exp(-10 * torch.linspace(0, 1, size))
        case "square":
            return torch.linspace(0, 1, size).pow(2)
        case "sine":
            tmp = torch.pi * torch.linspace(-0.5, 0.5, size)
            return (1 + tmp.sin()) / 2
        case _:
            raise ValueError(f"Unknown cross fade method {method}.")
