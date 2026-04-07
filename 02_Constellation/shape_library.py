"""
Shape Library for Constellation System
Defines target shapes that the morphogenic mesh can morph into.
Each shape is defined as a function that generates a wave field configuration.
"""

import numpy as np
import math
import numba

GRID_SIZE = 32
CENTER_X = GRID_SIZE // 2
CENTER_Y = GRID_SIZE // 2
CENTER_Z = GRID_SIZE // 2


@numba.jit(nopython=True)
def normalize_coords(x, y, z):
    """Normalize grid coordinates to -1 to 1 range."""
    nx = (x - CENTER_X) / (GRID_SIZE / 2.0)
    ny = (y - CENTER_Y) / (GRID_SIZE / 2.0)
    nz = (z - CENTER_Z) / (GRID_SIZE / 2.0)
    return nx, ny, nz


@numba.jit(nopython=True)
def sphere_shape(x, y, z, radius=0.7):
    """Generate sphere shape field."""
    nx, ny, nz = normalize_coords(x, y, z)
    dist = math.sqrt(nx*nx + ny*ny + nz*nz)
    # Sphere: distance from center
    shape_value = 1.0 - (dist / radius)
    return max(0.0, min(1.0, shape_value))


@numba.jit(nopython=True)
def cube_shape(x, y, z, size=0.6):
    """Generate cube shape field."""
    nx, ny, nz = normalize_coords(x, y, z)
    # Cube: max distance from center in any axis
    max_dist = max(abs(nx), abs(ny), abs(nz))
    shape_value = 1.0 - (max_dist / size)
    return max(0.0, min(1.0, shape_value))


@numba.jit(nopython=True)
def pyramid_shape(x, y, z, base_size=0.7, height=0.8):
    """Generate pyramid shape field (square base, pointed top)."""
    nx, ny, nz = normalize_coords(x, y, z)
    # Pyramid: square base in XZ, point at top
    base_dist = max(abs(nx), abs(nz))
    # Height decreases from top (ny = +1) to base (ny = -1)
    normalized_y = (ny + 1.0) / 2.0  # 0 to 1, where 1 is top
    # Base size decreases linearly from base to top
    current_base = base_size * (1.0 - normalized_y * 0.8)
    shape_value = 1.0 - (base_dist / current_base) if current_base > 0 else 0.0
    # Also check height bounds
    if ny > height or ny < -0.5:
        shape_value = 0.0
    return max(0.0, min(1.0, shape_value))


@numba.jit(nopython=True)
def torus_shape(x, y, z, major_radius=0.5, minor_radius=0.3):
    """Generate torus (donut) shape field."""
    nx, ny, nz = normalize_coords(x, y, z)
    # Torus: distance from a circle in XZ plane
    dist_from_center = math.sqrt(nx*nx + nz*nz)
    # Distance from the torus ring
    dist_from_ring = abs(dist_from_center - major_radius)
    # Combine with Y distance
    total_dist = math.sqrt(dist_from_ring*dist_from_ring + ny*ny)
    shape_value = 1.0 - (total_dist / minor_radius)
    return max(0.0, min(1.0, shape_value))


@numba.jit(nopython=True)
def cylinder_shape(x, y, z, radius=0.5, height=0.8):
    """Generate cylinder shape field."""
    nx, ny, nz = normalize_coords(x, y, z)
    # Cylinder: circular in XZ, bounded in Y
    dist_from_center = math.sqrt(nx*nx + nz*nz)
    # Check radius
    radial_dist = 1.0 - (dist_from_center / radius)
    # Check height bounds
    height_dist = 1.0 - (abs(ny) / height)
    shape_value = min(radial_dist, height_dist)
    return max(0.0, min(1.0, shape_value))


@numba.jit(nopython=True)
def octahedron_shape(x, y, z, size=0.6):
    """Generate octahedron (diamond) shape field."""
    nx, ny, nz = normalize_coords(x, y, z)
    # Octahedron: sum of absolute coordinates
    manhattan_dist = abs(nx) + abs(ny) + abs(nz)
    shape_value = 1.0 - (manhattan_dist / size)
    return max(0.0, min(1.0, shape_value))


def generate_shape_field(shape_func, offset_x=0.0, offset_y=0.0, offset_z=0.0):
    """
    Generate a 3D field for a shape at a specific position.
    Returns a 32x32x32 array with shape values.
    """
    field = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            for z in range(GRID_SIZE):
                # Apply offset to coordinates
                offset_x_local = x - CENTER_X - offset_x
                offset_y_local = y - CENTER_Y - offset_y
                offset_z_local = z - CENTER_Z - offset_z
                
                # Normalize with offset
                nx = offset_x_local / (GRID_SIZE / 2.0)
                ny = offset_y_local / (GRID_SIZE / 2.0)
                nz = offset_z_local / (GRID_SIZE / 2.0)
                
                # Calculate shape value
                field[x, y, z] = shape_func(x, y, z)
    
    return field


# Shape function registry
SHAPE_FUNCTIONS = {
    'sphere': sphere_shape,
    'cube': cube_shape,
    'pyramid': pyramid_shape,
    'torus': torus_shape,
    'cylinder': cylinder_shape,
    'octahedron': octahedron_shape,
}


def get_shape_function(shape_name):
    """Get shape function by name."""
    return SHAPE_FUNCTIONS.get(shape_name.lower(), sphere_shape)


def get_wireframe_vertices(shape_name, position, scale=1.0):
    """
    Generate wireframe vertices for a shape (for preview rendering).
    Returns array of line segment vertices.
    """
    # This is a simplified version - full implementation would generate
    # actual wireframe geometry for each shape type
    # For now, return empty array (will be implemented with proper wireframe generation)
    return np.array([], dtype=np.float32).reshape(0, 3)
