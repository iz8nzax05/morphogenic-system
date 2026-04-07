"""
Morphogenic Constellation System
Venom-style flowing mesh that morphs between wireframe target shapes arranged in a circle.
Extends the existing morphogenic voxel system with multi-target morphing and spatial movement.
"""

import numpy as np
import numba
import pygame
import math
import random
from pygame.locals import *
import moderngl
from pathlib import Path
from shape_library import (
    GRID_SIZE as BASE_GRID_SIZE
)
from wireframe_generator import get_wireframe_for_shape
try:
    from compute_field_generator import ComputeFieldGenerator
    COMPUTE_SHADERS_AVAILABLE = True
except ImportError:
    COMPUTE_SHADERS_AVAILABLE = False
    print("[WARNING] Compute shader module not available, using CPU generation")

# Constants
WINDOW_WIDTH = 1600  # Increased width to fit more UI text
WINDOW_HEIGHT = 900  # Increased height for better visibility
CONSTELLATION_RADIUS = 40.0  # Distance from center to target shapes
NUM_TARGETS = 3  # Number of target shapes in constellation (sphere, cube, pyramid)
BASE_GRID_SIZE = 32  # Base grid size (used as reference)
MIN_GRID_SIZE = 16  # Minimum grid size (lower quality, larger cubes)
MAX_GRID_SIZE = 128  # Maximum grid size (higher quality, smaller cubes) - increased for higher quality

# Default wave parameters
DEFAULT_FREQUENCY = 0.15
DEFAULT_AMPLITUDE = 0.5
DEFAULT_PHASE_SPEED = 0.02


@numba.jit(nopython=True)
def calculate_wave_field_3d(grid_size, time, frequency, amplitude, phase_speed):
    """
    Calculate 3D sine wave field that controls voxel spawn probability.
    Creates interference patterns from multiple wave directions.
    """
    field = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    phase = time * phase_speed
    center = grid_size // 2
    
    for x in range(grid_size):
        for y in range(grid_size):
            for z in range(grid_size):
                # Normalize coordinates to -1 to 1 range
                nx = (x - center) / (grid_size / 2.0)
                ny = (y - center) / (grid_size / 2.0)
                nz = (z - center) / (grid_size / 2.0)
                
                # Distance from center (for sphere-like shape)
                dist = math.sqrt(nx*nx + ny*ny + nz*nz)
                
                # Multiple sine waves in different directions (interference)
                wave1 = math.sin((nx * frequency + phase) * math.pi)
                wave2 = math.sin((ny * frequency + phase * 0.7) * math.pi)
                wave3 = math.sin((nz * frequency + phase * 1.3) * math.pi)
                
                # Combined wave interference
                combined_wave = (wave1 + wave2 + wave3) / 3.0
                
                # Distance-based sphere with wave modulation
                sphere_shape = 1.0 - (dist / 1.5)  # Sphere radius ~1.5
                sphere_shape = max(0.0, min(1.0, sphere_shape))
                
                # Combine shape with wave field
                field[x, y, z] = sphere_shape + combined_wave * amplitude
                
    return field


@numba.jit(nopython=True)
def shape_field_direct(nx, ny, nz, shape_type):
    """
    Calculate shape field value directly from normalized coordinates.
    Enhanced for better shape definition.
    shape_type: 0=sphere, 1=cube, 2=pyramid, 3=torus, 4=cylinder, 5=octahedron
    """
    dist = math.sqrt(nx*nx + ny*ny + nz*nz)
    
    if shape_type == 0:  # sphere - make more spherical
        radius = 0.65  # Slightly smaller for sharper edges
        shape_value = 1.0 - (dist / radius)
        # Sharpen the edge slightly for better definition
        if shape_value > 0.5:
            shape_value = 0.5 + (shape_value - 0.5) * 1.2
    elif shape_type == 1:  # cube - make more cube-like with sharper edges
        size = 0.55  # Slightly smaller for sharper edges
        max_dist = max(abs(nx), abs(ny), abs(nz))
        shape_value = 1.0 - (max_dist / size)
        # Sharpen cube edges for better definition
        if shape_value > 0.3:
            shape_value = 0.3 + (shape_value - 0.3) * 1.4
    elif shape_type == 2:  # pyramid - make more pyramid-like
        base_size = 0.65
        height = 0.85
        base_dist = max(abs(nx), abs(nz))
        normalized_y = (ny + 1.0) / 2.0
        current_base = base_size * (1.0 - normalized_y * 0.85)
        if current_base > 0 and ny <= height and ny >= -0.5:
            shape_value = 1.0 - (base_dist / current_base)
            # Sharpen pyramid for better definition
            if shape_value > 0.3:
                shape_value = 0.3 + (shape_value - 0.3) * 1.3
        else:
            shape_value = 0.0
    elif shape_type == 3:  # torus
        major_radius = 0.5
        minor_radius = 0.3
        dist_from_center = math.sqrt(nx*nx + nz*nz)
        dist_from_ring = abs(dist_from_center - major_radius)
        total_dist = math.sqrt(dist_from_ring*dist_from_ring + ny*ny)
        shape_value = 1.0 - (total_dist / minor_radius)
    elif shape_type == 4:  # cylinder
        radius = 0.5
        height = 0.8
        dist_from_center = math.sqrt(nx*nx + nz*nz)
        radial_dist = 1.0 - (dist_from_center / radius)
        height_dist = 1.0 - (abs(ny) / height)
        shape_value = min(radial_dist, height_dist)
    else:  # octahedron (5)
        size = 0.6
        manhattan_dist = abs(nx) + abs(ny) + abs(nz)
        shape_value = 1.0 - (manhattan_dist / size)
    
    return max(0.0, min(1.0, shape_value))


@numba.jit(nopython=True)
def generate_shape_field_at_position(grid_size, shape_type, world_pos_x, world_pos_y, world_pos_z):
    """
    Generate shape field centered at a world position.
    This is numba-compiled for performance.
    shape_type: 0=sphere, 1=cube, 2=pyramid, 3=torus, 4=cylinder, 5=octahedron
    """
    field = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    center = grid_size // 2
    
    for x in range(grid_size):
        for y in range(grid_size):
            for z in range(grid_size):
                # Calculate position relative to shape center
                rel_x = float(x) - world_pos_x
                rel_y = float(y) - world_pos_y
                rel_z = float(z) - world_pos_z
                
                # Normalize to -1 to 1 range
                nx = rel_x / (grid_size / 2.0)
                ny = rel_y / (grid_size / 2.0)
                nz = rel_z / (grid_size / 2.0)
                
                # Calculate distance for early exit optimization
                dist = math.sqrt(nx*nx + ny*ny + nz*nz)
                if dist > 1.5:
                    field[x, y, z] = 0.0
                    continue
                
                # Calculate shape value directly
                field[x, y, z] = shape_field_direct(nx, ny, nz, shape_type)
    
    return field


@numba.jit(nopython=True)
def smoothstep(t):
    """Smoothstep function for organic transitions: t^2 * (3 - 2*t)"""
    return t * t * (3.0 - 2.0 * t)

@numba.jit(nopython=True)
def organic_blend_factor(t):
    """Organic blending curve - smooth at ends, faster in middle"""
    # Use smoothstep for organic ease-in-out
    return smoothstep(t)

@numba.jit(nopython=True)
def blend_shape_fields(field1, field2, blend_factor):
    """Blend two shape fields together with organic transition."""
    # blend_factor: 0.0 = field1, 1.0 = field2
    # Apply organic easing curve for smoother transitions
    organic_t = organic_blend_factor(blend_factor)
    result = field1 * (1.0 - organic_t) + field2 * organic_t
    return result


@numba.jit(nopython=True)
def apply_shape_field_to_wave(wave_field, shape_field, influence=0.5):
    """Apply shape field as a mask/modifier to the wave field."""
    # Combine wave field with shape field
    # influence: 0.0 = pure wave, 1.0 = pure shape
    # Enhanced blending for better shape definition
    # Use shape field more directly for sharper shapes
    
    result = np.zeros_like(wave_field)
    
    for x in range(wave_field.shape[0]):
        for y in range(wave_field.shape[1]):
            for z in range(wave_field.shape[2]):
                # Boost shape field slightly for better definition
                shape_val = shape_field[x, y, z] * (1.0 + influence * 0.2)
                # Clamp shape value
                if shape_val > 1.0:
                    shape_val = 1.0
                elif shape_val < 0.0:
                    shape_val = 0.0
                
                # Blend: at high influence, use shape field more directly
                # This creates sharper shapes when influence is high
                wave_val = wave_field[x, y, z]
                blended = wave_val * (1.0 - influence) + (wave_val * shape_val + shape_val * 0.3) * influence
                result[x, y, z] = blended
    
    return result


@numba.jit(nopython=True)
def update_voxel_grid(wave_field, previous_grid, threshold, feedback_strength):
    """
    Update voxel grid based on wave field and feedback from previous state.
    Spawn/despawn voxels based on field values with mathematical feedback.
    """
    grid_size = wave_field.shape[0]
    new_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.bool_)
    
    for x in range(grid_size):
        for y in range(grid_size):
            for z in range(grid_size):
                # Get wave field value
                field_value = wave_field[x, y, z]
                
                # Feedback: previous voxel state influences threshold
                feedback_modifier = 0.0
                if previous_grid[x, y, z]:
                    feedback_modifier = feedback_strength * 0.2  # Existing voxels more likely to persist
                
                # Adjusted threshold with feedback
                adjusted_threshold = threshold - feedback_modifier
                
                # Determine voxel existence
                if field_value > adjusted_threshold:
                    new_grid[x, y, z] = True
                    
    return new_grid


@numba.jit(nopython=True)
def get_visible_faces(voxel_grid, x, y, z):
    """
    Check which faces of a voxel are visible (not hidden by adjacent voxels).
    Returns a 6-element boolean array: [top, bottom, front, back, right, left]
    """
    faces = np.zeros(6, dtype=np.bool_)
    grid_size = voxel_grid.shape[0]
    
    # Top face (y+1)
    if y + 1 >= grid_size or not voxel_grid[x, y + 1, z]:
        faces[0] = True
    
    # Bottom face (y-1)
    if y - 1 < 0 or not voxel_grid[x, y - 1, z]:
        faces[1] = True
    
    # Front face (z+1)
    if z + 1 >= grid_size or not voxel_grid[x, y, z + 1]:
        faces[2] = True
    
    # Back face (z-1)
    if z - 1 < 0 or not voxel_grid[x, y, z - 1]:
        faces[3] = True
    
    # Right face (x+1)
    if x + 1 >= grid_size or not voxel_grid[x + 1, y, z]:
        faces[4] = True
    
    # Left face (x-1)
    if x - 1 < 0 or not voxel_grid[x - 1, y, z]:
        faces[5] = True
    
    return faces


@numba.jit(nopython=True)
def is_face_facing_camera(face_normal_x, face_normal_y, face_normal_z, 
                          voxel_x, voxel_y, voxel_z,
                          camera_x, camera_y, camera_z):
    """
    Check if a face is facing the camera using camera-to-face direction.
    Returns True if face is facing camera (visible from camera position).
    
    Face normal points OUTWARD from voxel (e.g., top face normal = (0, 1, 0) pointing up).
    Direction from camera TO face: (face_pos - camera_pos) points toward face.
    
    Dot product: face_normal · (face_pos - camera_pos)
    - If > 0: face normal points toward camera → face IS facing camera ✓
    - If < 0: face normal points away from camera → face is facing away ✗
    """
    # Calculate direction from camera to face center (voxel position)
    dx = voxel_x - camera_x
    dy = voxel_y - camera_y
    dz = voxel_z - camera_z
    
    # Normalize direction (with safety check for zero distance)
    dist_sq = dx*dx + dy*dy + dz*dz
    if dist_sq < 0.0001:  # Very close or same position
        return True  # Consider visible if camera is inside voxel
    
    inv_dist = 1.0 / math.sqrt(dist_sq)
    camera_to_face_x = dx * inv_dist  # Normalized vector from camera to face
    camera_to_face_y = dy * inv_dist
    camera_to_face_z = dz * inv_dist
    
    # Dot product of face normal with camera-to-face direction
    # Positive dot = face normal points toward camera (face is facing camera) ✓
    # Negative dot = face normal points away from camera (face is facing away) ✗
    dot_product = face_normal_x * camera_to_face_x + face_normal_y * camera_to_face_y + face_normal_z * camera_to_face_z
    
    # Face is facing camera if dot product is POSITIVE (normal points toward camera)
    return dot_product > 0.0


@numba.jit(nopython=True)
def get_camera_facing_faces(voxel_grid, x, y, z, 
                            voxel_world_x, voxel_world_y, voxel_world_z,
                            camera_x, camera_y, camera_z,
                            spacing_scale=1.0):
    """
    Check which faces of a voxel are visible AND facing the camera.
    Uses actual camera position and voxel world position for accurate culling.
    Returns a 6-element boolean array: [top, bottom, front, back, right, left]
    Face normals: top=(0,1,0), bottom=(0,-1,0), front=(0,0,1), back=(0,0,-1), right=(1,0,0), left=(-1,0,0)
    spacing_scale: Scale factor for face offset (accounts for voxel size)
    """
    faces = get_visible_faces(voxel_grid, x, y, z)
    camera_facing = np.zeros(6, dtype=np.bool_)
    
    # Face center offsets (scaled by spacing_scale to account for voxel size)
    # For unit cube, face center is 0.5 units from voxel center
    face_offset = 0.5 * spacing_scale
    
    # Check each visible face if it's facing the camera
    # Top face (normal: 0, 1, 0) - center at (voxel_x, voxel_y + offset, voxel_z)
    if faces[0]:
        if is_face_facing_camera(0.0, 1.0, 0.0, 
                                voxel_world_x, voxel_world_y + face_offset, voxel_world_z,
                                camera_x, camera_y, camera_z):
            camera_facing[0] = True
    
    # Bottom face (normal: 0, -1, 0) - center at (voxel_x, voxel_y - offset, voxel_z)
    if faces[1]:
        if is_face_facing_camera(0.0, -1.0, 0.0, 
                                voxel_world_x, voxel_world_y - face_offset, voxel_world_z,
                                camera_x, camera_y, camera_z):
            camera_facing[1] = True
    
    # Front face (normal: 0, 0, 1) - center at (voxel_x, voxel_y, voxel_z + offset)
    if faces[2]:
        if is_face_facing_camera(0.0, 0.0, 1.0, 
                                voxel_world_x, voxel_world_y, voxel_world_z + face_offset,
                                camera_x, camera_y, camera_z):
            camera_facing[2] = True
    
    # Back face (normal: 0, 0, -1) - center at (voxel_x, voxel_y, voxel_z - offset)
    if faces[3]:
        if is_face_facing_camera(0.0, 0.0, -1.0, 
                                voxel_world_x, voxel_world_y, voxel_world_z - face_offset,
                                camera_x, camera_y, camera_z):
            camera_facing[3] = True
    
    # Right face (normal: 1, 0, 0) - center at (voxel_x + offset, voxel_y, voxel_z)
    if faces[4]:
        if is_face_facing_camera(1.0, 0.0, 0.0, 
                                voxel_world_x + face_offset, voxel_world_y, voxel_world_z,
                                camera_x, camera_y, camera_z):
            camera_facing[4] = True
    
    # Left face (normal: -1, 0, 0) - center at (voxel_x - offset, voxel_y, voxel_z)
    if faces[5]:
        if is_face_facing_camera(-1.0, 0.0, 0.0, 
                                voxel_world_x - face_offset, voxel_world_y, voxel_world_z,
                                camera_x, camera_y, camera_z):
            camera_facing[5] = True
    
    return camera_facing


def create_cube_geometry():
    """
    Create cube geometry with vertices, normals, face IDs, and indices.
    Each face has 4 vertices, 6 faces total = 24 vertices.
    """
    # Cube vertices (8 corners)
    vertices = np.array([
        # Position (x, y, z), Normal (nx, ny, nz), FaceID
        # Top face (y = +0.5, face ID 0)
        [-0.5,  0.5, -0.5,  0.0,  1.0,  0.0,  0.0],  # 0
        [ 0.5,  0.5, -0.5,  0.0,  1.0,  0.0,  0.0],  # 1
        [ 0.5,  0.5,  0.5,  0.0,  1.0,  0.0,  0.0],  # 2
        [-0.5,  0.5,  0.5,  0.0,  1.0,  0.0,  0.0],  # 3
        
        # Bottom face (y = -0.5, face ID 1)
        [-0.5, -0.5, -0.5,  0.0, -1.0,  0.0,  1.0],  # 4
        [ 0.5, -0.5, -0.5,  0.0, -1.0,  0.0,  1.0],  # 5
        [ 0.5, -0.5,  0.5,  0.0, -1.0,  0.0,  1.0],  # 6
        [-0.5, -0.5,  0.5,  0.0, -1.0,  0.0,  1.0],  # 7
        
        # Front face (z = +0.5, face ID 2)
        [-0.5, -0.5,  0.5,  0.0,  0.0,  1.0,  2.0],  # 8
        [ 0.5, -0.5,  0.5,  0.0,  0.0,  1.0,  2.0],  # 9
        [ 0.5,  0.5,  0.5,  0.0,  0.0,  1.0,  2.0],  # 10
        [-0.5,  0.5,  0.5,  0.0,  0.0,  1.0,  2.0],  # 11
        
        # Back face (z = -0.5, face ID 3)
        [-0.5, -0.5, -0.5,  0.0,  0.0, -1.0,  3.0],  # 12
        [ 0.5, -0.5, -0.5,  0.0,  0.0, -1.0,  3.0],  # 13
        [ 0.5,  0.5, -0.5,  0.0,  0.0, -1.0,  3.0],  # 14
        [-0.5,  0.5, -0.5,  0.0,  0.0, -1.0,  3.0],  # 15
        
        # Right face (x = +0.5, face ID 4)
        [ 0.5, -0.5, -0.5,  1.0,  0.0,  0.0,  4.0],  # 16
        [ 0.5, -0.5,  0.5,  1.0,  0.0,  0.0,  4.0],  # 17
        [ 0.5,  0.5,  0.5,  1.0,  0.0,  0.0,  4.0],  # 18
        [ 0.5,  0.5, -0.5,  1.0,  0.0,  0.0,  4.0],  # 19
        
        # Left face (x = -0.5, face ID 5)
        [-0.5, -0.5, -0.5, -1.0,  0.0,  0.0,  5.0],  # 20
        [-0.5, -0.5,  0.5, -1.0,  0.0,  0.0,  5.0],  # 21
        [-0.5,  0.5,  0.5, -1.0,  0.0,  0.0,  5.0],  # 22
        [-0.5,  0.5, -0.5, -1.0,  0.0,  0.0,  5.0],  # 23
    ], dtype=np.float32)
    
    # Indices for 6 faces (each face = 2 triangles = 6 indices)
    # FINAL WORKING CONFIGURATION: All faces now render correctly with front_face='cw'
    # Top, Back, Right: Original CW winding (0,1,2 / 12,13,14 / 16,17,18)
    # Bottom, Front, Left: Flipped to CCW winding (4,7,6 / 8,11,10 / 20,23,22)
    # This mixed winding order works correctly with GPU back-face culling
    indices = np.array([
        # Top face (y = +0.5, normal = (0, 1, 0)) - ORIGINAL CW - WORKING
        0, 1, 2,  0, 2, 3,
        
        # Bottom face (y = -0.5, normal = (0, -1, 0)) - FLIPPED - WORKING
        4, 7, 6,  4, 6, 5,
        
        # Front face (z = +0.5, normal = (0, 0, 1)) - FLIPPED - WORKING
        8, 11, 10,  8, 10, 9,
        
        # Back face (z = -0.5, normal = (0, 0, -1)) - ORIGINAL CW - WORKING
        12, 13, 14,  12, 14, 15,
        
        # Right face (x = +0.5, normal = (1, 0, 0)) - ORIGINAL CW - WORKING
        16, 17, 18,  16, 18, 19,
        
        # Left face (x = -0.5, normal = (-1, 0, 0)) - FLIPPED - WORKING
        20, 23, 22,  20, 22, 21,
    ], dtype=np.uint32)
    
    return vertices, indices


def generate_instances(voxel_grid, world_offset, spacing_scale=1.0, camera_pos=None):
    """
    Generate instance positions for visible voxels with proper camera-facing face culling.
    Returns array of (x, y, z) positions for visible voxels.
    world_offset: 3D position offset for the entire grid in world space
    spacing_scale: Scale factor for spacing between cubes (1.0 = normal, <1.0 = closer, >1.0 = farther)
    camera_pos: Camera position in world space (3D array) for proper face culling
    """
    instances = []
    grid_size = voxel_grid.shape[0]
    
    # If camera position not provided, use basic face culling (no camera-facing check)
    if camera_pos is None:
        for x in range(grid_size):
            for y in range(grid_size):
                for z in range(grid_size):
                    if voxel_grid[x, y, z]:
                        # Check if any face is visible (not occluded by adjacent voxels)
                        faces = get_visible_faces(voxel_grid, x, y, z)
                        if np.any(faces):
                            # Scale spacing to match cube size, then apply world offset
                            world_x = float(x) * spacing_scale + world_offset[0]
                            world_y = float(y) * spacing_scale + world_offset[1]
                            world_z = float(z) * spacing_scale + world_offset[2]
                            instances.append([world_x, world_y, world_z])
    else:
        # Use proper camera-facing face culling (uses actual camera position)
        camera_x = float(camera_pos[0])
        camera_y = float(camera_pos[1])
        camera_z = float(camera_pos[2])
        
        for x in range(grid_size):
            for y in range(grid_size):
                for z in range(grid_size):
                    if voxel_grid[x, y, z]:
                        # Calculate world position of this voxel
                        world_x = float(x) * spacing_scale + world_offset[0]
                        world_y = float(y) * spacing_scale + world_offset[1]
                        world_z = float(z) * spacing_scale + world_offset[2]
                        
                        # Check if any face is visible AND facing the camera
                        camera_facing_faces = get_camera_facing_faces(
                            voxel_grid, x, y, z,
                            world_x, world_y, world_z,  # World position of voxel
                            camera_x, camera_y, camera_z,  # Camera position
                            spacing_scale  # Scale factor for face offset
                        )
                        if np.any(camera_facing_faces):
                            instances.append([world_x, world_y, world_z])
    
    if len(instances) == 0:
        return np.array([], dtype=np.float32).reshape(0, 3)
    
    return np.array(instances, dtype=np.float32)


def create_view_matrix_free_look(camera_pos, forward, up):
    """
    Create view matrix for fully free 3D camera.
    camera_pos: position of camera in world space
    forward: normalized forward direction vector
    up: normalized up direction vector (world up, used to calculate right)
    """
    # Calculate right vector (perpendicular to forward and up)
    right = np.cross(forward, up)
    right_len = np.linalg.norm(right)
    if right_len < 0.0001:
        # If forward and up are parallel, use default right
        right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        right = right / right_len
    
    # Recalculate up to ensure orthonormal basis
    new_up = np.cross(right, forward)
    new_up = new_up / np.linalg.norm(new_up)
    
    # Create view matrix (inverse of camera transform)
    # This rotates and translates the world to camera space
    view = np.eye(4, dtype=np.float32)
    
    # Rotation part (transpose = inverse for orthonormal matrix)
    view[0, 0] = right[0]
    view[1, 0] = right[1]
    view[2, 0] = right[2]
    
    view[0, 1] = new_up[0]
    view[1, 1] = new_up[1]
    view[2, 1] = new_up[2]
    
    view[0, 2] = -forward[0]
    view[1, 2] = -forward[1]
    view[2, 2] = -forward[2]
    
    # Translation part (translate by negative camera position)
    view[3, 0] = -np.dot(right, camera_pos)
    view[3, 1] = -np.dot(new_up, camera_pos)
    view[3, 2] = np.dot(forward, camera_pos)
    
    return view


def create_projection_matrix(fov, aspect, near, far):
    """Create perspective projection matrix."""
    f = 1.0 / math.tan(fov / 2.0)
    proj = np.zeros((4, 4), dtype=np.float32)
    
    proj[0, 0] = f / aspect
    proj[1, 1] = f
    proj[2, 2] = (far + near) / (near - far)
    proj[2, 3] = -1.0
    proj[3, 2] = (2.0 * far * near) / (near - far)
    
    return proj


def generate_constellation_positions(num_targets, radius, center_height=0.0):
    """
    Generate positions for target shapes arranged in a circle.
    Returns list of (x, y, z) positions.
    """
    positions = []
    angle_step = (2.0 * math.pi) / num_targets
    
    for i in range(num_targets):
        angle = i * angle_step
        x = radius * math.cos(angle)
        y = center_height
        z = radius * math.sin(angle)
        positions.append((x, y, z))
    
    return positions


def interpolate_position(pos1, pos2, t):
    """Interpolate between two 3D positions."""
    return (
        pos1[0] * (1.0 - t) + pos2[0] * t,
        pos1[1] * (1.0 - t) + pos2[1] * t,
        pos1[2] * (1.0 - t) + pos2[2] * t
    )


class ConstellationSystem:
    def __init__(self):
        pygame.init()
        pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF)
        pygame.display.set_caption("Morphogenic Constellation System")
        
        # Create ModernGL context
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST)
        
        # GPU back-face culling setup (CRITICAL: Required for correct face visibility)
        # FINAL FIX: With front_face='cw', some faces need original CW winding (Top, Back, Right),
        # while others need flipped CCW winding (Bottom, Front, Left) to render correctly
        self.face_culling_enabled = True  # Debug flag to toggle face culling (C key)
        self.ctx.enable(moderngl.CULL_FACE)  # Enable GPU back-face culling
        self.ctx.front_face = 'cw'  # Set CW (clockwise) as front-facing
        self.ctx.cull_face = 'back'  # Cull back-facing faces (default, but explicit for clarity)
        # GPU back-face culling will automatically hide back-facing faces based on view matrix
        # This works correctly with free-look camera - faces update automatically as camera rotates
        # Press C key to toggle face culling ON/OFF to verify it's working
        
        self.ctx.clear_color = (0.05, 0.05, 0.1, 1.0)
        
        # Load shaders
        shader_dir = Path(__file__).parent / "shaders"
        shader_dir.mkdir(exist_ok=True)
        
        # Try to load shaders, fallback to inline
        try:
            with open(shader_dir / "constellation_shaders.vert") as f:
                vertex_shader = f.read()
            with open(shader_dir / "constellation_shaders.frag") as f:
                fragment_shader = f.read()
        except FileNotFoundError:
            # Use same shaders as foundation system
            vertex_shader = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in float aFaceID;
layout(location = 3) in vec3 aInstancePos;
uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;
uniform float voxelScale;  // Quality control: scale of individual voxels
out vec3 FragPos;
out vec3 FragNormal;
out float FaceID;
void main() {
    // Scale voxel size by quality parameter
    vec3 scaledPos = aPos * voxelScale;
    vec3 worldPos = scaledPos + aInstancePos;
    vec4 pos = vec4(worldPos, 1.0);
    gl_Position = projection * view * model * pos;
    FragPos = worldPos;
    FragNormal = aNormal;
    FaceID = aFaceID;
}
"""
            fragment_shader = """
#version 330 core
in vec3 FragPos;
in vec3 FragNormal;
in float FaceID;
out vec4 FragColor;
uniform vec3 lightDir;
uniform vec3 lightColor;
uniform float ambientStrength;
uniform float diffuseStrength;
vec3 getFaceColor(float faceID) {
    if (faceID < 0.5) return vec3(1.0, 0.3, 0.3);
    else if (faceID < 1.5) return vec3(0.3, 0.1, 0.1);
    else if (faceID < 2.5) return vec3(0.3, 1.0, 0.3);
    else if (faceID < 3.5) return vec3(0.1, 0.3, 0.1);
    else if (faceID < 4.5) return vec3(0.3, 0.3, 1.0);
    else return vec3(0.1, 0.1, 0.3);
}
void main() {
    vec3 faceColor = getFaceColor(FaceID);
    vec3 norm = normalize(FragNormal);
    vec3 lightDirection = normalize(lightDir);
    vec3 ambient = ambientStrength * lightColor;
    float diff = max(dot(norm, lightDirection), 0.0);
    vec3 diffuse = diffuseStrength * diff * lightColor;
    vec3 result = (ambient + diffuse) * faceColor;
    result = pow(result, vec3(1.0/2.2));
    FragColor = vec4(result, 1.0);
}
"""
        
        self.program = self.ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
        
        # Wireframe shader program
        wireframe_vertex_shader = """
#version 330 core
layout(location = 0) in vec3 aPos;
uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;
void main() {
    vec4 pos = vec4(aPos, 1.0);
    gl_Position = projection * view * model * pos;
}
"""
        wireframe_fragment_shader = """
#version 330 core
out vec4 FragColor;
uniform vec3 wireframeColor;
void main() {
    FragColor = vec4(wireframeColor, 0.4);  // Semi-transparent wireframes
}
"""
        self.wireframe_program = self.ctx.program(
            vertex_shader=wireframe_vertex_shader,
            fragment_shader=wireframe_fragment_shader
        )
        
        # Create cube geometry
        vertices, indices = create_cube_geometry()
        
        # Create vertex buffer
        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.ibo = self.ctx.buffer(indices.tobytes())
        
        # Create instance buffer (use max grid size for allocation)
        max_instances = MAX_GRID_SIZE**3  # Allocate for maximum possible grid size
        max_buffer_size = max_instances * 3 * 4
        self.instance_buffer = self.ctx.buffer(reserve=max_buffer_size)
        
        dummy_instances = np.zeros((1, 3), dtype=np.float32)
        self.instance_buffer.write(dummy_instances.tobytes())
        
        # Create vertex array
        self.vao = self.ctx.vertex_array(
            self.program,
            [
                (self.vbo, '3f 3f 1f', 'aPos', 'aNormal', 'aFaceID'),
                (self.instance_buffer, '3f /i', 'aInstancePos'),
            ],
            self.ibo,
        )
        
        # Constellation setup - 3 simple shapes equally spaced
        shape_names = ['sphere', 'cube', 'pyramid']  # Only 3 simple shapes
        self.target_shapes = []
        # Initialize grid size first (before using in generate_constellation_positions)
        self.base_grid_size = BASE_GRID_SIZE
        self.min_grid_size = MIN_GRID_SIZE
        self.max_grid_size = MAX_GRID_SIZE
        self.grid_size = BASE_GRID_SIZE  # Initialize before use
        
        self.target_positions = generate_constellation_positions(
            NUM_TARGETS, 
            CONSTELLATION_RADIUS,
            center_height=float(self.grid_size // 2)  # Use dynamic grid center
        )
        
        # Shape type mapping: 0=sphere, 1=cube, 2=pyramid
        shape_type_map = {
            'sphere': 0,
            'cube': 1,
            'pyramid': 2
        }
        
        # Assign shapes to positions (one of each: sphere, cube, pyramid)
        self.wireframe_buffers = []
        self.wireframe_vaos = []
        for i, pos in enumerate(self.target_positions):
            shape_name = shape_names[i]  # Direct assignment, no cycling
            shape_type = shape_type_map.get(shape_name, 0)
            self.target_shapes.append({
                'name': shape_name,
                'position': pos,
                'shape_type': shape_type
            })
            
            # Generate wireframe vertices for this target
            wireframe_vertices = get_wireframe_for_shape(shape_type, pos, scale=1.0)
            if len(wireframe_vertices) > 0:
                wireframe_vbo = self.ctx.buffer(wireframe_vertices.tobytes())
                wireframe_vao = self.ctx.simple_vertex_array(
                    self.wireframe_program,
                    wireframe_vbo,
                    'aPos'
                )
                self.wireframe_buffers.append(wireframe_vbo)
                self.wireframe_vaos.append((wireframe_vao, len(wireframe_vertices)))
            else:
                self.wireframe_buffers.append(None)
                self.wireframe_vaos.append(None)
        
        # Active mesh state
        self.current_target_index = 0
        self.next_target_index = 1
        self.morph_progress = 0.0  # 0.0 = current target, 1.0 = next target (for shape blending)
        self.position_progress = 0.0  # 0.0 = current position, 1.0 = next position (for movement)
        self.morph_progress = 0.0  # 0.0 = current shape, 1.0 = next shape (for morphing)
        self.movement_speed = 0.03  # How fast to MOVE between target positions
        self.morph_speed = 0.03  # How fast to MORPH between target shapes (separate from movement)
        self.reached_threshold = 0.98  # When to select next target
        
        # Current active mesh position (interpolated between targets)
        self.active_mesh_position = np.array([
            self.target_positions[0][0],
            self.target_positions[0][1],
            self.target_positions[0][2]
        ], dtype=np.float32)
        
        # Voxel scale is calculated from grid size to maintain shape size (already initialized above)
        # When grid is larger, cubes must be smaller to fit same shape
        # When grid is smaller, cubes must be larger to fill same shape
        self.voxel_scale = float(self.base_grid_size) / float(self.grid_size)  # Calculate from grid_size
        self.base_threshold = 0.5  # Base threshold for density control
        
        # Flag to recreate grids when size changes
        self.grid_size_changed = False
        
        # Voxel grid (dynamic size based on quality)
        self.voxel_grid = np.zeros((self.grid_size, self.grid_size, self.grid_size), dtype=bool)
        self.previous_grid = np.zeros((self.grid_size, self.grid_size, self.grid_size), dtype=bool)
        
        # Time and animation
        self.time = 0.0
        self.running = True
        self.paused = False
        
        # Wave parameters
        self.frequency = DEFAULT_FREQUENCY
        self.amplitude = DEFAULT_AMPLITUDE
        self.phase_speed = DEFAULT_PHASE_SPEED
        self.threshold = 0.5
        self.feedback_strength = 0.5
        self.base_shape_influence = 0.85  # Base shape influence (increased for better shape definition)
        self.shape_influence = 0.85  # Current shape influence (dynamic based on proximity to targets)
        self.max_shape_influence = 0.95  # Maximum shape influence when at target positions
        self.transition_shape_influence = 0.70  # Lower shape influence during transitions for organic flow
        
        # Camera controls
        self.camera_pos = np.array([0.0, float(self.grid_size // 2) + 20.0, 100.0], dtype=np.float32)
        self.camera_yaw = math.radians(-90.0)
        self.camera_pitch = math.radians(-10.0)
        self.camera_move_speed = 30.0
        self.camera_look_speed = 0.5  # Reduced for smoother, less "world rotating" feel
        
        # Light direction
        self.light_dir = np.array([0.5, 0.8, -0.5], dtype=np.float32)
        self.light_dir = self.light_dir / np.linalg.norm(self.light_dir)
        
        # Performance tracking
        self.frame_count = 0
        self.fps = 0
        self.last_fps_update = pygame.time.get_ticks()
        self.voxel_count = 0
        
        # Debug feedback messages (display on screen)
        self.debug_message = ""
        self.debug_message_timer = 0.0
        self.debug_message_duration = 2.0  # Show message for 2 seconds
        
        # Phase 1: Compute shader support (optional, falls back to CPU)
        self.compute_generator = None
        if COMPUTE_SHADERS_AVAILABLE:
            try:
                self.compute_generator = ComputeFieldGenerator(self.ctx, self.grid_size)
                if not self.compute_generator.use_compute_shaders:
                    self.compute_generator = None  # Not supported, use CPU
            except Exception as e:
                print(f"[WARNING] Compute shader initialization failed: {e}")
                self.compute_generator = None
        
        # Phase 1: Profiling support
        self.profiling_enabled = False
        self.frame_times = []
        self.max_frame_time_history = 120  # Keep last 2 seconds at 60 FPS
    
    def generate_shape_field(self, shape_type, world_position):
        """
        Generate shape field for a target shape at a specific world position.
        Uses numba-compiled function for performance.
        """
        return generate_shape_field_at_position(
            self.grid_size,  # Pass current grid size
            shape_type,
            world_position[0],
            world_position[1],
            world_position[2]
        )
    
    
    def update_target_selection(self):
        """Update target selection when current target position is reached."""
        if self.position_progress >= self.reached_threshold:
            # Select next target (random, but not the same as current)
            available_indices = [i for i in range(len(self.target_shapes)) 
                               if i != self.current_target_index]
            self.current_target_index = self.next_target_index
            self.next_target_index = random.choice(available_indices)
            self.position_progress = 0.0
            self.morph_progress = 0.0  # Reset morph progress when moving to new target
    
    def handle_input(self):
        """Process keyboard and mouse input."""
        keys = pygame.key.get_pressed()
        
        # Wave parameter adjustments
        if keys[K_i]:
            self.frequency += 0.001
        if keys[K_k]:
            self.frequency -= 0.001
        if keys[K_j]:
            self.amplitude = max(0.0, self.amplitude - 0.01)
        if keys[K_l]:
            self.amplitude = min(1.0, self.amplitude + 0.01)
        if keys[K_u]:
            self.phase_speed -= 0.001
        if keys[K_o]:
            self.phase_speed += 0.001
        
        # Movement speed control (T = faster, G = slower)
        if keys[K_t]:
            self.movement_speed = min(0.2, self.movement_speed + 0.001)
        if keys[K_g]:
            self.movement_speed = max(0.001, self.movement_speed - 0.001)
        
        # Morph speed control (Y = faster, H = slower)
        if keys[K_y]:
            self.morph_speed = min(0.2, self.morph_speed + 0.001)
        if keys[K_h]:
            self.morph_speed = max(0.001, self.morph_speed - 0.001)
        
        # Quality control (N = higher quality/bigger grid/more cubes, M = lower quality/smaller grid/fewer cubes)
        # Higher quality (N): increase grid size (more cubes total) + smaller cubes + smaller spacing = same shape size
        # Lower quality (M): decrease grid size (fewer cubes total) + larger cubes + larger spacing = same shape size
        # Grid size directly controls total number of cubes - bigger grid = more resolution
        if keys[K_n]:  # Higher quality: increase grid size (more cubes)
            if self.grid_size < self.max_grid_size:
                self.grid_size += 4  # Increase grid size by 4 for faster quality changes
                self.grid_size = min(self.max_grid_size, self.grid_size)
                self.grid_size_changed = True
        if keys[K_m]:  # Lower quality: decrease grid size (fewer cubes)
            if self.grid_size > self.min_grid_size:
                self.grid_size -= 4  # Decrease grid size by 4 for faster quality changes
                self.grid_size = max(self.min_grid_size, self.grid_size)
                self.grid_size_changed = True
        
        # Calculate voxel scale from grid size to maintain shape size
        # When grid is larger, cubes must be smaller (scale down)
        # When grid is smaller, cubes must be larger (scale up)
        # Formula: scale = base_size / current_size (inverse relationship)
        self.voxel_scale = float(self.base_grid_size) / float(self.grid_size)
        
        # When grid size changes, recreate voxel grids with new size
        if self.grid_size_changed:
            self.voxel_grid = np.zeros((self.grid_size, self.grid_size, self.grid_size), dtype=bool)
            self.previous_grid = np.zeros((self.grid_size, self.grid_size, self.grid_size), dtype=bool)
            self.grid_size_changed = False
            # Phase 1: Resize compute generator if available
            if self.compute_generator:
                self.compute_generator.resize(self.grid_size)
        
        # Camera look (smooth rotation, independent of frame rate)
        dt = getattr(self, "_last_dt_seconds", 1.0 / 60.0)
        # Use fixed small increments for smoother, more predictable rotation
        look_delta = self.camera_look_speed * dt
        
        if keys[K_LEFT]:
            self.camera_yaw -= look_delta
        if keys[K_RIGHT]:
            self.camera_yaw += look_delta
        if keys[K_UP]:
            self.camera_pitch += look_delta  
        if keys[K_DOWN]:
            self.camera_pitch -= look_delta 
        
        # No pitch clamping - fully free rotation in 3D space
        # Allow full 360-degree rotation on all axes
        
        # Camera movement
        speed = self.camera_move_speed
        if keys[K_LSHIFT] or keys[K_RSHIFT]:
            speed *= 2.5
        if keys[K_LCTRL] or keys[K_RCTRL]:
            speed *= 0.4
        
        # Calculate camera forward direction from yaw/pitch (fully free rotation)
        fx = math.cos(self.camera_yaw) * math.cos(self.camera_pitch)
        fy = math.sin(self.camera_pitch)
        fz = math.sin(self.camera_yaw) * math.cos(self.camera_pitch)
        forward = np.array([fx, fy, fz], dtype=np.float32)
        forward /= np.linalg.norm(forward)
        
        # Calculate right and up vectors for movement (camera-relative, fully free)
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, world_up)
        right_len = np.linalg.norm(right)
        if right_len > 0.0001:
            right /= right_len
        else:
            # If forward is parallel to world_up, use default right
            right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        
        # Calculate camera-relative up vector (perpendicular to forward and right)
        up = np.cross(right, forward)
        up /= np.linalg.norm(up)
        
        # Fully free camera movement (relative to camera orientation)
        move = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        if keys[K_w]:
            move += forward  # Move forward in camera direction
        if keys[K_s]:
            move -= forward  # Move backward
        if keys[K_d]:
            move += right  # Strafe right
        if keys[K_a]:
            move -= right  # Strafe left
        if keys[K_e]:
            move += up  # Move up (camera-relative up)
        if keys[K_q]:
            move -= up  # Move down (camera-relative down)
        
        move_len = float(np.linalg.norm(move))
        if move_len > 1e-6:
            move /= move_len
            self.camera_pos += move * (speed * dt)
        
        # Reset
        if keys[K_r]:
            self.frequency = DEFAULT_FREQUENCY
            self.amplitude = DEFAULT_AMPLITUDE
            self.phase_speed = DEFAULT_PHASE_SPEED
            self.movement_speed = 0.03
            self.morph_speed = 0.03
            self.threshold = self.base_threshold
            self.grid_size = self.base_grid_size
            self.voxel_scale = 1.0
            self.threshold = self.base_threshold
            self.voxel_grid = np.zeros((self.grid_size, self.grid_size, self.grid_size), dtype=bool)
            self.previous_grid = np.zeros((self.grid_size, self.grid_size, self.grid_size), dtype=bool)
            self.grid_size_changed = False
            self.time = 0.0
        
        # Pause
        if keys[K_SPACE]:
            self.paused = not self.paused
        
        # Handle key press events (for toggles)
        for event in pygame.event.get():
            if event.type == QUIT:
                self.running = False
            elif event.type == KEYDOWN:
                # Phase 1: Profiling toggle (P key)
                if event.key == K_p:
                    self.profiling_enabled = not self.profiling_enabled
                    if self.profiling_enabled:
                        # Clear any old stats when starting fresh
                        self.frame_times = []
                        self.debug_message = "PROFILING: ON (Collecting...) Press P again to print stats"
                        print("\n" + "="*60)
                        print(">>> PROFILING: ENABLED <<<")
                        print("NOW COLLECTING frame time statistics silently...")
                        print("Stats will be printed to CONSOLE when you press P again")
                        print(f"Collecting for up to {self.max_frame_time_history} frames (~2 seconds at 60 FPS)")
                        print("CHECK THE CONSOLE WINDOW for stats when you disable profiling")
                        print("="*60 + "\n")
                        self.debug_message_timer = 10.0  # Show message for 10 seconds to give time to read
                    else:
                        self.print_profiling_stats()
                        self.debug_message = "PROFILING: OFF - Check CONSOLE for stats!"
                        print("\n" + "="*60)
                        print(">>> PROFILING: DISABLED <<<")
                        print("STATS PRINTED TO CONSOLE - Check your terminal/console window!")
                        print("Look for 'PHASE 1 PROFILING STATISTICS' section above")
                        print("="*60 + "\n")
                        self.debug_message_timer = self.debug_message_duration
                
                # Debug: Face culling toggle (C key) - to verify face culling is working
                elif event.key == K_c:
                    self.face_culling_enabled = not self.face_culling_enabled
                    # Apply immediately (will be re-applied in render() every frame)
                    if self.face_culling_enabled:
                        self.ctx.enable(moderngl.CULL_FACE)
                        self.ctx.enable(moderngl.DEPTH_TEST)  # Re-enable depth test
                        self.debug_message = "FACE CULLING: ON"
                        print("\n" + "="*60)
                        print(">>> FACE CULLING: ENABLED <<<")
                        print(f"State: face_culling_enabled = {self.face_culling_enabled}")
                        print("Back faces HIDDEN by GPU face culling")
                        print("Depth test ENABLED (hides faces behind others)")
                        print("Rotate camera to see faces update dynamically")
                        print("="*60 + "\n")
                    else:
                        self.ctx.disable(moderngl.CULL_FACE)
                        self.ctx.disable(moderngl.DEPTH_TEST)  # Disable depth test to see back faces
                        self.debug_message = "FACE CULLING: OFF + NO DEPTH"
                        print("\n" + "="*60)
                        print(">>> FACE CULLING: DISABLED <<<")
                        print(f"State: face_culling_enabled = {self.face_culling_enabled}")
                        print("ALL 6 FACES visible + DEPTH TEST DISABLED")
                        print("You should see:")
                        print("  - ALL faces of each voxel (front + back)")
                        print("  - Overlapping faces visible")
                        print("  - Can see through mesh (back faces show)")
                        print("Note: Depth test also disabled to show back faces clearly")
                        print("Press C again to re-enable face culling and depth test")
                        print("="*60 + "\n")
                    self.debug_message_timer = self.debug_message_duration
    
    def update(self, dt):
        """Update simulation state."""
        self._last_dt_seconds = float(dt) * (1.0 / 60.0)
        
        if not self.paused:
            self.time += dt * 0.01
            
            # Update target selection (based on position progress)
            self.update_target_selection()
            
            # Update position progress (movement speed) - separate from morphing
            self.position_progress += self.movement_speed * dt
            if self.position_progress > 1.0:
                self.position_progress = 1.0
            
            # Update morph progress (morph speed) - separate from movement
            self.morph_progress += self.morph_speed * dt
            if self.morph_progress > 1.0:
                self.morph_progress = 1.0
            
            # Interpolate active mesh position using POSITION progress (movement)
            current_pos = self.target_positions[self.current_target_index]
            next_pos = self.target_positions[self.next_target_index]
            self.active_mesh_position = np.array(
                interpolate_position(current_pos, next_pos, self.position_progress),
                dtype=np.float32
            )
            
            # Store previous grid
            self.previous_grid = self.voxel_grid.copy()
            
            # Get current and next target shapes
            current_target = self.target_shapes[self.current_target_index]
            next_target = self.target_shapes[self.next_target_index]
            
            # Generate shape fields centered at GRID CENTER (not world positions)
            # The grid size is dynamic based on quality - larger grid = more resolution
            # Then we render the mesh offset by active_mesh_position
            grid_center_x = float(self.grid_size // 2)
            grid_center_y = float(self.grid_size // 2)
            grid_center_z = float(self.grid_size // 2)
            grid_center = np.array([grid_center_x, grid_center_y, grid_center_z], dtype=np.float32)
            
            # Generate shape fields at grid center with target shape types
            # The mesh will morph from current shape to next shape as it moves
            current_shape_field = self.generate_shape_field(
                current_target['shape_type'],
                grid_center  # Always generate at grid center
            )
            next_shape_field = self.generate_shape_field(
                next_target['shape_type'],
                grid_center  # Always generate at grid center
            )
            
            # Blend shape fields based on MORPH progress with organic easing
            # As morph_progress increases, mesh morphs from current shape to next shape
            blended_shape_field = blend_shape_fields(
                current_shape_field,
                next_shape_field,
                self.morph_progress
            )
            
            # Dynamically adjust shape influence based on proximity to target positions
            # When close to target (position_progress near 0 or 1), use higher influence for sharper shapes
            # During transition (middle), use lower influence for more organic flow
            distance_from_target = min(self.position_progress, 1.0 - self.position_progress)
            # When at target: distance_from_target = 0, influence = max
            # When mid-transition: distance_from_target = 0.5, influence = min
            # Use smooth curve to transition
            transition_factor = distance_from_target * 2.0  # 0 to 1 as we go from target to middle
            organic_influence = self.max_shape_influence * (1.0 - transition_factor) + self.transition_shape_influence * transition_factor
            
            # Also consider morph progress - when fully morphed to a shape, increase influence
            morph_sharpness = 1.0 - abs(self.morph_progress - 0.5) * 2.0  # 1.0 at 0 or 1, 0.0 at 0.5
            # When morph_progress is 0 (pure current shape) or 1 (pure next shape), we're at a target
            # Blend between transition and target influence based on morph position
            self.shape_influence = (organic_influence * (1.0 - morph_sharpness * 0.3) + 
                                   self.max_shape_influence * morph_sharpness * 0.3)
            
            # Phase 1: Calculate base wave field with compute shader or CPU fallback
            wave_field = None
            if self.compute_generator and self.compute_generator.use_compute_shaders:
                # Try compute shader generation
                wave_field = self.compute_generator.generate_wave_field(
                    self.time, self.frequency, self.amplitude, self.phase_speed
                )
            
            # Fallback to CPU generation if compute shader failed or not available
            if wave_field is None:
                wave_field = calculate_wave_field_3d(
                    self.grid_size,  # Pass current grid size
                    self.time,
                    self.frequency,
                    self.amplitude,
                    self.phase_speed
                )
            
            # Apply shape field to wave field with dynamic influence
            # Higher influence when at targets = sharper shapes
            # Lower influence during transitions = more organic flow
            combined_field = apply_shape_field_to_wave(
                wave_field,
                blended_shape_field,
                self.shape_influence
            )
            
            # Update voxel grid
            self.voxel_grid = update_voxel_grid(
                combined_field,
                self.previous_grid,
                self.threshold,
                self.feedback_strength
            )
        
        # Update debug message timer
        if self.debug_message_timer > 0:
            self.debug_message_timer -= dt * (1.0 / 60.0)  # Decrease timer (assuming 60 FPS base)
            if self.debug_message_timer <= 0:
                self.debug_message = ""
        
        # Phase 1: Performance profiling (measure frame time)
        if self.profiling_enabled:
            import time
            if not hasattr(self, '_frame_start_time'):
                self._frame_start_time = time.perf_counter()
            else:
                frame_time_ms = (time.perf_counter() - self._frame_start_time) * 1000.0
                self.frame_times.append(frame_time_ms)
                if len(self.frame_times) > self.max_frame_time_history:
                    self.frame_times.pop(0)
                self._frame_start_time = time.perf_counter()
        
        # Update FPS
        self.frame_count += 1
        current_time = pygame.time.get_ticks()
        if current_time - self.last_fps_update > 1000:
            self.fps = self.frame_count
            self.frame_count = 0
            self.last_fps_update = current_time
    
    def render(self):
        """Render everything to screen."""
        # Apply face culling and depth test state every frame (CRITICAL: ensure state persists)
        # Must be set at the start of each render call
        if self.face_culling_enabled:
            self.ctx.enable(moderngl.CULL_FACE)
            self.ctx.enable(moderngl.DEPTH_TEST)  # Re-enable depth test
        else:
            self.ctx.disable(moderngl.CULL_FACE)
            self.ctx.disable(moderngl.DEPTH_TEST)  # Disable depth test to see back faces
        
        # Camera setup - fully free 3D camera with no constraints
        # Calculate forward direction from yaw and pitch (spherical coordinates)
        fx = math.cos(self.camera_yaw) * math.cos(self.camera_pitch)
        fy = math.sin(self.camera_pitch)
        fz = math.sin(self.camera_yaw) * math.cos(self.camera_pitch)
        forward = np.array([fx, fy, fz], dtype=np.float32)
        forward /= np.linalg.norm(forward)
        
        # Generate instances for active mesh
        # Scale spacing by same factor as cube size to keep proportions
        # When cubes are smaller, spacing is smaller (same scale)
        # Use adjacency-based culling only (faces not occluded by neighbors)
        # GPU back-face culling (enabled) automatically handles camera-facing faces
        # GPU back-face culling updates automatically with view matrix every frame
        # This is the correct approach for instanced rendering - GPU handles per-face visibility
        instances = generate_instances(
            self.voxel_grid, 
            self.active_mesh_position, 
            spacing_scale=self.voxel_scale,
            camera_pos=None  # Disable CPU camera-facing check - GPU back-face culling handles it automatically
        )
        # Voxel count: Number of SURFACE voxels rendered (not all voxels in grid)
        # Adjacency-based culling filters to only surface voxels (those with at least one visible face)
        # This is why the mesh appears "hollow" - only outer surface voxels are included
        self.voxel_count = len(instances)
        
        # Always render wireframes even if no voxels
        # Don't return early - we still want to see wireframes
        
        camera_pos = self.camera_pos
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        
        # Create view matrix for fully free camera
        view = create_view_matrix_free_look(camera_pos, forward, world_up)
        projection = create_projection_matrix(
            math.radians(60.0),
            WINDOW_WIDTH / WINDOW_HEIGHT,
            0.1,
            1000.0
        )
        model = np.eye(4, dtype=np.float32)
        
        # Set uniforms
        self.program['projection'].write(projection.tobytes())
        self.program['view'].write(view.tobytes())
        self.program['model'].write(model.tobytes())
        self.program['voxelScale'] = self.voxel_scale  # Quality control: voxel size
        self.program['lightDir'].write(self.light_dir.tobytes())
        self.program['lightColor'] = (1.0, 1.0, 1.0)
        self.program['ambientStrength'] = 0.3
        self.program['diffuseStrength'] = 0.7
        
        # Clear screen
        self.ctx.clear()
        
        # Update instance buffer
        self.instance_buffer.write(instances.tobytes())
        
        # Render active mesh (only if there are voxels)
        if len(instances) > 0:
            # Ensure face culling state is applied before rendering
            # (State is set at top of render(), but double-check here)
            if self.face_culling_enabled:
                self.ctx.enable(moderngl.CULL_FACE)
            else:
                self.ctx.disable(moderngl.CULL_FACE)
            
            self.vao.render(mode=moderngl.TRIANGLES, instances=len(instances))
        
        # Render wireframe targets
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.wireframe_program['projection'].write(projection.tobytes())
        self.wireframe_program['view'].write(view.tobytes())
        self.wireframe_program['model'].write(model.tobytes())
        
        for i, wireframe_data in enumerate(self.wireframe_vaos):
            if wireframe_data is not None:
                wireframe_vao, vertex_count = wireframe_data
                # Highlight current/next targets with different colors
                if i == self.current_target_index:
                    self.wireframe_program['wireframeColor'] = (1.0, 0.5, 0.0)  # Orange for current
                elif i == self.next_target_index:
                    self.wireframe_program['wireframeColor'] = (0.0, 1.0, 0.5)  # Green for next
                else:
                    self.wireframe_program['wireframeColor'] = (0.5, 0.8, 1.0)  # Cyan for others
                
                wireframe_vao.render(mode=moderngl.LINES)
        
        self.ctx.disable(moderngl.BLEND)
        
        # Draw debug message on screen (if active)
        if self.debug_message and self.debug_message_timer > 0:
            self.draw_debug_message()
        
        # Draw UI
        self.draw_ui()
        
        pygame.display.flip()
    
    def draw_debug_message(self):
        """Debug message shown in title bar - OpenGL mode doesn't support easy text rendering."""
        # Message is shown in title bar via draw_ui() - no action needed here
        pass
    
    def draw_ui(self):
        """Update window title with current stats (shortened to fit window)."""
        current_shape = self.target_shapes[self.current_target_index]['name']
        next_shape = self.target_shapes[self.next_target_index]['name']
        cull_status = "ON" if self.face_culling_enabled else "OFF"
        prof_status = "ON" if self.profiling_enabled else "OFF"
        
        # Add debug message to title if active
        debug_indicator = ""
        if self.debug_message and self.debug_message_timer > 0:
            # Short version for title bar
            if "FACE CULLING" in self.debug_message:
                debug_indicator = " [CULL:" + cull_status + "]"
            elif "PROFILING" in self.debug_message:
                debug_indicator = " [PROF:" + prof_status + "]"
        
        # Voxel count shows SURFACE voxels only (adjacency culling filters to outer faces)
        # Grid size shows total possible voxels (grid_size³), but only surface voxels are rendered
        # Shortened title to fit in window title bar
        title = (
            f"Surface:{self.voxel_count}/{self.grid_size**3} FPS:{self.fps} | "
            f"{current_shape}→{next_shape} | "
            f"Move:{self.position_progress:.1f}(T/G) Morph:{self.morph_progress:.1f}(Y/H) | "
            f"Grid:{self.grid_size}³ Scale:{self.voxel_scale:.2f}(N/M) | "
            f"Cull:{cull_status}(C) Prof:{prof_status}(P){debug_indicator}"
        )
        pygame.display.set_caption(title)
    
    def run(self):
        """Main game loop."""
        print("Starting Morphogenic Constellation System...")
        print("Controls:")
        print("  WASD: Move camera")
        print("  Q/E: Move up/down")
        print("  Arrow Keys: Look around")
        print("  Shift: Move faster")
        print("  Ctrl: Move slower")
        print("  I/K: Adjust frequency")
        print("  J/L: Adjust amplitude")
        print("  U/O: Adjust phase speed")
        print("  T/G: Increase/Decrease movement speed")
        print("  Y/H: Increase/Decrease morphing speed")
        print("  N/M: Quality control - N=higher quality (bigger grid, more cubes, smaller cubes) / M=lower quality (smaller grid, fewer cubes, larger cubes)")
        print("  Space: Pause/Resume")
        print("  R: Reset parameters")
        print("  P: Toggle profiling (collects frame time stats)")
        print("  C: Toggle face culling (verify face culling works - when OFF: see through mesh, when ON: normal)")
        print("  P: Toggle profiling (collects frame time stats silently - stats print to CONSOLE when disabled)")
        
        clock = pygame.time.Clock()
        
        while self.running:
            dt = clock.tick(60) / 16.67
            
            self.handle_input()
            self.update(dt)
            self.render()
        
        # Phase 1: Cleanup compute generator
        if self.compute_generator:
            self.compute_generator.release()
        
        pygame.quit()
    
    def print_profiling_stats(self):
        """Phase 1: Print profiling statistics."""
        if not self.frame_times:
            print("No profiling data collected yet. Enable profiling with P key and run for a few seconds.")
            return
        
        frame_times = np.array(self.frame_times)
        avg_frame_time = np.mean(frame_times)
        min_frame_time = np.min(frame_times)
        max_frame_time = np.max(frame_times)
        p95_frame_time = np.percentile(frame_times, 95)
        p99_frame_time = np.percentile(frame_times, 99)
        
        avg_fps = 1000.0 / avg_frame_time if avg_frame_time > 0 else 0
        min_fps = 1000.0 / max_frame_time if max_frame_time > 0 else 0
        max_fps = 1000.0 / min_frame_time if min_frame_time > 0 else 0
        
        stutter_count = np.sum(frame_times > 16.67)  # Frames > 16.67ms = below 60 FPS
        stutter_percent = (stutter_count / len(frame_times)) * 100.0
        
        print("\n" + "="*60)
        print("PHASE 1 PROFILING STATISTICS")
        print("="*60)
        print(f"Frames sampled: {len(frame_times)}")
        print(f"\nFrame Time (ms):")
        print(f"  Average: {avg_frame_time:.2f}ms")
        print(f"  Min:     {min_frame_time:.2f}ms")
        print(f"  Max:     {max_frame_time:.2f}ms")
        print(f"  P95:     {p95_frame_time:.2f}ms")
        print(f"  P99:     {p99_frame_time:.2f}ms")
        print(f"\nFrame Rate (FPS):")
        print(f"  Average: {avg_fps:.1f} FPS")
        print(f"  Min:     {min_fps:.1f} FPS")
        print(f"  Max:     {max_fps:.1f} FPS")
        print(f"\nPerformance:")
        print(f"  Stutter frames (>16.67ms): {stutter_count} ({stutter_percent:.1f}%)")
        print(f"  Grid size: {self.grid_size}³")
        print(f"  Visible voxels: {self.voxel_count}")
        print(f"  Compute shaders: {'[OK] Enabled' if (self.compute_generator and self.compute_generator.use_compute_shaders) else '[CPU] Fallback'}")
        print("="*60 + "\n")


if __name__ == "__main__":
    system = ConstellationSystem()
    system.run()
