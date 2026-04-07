"""
Wireframe Generator for Target Shapes
Generates wireframe line segments for visualizing target shapes.
"""

import numpy as np
import math


def generate_sphere_wireframe(center, radius=8.0, segments=16):
    """Generate wireframe vertices for a sphere as line segments."""
    vertices = []
    # Generate circles in XY, XZ, and YZ planes as line segments
    prev_xy = None
    prev_xz = None
    prev_yz = None
    first_xy = None
    first_xz = None
    first_yz = None
    
    for i in range(segments + 1):
        angle = (i / segments) * 2.0 * math.pi
        # XY plane circle
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        z = center[2]
        if prev_xy is not None:
            vertices.extend([prev_xy, [x, y, z]])
        else:
            first_xy = [x, y, z]
        prev_xy = [x, y, z]
        
        # XZ plane circle
        x = center[0] + radius * math.cos(angle)
        y = center[1]
        z = center[2] + radius * math.sin(angle)
        if prev_xz is not None:
            vertices.extend([prev_xz, [x, y, z]])
        else:
            first_xz = [x, y, z]
        prev_xz = [x, y, z]
        
        # YZ plane circle
        x = center[0]
        y = center[1] + radius * math.cos(angle)
        z = center[2] + radius * math.sin(angle)
        if prev_yz is not None:
            vertices.extend([prev_yz, [x, y, z]])
        else:
            first_yz = [x, y, z]
        prev_yz = [x, y, z]
    
    # Close loops
    if first_xy and prev_xy:
        vertices.extend([prev_xy, first_xy])
    if first_xz and prev_xz:
        vertices.extend([prev_xz, first_xz])
    if first_yz and prev_yz:
        vertices.extend([prev_yz, first_yz])
    
    return np.array(vertices, dtype=np.float32)


def generate_cube_wireframe(center, size=10.0):
    """Generate wireframe vertices for a cube as line segments."""
    s = size / 2.0
    # Generate as line segments (pairs of vertices)
    vertices = []
    # Bottom face (4 edges)
    vertices.extend([[center[0] - s, center[1] - s, center[2] - s], [center[0] + s, center[1] - s, center[2] - s]])
    vertices.extend([[center[0] + s, center[1] - s, center[2] - s], [center[0] + s, center[1] - s, center[2] + s]])
    vertices.extend([[center[0] + s, center[1] - s, center[2] + s], [center[0] - s, center[1] - s, center[2] + s]])
    vertices.extend([[center[0] - s, center[1] - s, center[2] + s], [center[0] - s, center[1] - s, center[2] - s]])
    # Top face (4 edges)
    vertices.extend([[center[0] - s, center[1] + s, center[2] - s], [center[0] + s, center[1] + s, center[2] - s]])
    vertices.extend([[center[0] + s, center[1] + s, center[2] - s], [center[0] + s, center[1] + s, center[2] + s]])
    vertices.extend([[center[0] + s, center[1] + s, center[2] + s], [center[0] - s, center[1] + s, center[2] + s]])
    vertices.extend([[center[0] - s, center[1] + s, center[2] + s], [center[0] - s, center[1] + s, center[2] - s]])
    # Vertical edges (4 edges)
    vertices.extend([[center[0] - s, center[1] - s, center[2] - s], [center[0] - s, center[1] + s, center[2] - s]])
    vertices.extend([[center[0] + s, center[1] - s, center[2] - s], [center[0] + s, center[1] + s, center[2] - s]])
    vertices.extend([[center[0] + s, center[1] - s, center[2] + s], [center[0] + s, center[1] + s, center[2] + s]])
    vertices.extend([[center[0] - s, center[1] - s, center[2] + s], [center[0] - s, center[1] + s, center[2] + s]])
    return np.array(vertices, dtype=np.float32)


def generate_pyramid_wireframe(center, base_size=10.0, height=12.0):
    """Generate wireframe vertices for a pyramid as line segments."""
    s = base_size / 2.0
    top = [center[0], center[1] + height, center[2]]
    vertices = []
    # Base square (4 edges)
    vertices.extend([[center[0] - s, center[1], center[2] - s], [center[0] + s, center[1], center[2] - s]])
    vertices.extend([[center[0] + s, center[1], center[2] - s], [center[0] + s, center[1], center[2] + s]])
    vertices.extend([[center[0] + s, center[1], center[2] + s], [center[0] - s, center[1], center[2] + s]])
    vertices.extend([[center[0] - s, center[1], center[2] + s], [center[0] - s, center[1], center[2] - s]])
    # Edges to top (4 edges)
    vertices.extend([[center[0] - s, center[1], center[2] - s], top])
    vertices.extend([[center[0] + s, center[1], center[2] - s], top])
    vertices.extend([[center[0] + s, center[1], center[2] + s], top])
    vertices.extend([[center[0] - s, center[1], center[2] + s], top])
    return np.array(vertices, dtype=np.float32)


def generate_torus_wireframe(center, major_radius=8.0, minor_radius=4.0, segments=16):
    """Generate wireframe vertices for a torus."""
    vertices = []
    # Generate major circles
    for i in range(segments):
        angle = (i / segments) * 2.0 * math.pi
        # Major circle in XZ plane
        x = center[0] + major_radius * math.cos(angle)
        y = center[1]
        z = center[2] + major_radius * math.sin(angle)
        vertices.append([x, y, z])
        # Minor circles around the major circle
        for j in range(4):  # 4 minor circles
            minor_angle = (j / 4.0) * 2.0 * math.pi
            x = center[0] + (major_radius + minor_radius * math.cos(minor_angle)) * math.cos(angle)
            y = center[1] + minor_radius * math.sin(minor_angle)
            z = center[2] + (major_radius + minor_radius * math.cos(minor_angle)) * math.sin(angle)
            vertices.append([x, y, z])
    return np.array(vertices, dtype=np.float32)


def generate_cylinder_wireframe(center, radius=8.0, height=12.0, segments=16):
    """Generate wireframe vertices for a cylinder."""
    vertices = []
    # Top and bottom circles
    for i in range(segments + 1):
        angle = (i / segments) * 2.0 * math.pi
        x = center[0] + radius * math.cos(angle)
        z = center[2] + radius * math.sin(angle)
        # Bottom circle
        vertices.append([x, center[1] - height/2, z])
        # Top circle
        vertices.append([x, center[1] + height/2, z])
        # Vertical lines
        if i < segments:
            vertices.append([x, center[1] - height/2, z])
            vertices.append([x, center[1] + height/2, z])
    return np.array(vertices, dtype=np.float32)


def generate_octahedron_wireframe(center, size=10.0):
    """Generate wireframe vertices for an octahedron."""
    s = size
    vertices = [
        # Top point
        [center[0], center[1] + s, center[2]],
        # Top square
        [center[0] - s, center[1], center[2]],
        [center[0] + s, center[1], center[2]],
        [center[0], center[1], center[2] - s],
        [center[0], center[1], center[2] + s],
        [center[0] - s, center[1], center[2]],
        # Bottom point
        [center[0], center[1] - s, center[2]],
        # Edges
        [center[0] - s, center[1], center[2]],
        [center[0], center[1] + s, center[2]],
        [center[0] + s, center[1], center[2]],
        [center[0], center[1] + s, center[2]],
        [center[0], center[1], center[2] - s],
        [center[0], center[1] + s, center[2]],
        [center[0], center[1], center[2] + s],
        [center[0], center[1] + s, center[2]],
        [center[0] - s, center[1], center[2]],
        [center[0], center[1] - s, center[2]],
        [center[0] + s, center[1], center[2]],
        [center[0], center[1] - s, center[2]],
        [center[0], center[1], center[2] - s],
        [center[0], center[1] - s, center[2]],
        [center[0], center[1], center[2] + s],
        [center[0], center[1] - s, center[2]],
    ]
    return np.array(vertices, dtype=np.float32)


def get_wireframe_for_shape(shape_type, position, scale=1.0):
    """
    Get wireframe vertices for a shape type.
    shape_type: 0=sphere, 1=cube, 2=pyramid, 3=torus, 4=cylinder, 5=octahedron
    """
    size = 8.0 * scale
    if shape_type == 0:  # sphere
        return generate_sphere_wireframe(position, radius=size)
    elif shape_type == 1:  # cube
        return generate_cube_wireframe(position, size=size)
    elif shape_type == 2:  # pyramid
        return generate_pyramid_wireframe(position, base_size=size, height=size*1.2)
    elif shape_type == 3:  # torus
        return generate_torus_wireframe(position, major_radius=size*0.6, minor_radius=size*0.4)
    elif shape_type == 4:  # cylinder
        return generate_cylinder_wireframe(position, radius=size*0.6, height=size*1.2)
    else:  # octahedron (5)
        return generate_octahedron_wireframe(position, size=size)
