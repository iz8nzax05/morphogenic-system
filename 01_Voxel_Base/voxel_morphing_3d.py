"""
32x32x32 Morphogenic Voxel System - 3D Solid Rendering
Upgraded version with ModernGL for proper 3D cube rendering with face-based lighting.
"""

import numpy as np
import numba
import pygame
import math
from pygame.locals import *
import moderngl
from pathlib import Path

# Constants
GRID_SIZE = 32
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
CENTER_X = GRID_SIZE // 2
CENTER_Y = GRID_SIZE // 2
CENTER_Z = GRID_SIZE // 2

# Default wave parameters
DEFAULT_FREQUENCY = 0.15
DEFAULT_AMPLITUDE = 0.5
DEFAULT_PHASE_SPEED = 0.02


@numba.jit(nopython=True)
def calculate_wave_field_3d(time, frequency, amplitude, phase_speed):
    """
    Calculate 3D sine wave field that controls voxel spawn probability.
    Creates interference patterns from multiple wave directions.
    """
    field = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    phase = time * phase_speed
    
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            for z in range(GRID_SIZE):
                # Normalize coordinates to -1 to 1 range
                nx = (x - CENTER_X) / (GRID_SIZE / 2.0)
                ny = (y - CENTER_Y) / (GRID_SIZE / 2.0)
                nz = (z - CENTER_Z) / (GRID_SIZE / 2.0)
                
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
def update_voxel_grid(wave_field, previous_grid, threshold, feedback_strength):
    """
    Update voxel grid based on wave field and feedback from previous state.
    Spawn/despawn voxels based on field values with mathematical feedback.
    """
    new_grid = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=np.bool_)
    
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            for z in range(GRID_SIZE):
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
    
    # Top face (y+1)
    if y + 1 >= GRID_SIZE or not voxel_grid[x, y + 1, z]:
        faces[0] = True
    
    # Bottom face (y-1)
    if y - 1 < 0 or not voxel_grid[x, y - 1, z]:
        faces[1] = True
    
    # Front face (z+1)
    if z + 1 >= GRID_SIZE or not voxel_grid[x, y, z + 1]:
        faces[2] = True
    
    # Back face (z-1)
    if z - 1 < 0 or not voxel_grid[x, y, z - 1]:
        faces[3] = True
    
    # Right face (x+1)
    if x + 1 >= GRID_SIZE or not voxel_grid[x + 1, y, z]:
        faces[4] = True
    
    # Left face (x-1)
    if x - 1 < 0 or not voxel_grid[x - 1, y, z]:
        faces[5] = True
    
    return faces


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
    # All faces wound counter-clockwise when viewed from outside
    indices = np.array([
        # Top face (y = +0.5, looking down)
        0, 1, 2,  0, 2, 3,
        # Bottom face (y = -0.5, looking up) - FIXED winding
        4, 5, 6,  4, 6, 7,
        # Front face (z = +0.5, looking from -Z)
        8, 9, 10,  8, 10, 11,
        # Back face (z = -0.5, looking from +Z) - FIXED winding
        12, 13, 14,  12, 14, 15,
        # Right face (x = +0.5, looking from -X)
        16, 17, 18,  16, 18, 19,
        # Left face (x = -0.5, looking from +X) - FIXED winding
        20, 21, 22,  20, 22, 23,
    ], dtype=np.uint32)
    
    return vertices, indices


def generate_instances(voxel_grid):
    """
    Generate instance positions for visible voxels with face culling.
    Returns array of (x, y, z) positions for visible voxels.
    """
    instances = []
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            for z in range(GRID_SIZE):
                if voxel_grid[x, y, z]:
                    # Check if any face is visible
                    faces = get_visible_faces(voxel_grid, x, y, z)
                    if np.any(faces):
                        instances.append([float(x), float(y), float(z)])
    
    if len(instances) == 0:
        return np.array([], dtype=np.float32).reshape(0, 3)
    
    return np.array(instances, dtype=np.float32)


def create_view_matrix(camera_pos, target, up):
    """Create view matrix (camera transformation) - fixed according to research."""
    forward = target - camera_pos
    forward_len = np.linalg.norm(forward)
    if forward_len < 0.0001:  # Avoid division by zero
        forward = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    else:
        forward = forward / forward_len
    
    right = np.cross(forward, up)
    right_len = np.linalg.norm(right)
    if right_len < 0.0001:  # Avoid division by zero
        right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        right = right / right_len
    
    new_up = np.cross(right, forward)
    new_up = new_up / np.linalg.norm(new_up)
    
    # Create view matrix (look-at matrix)
    # Column-major order for OpenGL
    view = np.eye(4, dtype=np.float32)
    view[0, :3] = right
    view[1, :3] = new_up
    view[2, :3] = -forward
    
    # Translate by negative camera position
    translation = -camera_pos
    view[3, 0] = np.dot(right, translation)
    view[3, 1] = np.dot(new_up, translation)
    view[3, 2] = np.dot(-forward, translation)
    
    return view


def create_projection_matrix(fov, aspect, near, far):
    """Create perspective projection matrix (fixed according to research)."""
    f = 1.0 / math.tan(fov / 2.0)
    proj = np.zeros((4, 4), dtype=np.float32)
    
    # Fix #6: Correct perspective projection matrix
    proj[0, 0] = f / aspect
    proj[1, 1] = f
    proj[2, 2] = (far + near) / (near - far)
    proj[2, 3] = -1.0
    proj[3, 2] = (2.0 * far * near) / (near - far)
    # proj[3, 3] = 0.0 (default, but explicit for clarity)
    
    return proj


class VoxelMorphing3D:
    def __init__(self):
        pygame.init()
        pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF)
        pygame.display.set_caption("32x32x32 Morphogenic Voxel System - 3D Solid Rendering")
        
        # Create ModernGL context
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST)
        # Disable face culling so all 6 faces of cubes are visible
        # self.ctx.enable(moderngl.CULL_FACE)  # Disabled - user wants to see all faces
        
        # Fix #7: Set clear color once (best practice from research)
        self.ctx.clear_color = (0.05, 0.05, 0.1, 1.0)  # Dark blue background
        
        # Load shaders
        shader_dir = Path(__file__).parent
        try:
            with open(shader_dir / "voxel_shaders.vert") as f:
                vertex_shader = f.read()
            with open(shader_dir / "voxel_shaders.frag") as f:
                fragment_shader = f.read()
        except FileNotFoundError:
            # Fallback: inline shaders if files not found
            vertex_shader = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in float aFaceID;
layout(location = 3) in vec3 aInstancePos;
uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;
out vec3 FragPos;
out vec3 FragNormal;
out float FaceID;
void main() {
    vec3 worldPos = aPos + aInstancePos;  // Changed from * 0.5 to full size (1.0 unit cubes, no gaps)
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
        
        # Create cube geometry
        vertices, indices = create_cube_geometry()
        
        # Create vertex buffer
        self.vbo = self.ctx.buffer(vertices.tobytes())
        
        # Create index buffer
        self.ibo = self.ctx.buffer(indices.tobytes())
        
        # Create instance buffer (will be updated each frame)
        # Allocate buffer with maximum possible size (all voxels)
        max_instances = GRID_SIZE**3
        max_buffer_size = max_instances * 3 * 4  # 3 floats per position * 4 bytes per float
        self.instance_buffer = self.ctx.buffer(reserve=max_buffer_size)
        
        # Initialize with dummy data for VAO setup
        dummy_instances = np.zeros((1, 3), dtype=np.float32)
        self.instance_buffer.write(dummy_instances.tobytes())
        
        # Create vertex array with instance buffer included
        # The '/i' syntax marks it as instanced (one value per instance)
        self.vao = self.ctx.vertex_array(
            self.program,
            [
                (self.vbo, '3f 3f 1f', 'aPos', 'aNormal', 'aFaceID'),
                (self.instance_buffer, '3f /i', 'aInstancePos'),  # /i = instanced
            ],
            self.ibo,
        )
        
        # Voxel grid
        self.voxel_grid = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=bool)
        self.previous_grid = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=bool)
        
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
        
        # Camera controls (free-fly)
        # More intuitive movement: WASD = strafe/forward/back, Q/E = down/up, arrows = look
        self.camera_pos = np.array([CENTER_X, CENTER_Y + 25.0, CENTER_Z + 90.0], dtype=np.float32)
        self.camera_yaw = math.radians(-90.0)   # facing toward -Z-ish
        self.camera_pitch = math.radians(-10.0)
        self.camera_move_speed = 30.0          # units/sec
        self.camera_look_speed = 1.5           # radians/sec
        
        # Light direction
        self.light_dir = np.array([0.5, 0.8, -0.5], dtype=np.float32)
        self.light_dir = self.light_dir / np.linalg.norm(self.light_dir)
        
        # Font for UI
        pygame.font.init()
        self.font = pygame.font.Font(None, 24)
        
        # Performance tracking
        self.frame_count = 0
        self.fps = 0
        self.last_fps_update = pygame.time.get_ticks()
        self.voxel_count = 0
        
    def handle_input(self):
        """Process keyboard and mouse input."""
        keys = pygame.key.get_pressed()
        
        # --- Wave parameter adjustments (moved off WASD/QE to free up movement) ---
        # Frequency: I/K
        if keys[K_i]:
            self.frequency += 0.001
        if keys[K_k]:
            self.frequency -= 0.001
        # Amplitude: J/L
        if keys[K_j]:
            self.amplitude = max(0.0, self.amplitude - 0.01)
        if keys[K_l]:
            self.amplitude = min(1.0, self.amplitude + 0.01)
        # Phase speed: U/O
        if keys[K_u]:
            self.phase_speed -= 0.001
        if keys[K_o]:
            self.phase_speed += 0.001

        # --- Camera look (arrow keys) ---
        dt = getattr(self, "_last_dt_seconds", 1.0 / 60.0)
        if keys[K_LEFT]:
            self.camera_yaw -= self.camera_look_speed * dt
        if keys[K_RIGHT]:
            self.camera_yaw += self.camera_look_speed * dt
        if keys[K_UP]:
            self.camera_pitch += self.camera_look_speed * dt
        if keys[K_DOWN]:
            self.camera_pitch -= self.camera_look_speed * dt

        # Clamp pitch to avoid flipping
        max_pitch = math.radians(89.0)
        if self.camera_pitch > max_pitch:
            self.camera_pitch = max_pitch
        if self.camera_pitch < -max_pitch:
            self.camera_pitch = -max_pitch

        # --- Camera movement (WASD + Q/E) ---
        # Speed modifiers
        speed = self.camera_move_speed
        if keys[K_LSHIFT] or keys[K_RSHIFT]:
            speed *= 2.5
        if keys[K_LCTRL] or keys[K_RCTRL]:
            speed *= 0.4

        # Forward vector from yaw/pitch
        fx = math.cos(self.camera_yaw) * math.cos(self.camera_pitch)
        fy = math.sin(self.camera_pitch)
        fz = math.sin(self.camera_yaw) * math.cos(self.camera_pitch)
        forward = np.array([fx, fy, fz], dtype=np.float32)
        forward /= np.linalg.norm(forward)
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)

        move = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        if keys[K_w]:
            move += forward
        if keys[K_s]:
            move -= forward
        if keys[K_d]:
            move += right
        if keys[K_a]:
            move -= right
        if keys[K_e]:
            move += world_up
        if keys[K_q]:
            move -= world_up

        move_len = float(np.linalg.norm(move))
        if move_len > 1e-6:
            move /= move_len
            self.camera_pos += move * (speed * dt)
        
        # Reset
        if keys[K_r]:
            self.frequency = DEFAULT_FREQUENCY
            self.amplitude = DEFAULT_AMPLITUDE
            self.phase_speed = DEFAULT_PHASE_SPEED
            self.time = 0.0
        
        # Pause
        if keys[K_SPACE]:
            self.paused = not self.paused
        
        # Presets
        if keys[K_1]:
            self.frequency = 0.1
            self.amplitude = 0.3
            self.threshold = 0.6
        if keys[K_2]:
            self.frequency = 0.2
            self.amplitude = 0.5
            self.threshold = 0.5
        if keys[K_3]:
            self.frequency = 0.25
            self.amplitude = 0.7
            self.threshold = 0.4
        
        for event in pygame.event.get():
            if event.type == QUIT:
                self.running = False
    
    def update(self, dt):
        """Update simulation state."""
        # Store dt in seconds for input smoothing
        # dt passed in is normalized-to-60fps, so convert back to seconds
        self._last_dt_seconds = float(dt) * (1.0 / 60.0)
        if not self.paused:
            self.time += dt * 0.01
            
            # Store previous grid for feedback
            self.previous_grid = self.voxel_grid.copy()
            
            # Calculate wave field
            wave_field = calculate_wave_field_3d(
                self.time, 
                self.frequency, 
                self.amplitude, 
                self.phase_speed
            )
            
            # Update voxel grid with feedback
            self.voxel_grid = update_voxel_grid(
                wave_field, 
                self.previous_grid, 
                self.threshold, 
                self.feedback_strength
            )
        
        # Update FPS
        self.frame_count += 1
        current_time = pygame.time.get_ticks()
        if current_time - self.last_fps_update > 1000:
            self.fps = self.frame_count
            self.frame_count = 0
            self.last_fps_update = current_time
    
    def render(self):
        """Render everything to screen."""
        # Generate instances for visible voxels FIRST
        instances = generate_instances(self.voxel_grid)
        self.voxel_count = len(instances)
        
        if len(instances) == 0:
            # Still clear screen even if no instances
            self.ctx.clear()
            pygame.display.flip()
            return
        
        # Free-fly camera: look direction from yaw/pitch
        fx = math.cos(self.camera_yaw) * math.cos(self.camera_pitch)
        fy = math.sin(self.camera_pitch)
        fz = math.sin(self.camera_yaw) * math.cos(self.camera_pitch)
        forward = np.array([fx, fy, fz], dtype=np.float32)
        forward /= np.linalg.norm(forward)
        camera_pos = self.camera_pos
        target = camera_pos + forward
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        
        # Create matrices BEFORE clearing (Fix #9: Set matrices before render)
        view = create_view_matrix(camera_pos, target, up)
        projection = create_projection_matrix(
            math.radians(60.0),  # FOV (Fix #3: Increased from 45 for better view)
            WINDOW_WIDTH / WINDOW_HEIGHT,  # Aspect ratio
            0.1,  # Near plane
            1000.0  # Far plane (Fix #3: Increased from 200 for large scene)
        )
        model = np.eye(4, dtype=np.float32)
        
        # Set uniforms BEFORE rendering
        self.program['projection'].write(projection.tobytes())
        self.program['view'].write(view.tobytes())
        self.program['model'].write(model.tobytes())
        self.program['lightDir'].write(self.light_dir.tobytes())
        self.program['lightColor'] = (1.0, 1.0, 1.0)
        self.program['ambientStrength'] = 0.3
        self.program['diffuseStrength'] = 0.7
        
        # Clear screen AFTER setting matrices (Fix #10: Clear just before render)
        self.ctx.clear()
        
        # Update instance buffer AFTER matrices are set
        self.instance_buffer.write(instances.tobytes())
        
        # Render with instancing
        # ModernGL render method: mode, first, count, instances
        if len(instances) > 0:
            self.vao.render(mode=moderngl.TRIANGLES, instances=len(instances))
        
        # Draw UI (using pygame on top)
        self.draw_ui()
        
        # Swap buffers
        pygame.display.flip()
    
    def draw_ui(self):
        """Update window title with current stats (simple UI alternative)."""
        # Fix #11: Can't use pygame text rendering over OpenGL easily.
        # Instead, update the window title with current stats.
        title = (
            f"Voxels: {self.voxel_count} | FPS: {self.fps} | "
            f"Freq: {self.frequency:.3f} Amp: {self.amplitude:.2f} Phase: {self.phase_speed:.3f} | "
            f"Move: WASD + QE | Look: Arrows | Boost: Shift | Slow: Ctrl | "
            f"Params: I/K freq, J/L amp, U/O phase | Space pause | R reset | 1-3 presets"
        )
        pygame.display.set_caption(title)
    
    def run(self):
        """Main game loop."""
        print("Starting Morphogenic Voxel System - 3D Solid Rendering...")
        print("Controls:")
        print("  WASD: Move (forward/back/strafe)")
        print("  Q/E: Move down/up")
        print("  Arrow Keys: Look around")
        print("  Shift: Move faster")
        print("  Ctrl: Move slower")
        print("  I/K: Adjust frequency")
        print("  J/L: Adjust amplitude")
        print("  U/O: Adjust phase speed")
        print("  Space: Pause/Resume")
        print("  R: Reset parameters")
        print("  1-3: Preset configurations")
        
        clock = pygame.time.Clock()
        
        while self.running:
            dt = clock.tick(60) / 16.67  # Normalize to ~60fps
            
            self.handle_input()
            self.update(dt)
            self.render()
            
            # Print stats every 60 frames
            if self.frame_count % 60 == 0:
                # Debug output (Fix #8: Add debugging from research)
                print(f"FPS: {self.fps}, Voxels: {self.voxel_count}/{GRID_SIZE**3}, Time: {self.time:.2f}")
                if self.voxel_count == 0:
                    print("  ⚠️  WARNING: No voxels to render! (Black screen expected)")
                else:
                    # Print camera info for debugging (free-fly)
                    cam = self.camera_pos
                    fx = math.cos(self.camera_yaw) * math.cos(self.camera_pitch)
                    fy = math.sin(self.camera_pitch)
                    fz = math.sin(self.camera_yaw) * math.cos(self.camera_pitch)
                    print(f"  Camera: ({cam[0]:.1f}, {cam[1]:.1f}, {cam[2]:.1f})")
                    print(f"  Look yaw/pitch (deg): ({math.degrees(self.camera_yaw):.1f}, {math.degrees(self.camera_pitch):.1f})")
                    print(f"  Forward: ({fx:.2f}, {fy:.2f}, {fz:.2f})")
                    print(f"  Rendering {self.voxel_count} voxel instances")
        
        pygame.quit()


if __name__ == "__main__":
    system = VoxelMorphing3D()
    system.run()

