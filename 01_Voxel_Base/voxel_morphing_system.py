"""
32x32x32 Morphogenic Voxel System
Mathematical feedback loops creating "unstable yet stable" forms.
Individual voxels spawn/despawn based on 3D sine wave resistance fields.
"""

import numpy as np
import numba
import pygame
import math
from pygame.locals import *

# Constants
GRID_SIZE = 32
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
CENTER_X = GRID_SIZE // 2
CENTER_Y = GRID_SIZE // 2
CENTER_Z = GRID_SIZE // 2

# Color definitions
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (100, 150, 255)
RED = (255, 100, 100)
GREEN = (100, 255, 150)
YELLOW = (255, 255, 100)
CYAN = (100, 255, 255)

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


def project_3d_to_2d(x, y, z, camera_angle_x, camera_angle_y, zoom=2.0):
    """
    Project 3D coordinates to 2D screen coordinates with rotation.
    """
    # Rotate around Y axis
    cos_y = math.cos(camera_angle_y)
    sin_y = math.sin(camera_angle_y)
    x_rot = x * cos_y - z * sin_y
    z_rot = x * sin_y + z * cos_y
    
    # Rotate around X axis
    cos_x = math.cos(camera_angle_x)
    sin_x = math.sin(camera_angle_x)
    y_rot = y * cos_x - z_rot * sin_x
    z_final = y * sin_x + z_rot * cos_x
    
    # Perspective projection
    if z_final < 0.1:
        z_final = 0.1
    
    scale = zoom * 400 / (GRID_SIZE + z_final * 10)
    screen_x = WINDOW_WIDTH // 2 + (x_rot - CENTER_X) * scale
    screen_y = WINDOW_HEIGHT // 2 + (y_rot - CENTER_Y) * scale
    
    return int(screen_x), int(screen_y), scale


class VoxelMorphingSystem:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("32x32x32 Morphogenic Voxel System")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        
        # Voxel grid
        self.voxel_grid = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=bool)
        self.previous_grid = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=bool)
        
        # Time and animation
        self.time = 0.0
        self.running = True
        self.paused = False
        
        # Wave parameters (adjustable with keyboard)
        self.frequency = DEFAULT_FREQUENCY
        self.amplitude = DEFAULT_AMPLITUDE
        self.phase_speed = DEFAULT_PHASE_SPEED
        self.threshold = 0.5
        self.feedback_strength = 0.5
        
        # Camera controls
        self.camera_angle_x = 0.5
        self.camera_angle_y = 0.5
        self.zoom = 2.0
        
        # Visualization mode
        self.view_mode = "wireframe"  # "wireframe" or "slice"
        self.slice_depth = GRID_SIZE // 2
        
        # Performance tracking
        self.frame_count = 0
        self.fps = 0
        self.last_fps_update = pygame.time.get_ticks()
        
    def handle_input(self):
        """Process keyboard and mouse input."""
        keys = pygame.key.get_pressed()
        
        # Parameter adjustments
        if keys[K_w]:
            self.frequency += 0.001
        if keys[K_s]:
            self.frequency -= 0.001
        if keys[K_a]:
            self.amplitude = max(0.0, self.amplitude - 0.01)
        if keys[K_d]:
            self.amplitude = min(1.0, self.amplitude + 0.01)
        if keys[K_q]:
            self.phase_speed -= 0.001
        if keys[K_e]:
            self.phase_speed += 0.001
        
        # Camera rotation
        if keys[K_LEFT]:
            self.camera_angle_y -= 0.02
        if keys[K_RIGHT]:
            self.camera_angle_y += 0.02
        if keys[K_UP]:
            self.camera_angle_x -= 0.02
        if keys[K_DOWN]:
            self.camera_angle_x += 0.02
        
        # Zoom
        if keys[K_PLUS] or keys[K_EQUALS]:
            self.zoom += 0.1
        if keys[K_MINUS]:
            self.zoom = max(0.5, self.zoom - 0.1)
        
        # View mode toggle
        if keys[K_v]:
            if self.view_mode == "wireframe":
                self.view_mode = "slice"
            else:
                self.view_mode = "wireframe"
        
        # Slice depth (for slice view)
        if self.view_mode == "slice":
            if keys[K_z]:
                self.slice_depth = max(0, self.slice_depth - 1)
            if keys[K_x]:
                self.slice_depth = min(GRID_SIZE - 1, self.slice_depth + 1)
        
        # Reset
        if keys[K_r]:
            self.frequency = DEFAULT_FREQUENCY
            self.amplitude = DEFAULT_AMPLITUDE
            self.phase_speed = DEFAULT_PHASE_SPEED
            self.time = 0.0
        
        # Pause
        if keys[K_SPACE]:
            self.paused = not self.paused
        
        # Number keys for presets
        if keys[K_1]:
            # Sphere preset
            self.frequency = 0.1
            self.amplitude = 0.3
            self.threshold = 0.6
        if keys[K_2]:
            # Morphing cube preset
            self.frequency = 0.2
            self.amplitude = 0.5
            self.threshold = 0.5
        if keys[K_3]:
            # High frequency preset
            self.frequency = 0.25
            self.amplitude = 0.7
            self.threshold = 0.4
        
        for event in pygame.event.get():
            if event.type == QUIT:
                self.running = False
    
    def update(self, dt):
        """Update simulation state."""
        if not self.paused:
            self.time += dt * 0.01  # Slow down time for visualization
            
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
    
    def draw_wireframe(self):
        """Draw 3D wireframe visualization of voxels."""
        # Draw voxels as small cubes
        voxel_count = 0
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                for z in range(GRID_SIZE):
                    if self.voxel_grid[x, y, z]:
                        voxel_count += 1
                        # Project 3D to 2D
                        sx, sy, scale = project_3d_to_2d(
                            x, y, z, 
                            self.camera_angle_x, 
                            self.camera_angle_y, 
                            self.zoom
                        )
                        
                        # Draw small square representing voxel
                        size = max(1, int(scale * 0.8))
                        if 0 <= sx < WINDOW_WIDTH and 0 <= sy < WINDOW_HEIGHT:
                            # Color based on position for visual interest
                            color_intensity = int(100 + (z / GRID_SIZE) * 155)
                            color = (color_intensity, 150, 255)
                            pygame.draw.rect(self.screen, color, 
                                           (sx - size//2, sy - size//2, size, size))
        
        return voxel_count
    
    def draw_slice_view(self):
        """Draw 2D slice visualization."""
        # Draw XY slice at current slice depth
        cell_size = min(WINDOW_WIDTH, WINDOW_HEIGHT) // (GRID_SIZE + 2)
        start_x = (WINDOW_WIDTH - GRID_SIZE * cell_size) // 2
        start_y = (WINDOW_HEIGHT - GRID_SIZE * cell_size) // 2
        
        voxel_count = 0
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                if self.voxel_grid[x, y, self.slice_depth]:
                    voxel_count += 1
                    color_intensity = int(100 + (self.slice_depth / GRID_SIZE) * 155)
                    color = (color_intensity, 150, 255)
                    rect = (start_x + x * cell_size, 
                           start_y + y * cell_size, 
                           cell_size - 1, 
                           cell_size - 1)
                    pygame.draw.rect(self.screen, color, rect)
        
        # Draw slice indicator
        text = self.font.render(f"Slice Z={self.slice_depth}", True, WHITE)
        self.screen.blit(text, (10, WINDOW_HEIGHT - 60))
        
        return voxel_count
    
    def draw_ui(self, voxel_count):
        """Draw UI information."""
        # Background panel
        panel = pygame.Surface((300, 200))
        panel.set_alpha(200)
        panel.fill(BLACK)
        self.screen.blit(panel, (10, 10))
        
        # Text information
        info_lines = [
            f"FPS: {self.fps}",
            f"Voxels: {voxel_count} / {GRID_SIZE**3}",
            f"Time: {self.time:.2f}",
            f"",
            f"Frequency: {self.frequency:.3f} (W/S)",
            f"Amplitude: {self.amplitude:.2f} (A/D)",
            f"Phase Speed: {self.phase_speed:.3f} (Q/E)",
            f"Threshold: {self.threshold:.2f}",
            f"",
            f"View: {self.view_mode.upper()} (V to toggle)",
            f"Controls: Arrows=Rotate, +/-=Zoom",
            f"Space=Pause, R=Reset, 1-3=Presets"
        ]
        
        y_offset = 15
        for line in info_lines:
            if line:  # Skip empty lines
                text = self.font.render(line, True, WHITE)
                self.screen.blit(text, (20, y_offset))
            y_offset += 20
    
    def render(self):
        """Render everything to screen."""
        self.screen.fill(BLACK)
        
        # Draw voxels based on view mode
        if self.view_mode == "wireframe":
            voxel_count = self.draw_wireframe()
        else:
            voxel_count = self.draw_slice_view()
        
        # Draw UI
        self.draw_ui(voxel_count)
        
        pygame.display.flip()
    
    def run(self):
        """Main game loop."""
        print("Starting Morphogenic Voxel System...")
        print("Controls:")
        print("  W/S: Adjust frequency")
        print("  A/D: Adjust amplitude")
        print("  Q/E: Adjust phase speed")
        print("  Arrow Keys: Rotate camera")
        print("  +/-: Zoom in/out")
        print("  V: Toggle wireframe/slice view")
        print("  Z/X: Adjust slice depth (slice view)")
        print("  Space: Pause/Resume")
        print("  R: Reset parameters")
        print("  1-3: Preset configurations")
        
        while self.running:
            dt = self.clock.tick(60) / 16.67  # Normalize to ~60fps
            
            self.handle_input()
            self.update(dt)
            self.render()
        
        pygame.quit()


if __name__ == "__main__":
    system = VoxelMorphingSystem()
    system.run()

