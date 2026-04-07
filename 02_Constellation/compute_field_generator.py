"""
Compute Shader Field Generator
Replaces CPU-side Numba field generation with GPU compute shaders for better performance.
"""

import moderngl
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


class ComputeFieldGenerator:
    """
    GPU-accelerated field generator using compute shaders.
    Falls back to CPU generation if compute shaders aren't supported.
    """
    
    def __init__(self, ctx: moderngl.Context, grid_size: int):
        """
        Initialize compute field generator.
        
        Args:
            ctx: ModernGL context
            grid_size: Grid size (16, 32, 64, 128, etc.)
        """
        self.ctx = ctx
        self.grid_size = grid_size
        self.use_compute_shaders = False
        self.wave_compute = None
        self.shape_compute = None
        self.wave_field_texture = None
        self.shape_field_texture = None
        
        # Check if compute shaders are supported
        if self._check_compute_support():
            try:
                self._init_compute_shaders()
                self.use_compute_shaders = True
                print(f"[OK] Compute shaders enabled for {grid_size}^3 grid")
            except Exception as e:
                print(f"[WARNING] Compute shader initialization failed: {e}")
                print("   Falling back to CPU generation")
                self.use_compute_shaders = False
        else:
            print("[WARNING] Compute shaders not supported (OpenGL 4.3+ required)")
            print("   Falling back to CPU generation")
    
    def _check_compute_support(self) -> bool:
        """Check if compute shaders are supported."""
        try:
            # Check OpenGL version (compute shaders require 4.3+)
            gl_version = self.ctx.info.get('GL_VERSION', '')
            if gl_version:
                major = int(gl_version.split('.')[0])
                minor = int(gl_version.split('.')[1].split()[0])
                if major < 4 or (major == 4 and minor < 3):
                    return False
            
            # Check extension
            extensions = self.ctx.info.get('GL_EXTENSIONS', '')
            if 'GL_ARB_compute_shader' not in extensions and 'compute_shader' not in str(self.ctx.extensions):
                # Try to create a simple compute shader to test
                try:
                    test_source = """
#version 430 core
layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
void main() {}
"""
                    test_program = self.ctx.compute_shader(test_source)
                    test_program.release()
                    return True
                except:
                    return False
            return True
        except:
            return False
    
    def _init_compute_shaders(self):
        """Initialize compute shaders and textures."""
        shader_dir = Path(__file__).parent / "shaders"
        
        # Load compute shader sources
        wave_shader_path = shader_dir / "compute_wave_field.comp"
        shape_shader_path = shader_dir / "compute_shape_field.comp"
        
        if not wave_shader_path.exists():
            raise FileNotFoundError(f"Wave field compute shader not found: {wave_shader_path}")
        if not shape_shader_path.exists():
            raise FileNotFoundError(f"Shape field compute shader not found: {shape_shader_path}")
        
        with open(wave_shader_path) as f:
            wave_shader_source = f.read()
        
        with open(shape_shader_path) as f:
            shape_shader_source = f.read()
        
        # Compile compute shaders
        self.wave_compute = self.ctx.compute_shader(wave_shader_source)
        self.shape_compute = self.ctx.compute_shader(shape_shader_source)
        
        # Create 3D textures for field storage
        grid_tuple = (self.grid_size, self.grid_size, self.grid_size)
        
        self.wave_field_texture = self.ctx.texture3d(
            size=grid_tuple,
            components=1,
            dtype='f4'
        )
        
        self.shape_field_texture = self.ctx.texture3d(
            size=grid_tuple,
            components=1,
            dtype='f4'
        )
    
    def generate_wave_field(self, time: float, frequency: float, amplitude: float, phase_speed: float) -> np.ndarray:
        """
        Generate wave field using compute shader or CPU fallback.
        
        Returns:
            numpy array of shape (grid_size, grid_size, grid_size)
        """
        if not self.use_compute_shaders:
            # CPU fallback handled by caller
            return None
        
        try:
            # Bind texture to image unit for writing
            self.wave_field_texture.bind_to_image(0, read=False, write=True, level=0)
            
            # Set uniforms
            self.wave_compute['time'] = time
            self.wave_compute['frequency'] = frequency
            self.wave_compute['amplitude'] = amplitude
            self.wave_compute['phase_speed'] = phase_speed
            self.wave_compute['grid_size'] = self.grid_size
            
            # Calculate dispatch groups (8x8x8 local size)
            local_size = 8
            gx = (self.grid_size + local_size - 1) // local_size
            gy = (self.grid_size + local_size - 1) // local_size
            gz = (self.grid_size + local_size - 1) // local_size
            
            # Dispatch compute shader (no .use() needed for compute shaders in ModernGL)
            self.wave_compute.run(gx, gy, gz)
            
            # Memory barrier - CRITICAL for 3D textures!
            self.ctx.memory_barrier(
                moderngl.SHADER_IMAGE_ACCESS_BARRIER_BIT |
                moderngl.TEXTURE_FETCH_BARRIER_BIT
            )
            
            # Note: Reading back from GPU is expensive!
            # For Phase 1, we'll use CPU fallback. In Phase 2, we can use textures directly.
            # For now, return None to indicate CPU fallback should be used
            # (Reading back defeats the purpose of GPU compute)
            return None
            
        except Exception as e:
            print(f"[WARNING] Compute shader wave field generation failed: {e}")
            return None
    
    def generate_shape_field(self, shape_type: int, shape_center: Tuple[float, float, float], shape_scale: float = 1.0) -> Optional[np.ndarray]:
        """
        Generate shape field using compute shader.
        
        Args:
            shape_type: 0=sphere, 1=cube, 2=pyramid
            shape_center: Center position in grid space (normalized 0-1)
            shape_scale: Scale factor for the shape
        
        Returns:
            numpy array of shape (grid_size, grid_size, grid_size) or None if failed
        """
        if not self.use_compute_shaders:
            return None
        
        try:
            # Bind texture to image unit for writing
            self.shape_field_texture.bind_to_image(0, read=False, write=True, level=0)
            
            # Set uniforms
            self.shape_compute['grid_size'] = self.grid_size
            self.shape_compute['shape_type'] = shape_type
            self.shape_compute['shape_center'] = shape_center
            self.shape_compute['shape_scale'] = shape_scale
            
            # Calculate dispatch groups
            local_size = 8
            gx = (self.grid_size + local_size - 1) // local_size
            gy = (self.grid_size + local_size - 1) // local_size
            gz = (self.grid_size + local_size - 1) // local_size
            
            # Dispatch compute shader (no .use() needed for compute shaders in ModernGL)
            self.shape_compute.run(gx, gy, gz)
            
            # Memory barrier
            self.ctx.memory_barrier(
                moderngl.SHADER_IMAGE_ACCESS_BARRIER_BIT |
                moderngl.TEXTURE_FETCH_BARRIER_BIT
            )
            
            # Note: Reading back from GPU is expensive!
            # For Phase 1, we'll use CPU fallback. In Phase 2, we can optimize to use textures directly.
            return None
            
        except Exception as e:
            print(f"[WARNING] Compute shader shape field generation failed: {e}")
            return None
    
    def resize(self, new_grid_size: int):
        """Resize textures when grid size changes."""
        if not self.use_compute_shaders:
            return
        
        self.grid_size = new_grid_size
        
        # Release old textures
        if self.wave_field_texture:
            self.wave_field_texture.release()
        if self.shape_field_texture:
            self.shape_field_texture.release()
        
        # Create new textures
        grid_tuple = (new_grid_size, new_grid_size, new_grid_size)
        self.wave_field_texture = self.ctx.texture3d(
            size=grid_tuple,
            components=1,
            dtype='f4'
        )
        self.shape_field_texture = self.ctx.texture3d(
            size=grid_tuple,
            components=1,
            dtype='f4'
        )
    
    def release(self):
        """Clean up resources."""
        if self.wave_compute:
            self.wave_compute.release()
        if self.shape_compute:
            self.shape_compute.release()
        if self.wave_field_texture:
            self.wave_field_texture.release()
        if self.shape_field_texture:
            self.shape_field_texture.release()
