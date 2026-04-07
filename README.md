# Morphogenic System

A real-time 3D voxel morphing engine exploring mathematical feedback loops through GPU-accelerated rendering. Built in three stages, each expanding on the last.

---

## The three stages

### Stage 1 — `01_Voxel_Base/voxel_morphing_system.py`
2D wireframe visualization of a 32×32×32 voxel grid driven by sine wave interference.

- Three orthogonal sine waves combine with phase offsets to create a dynamic field
- Voxels spawn/despawn based on field threshold — with a configurable feedback loop: existing voxels lower the spawn threshold (hysteresis), making active voxels more likely to persist
- Real-time parameter controls: frequency, amplitude, phase speed, threshold
- Dual view: 3D wireframe or 2D cross-section slice
- Numba JIT-compiled core for performance on the full 32,768-voxel grid

### Stage 1 — `01_Voxel_Base/voxel_morphing_3d.py`
Same system, upgraded to full 3D solid rendering via ModernGL.

- GPU instanced rendering — one 24-vertex cube mesh drawn up to 32,768 times per frame
- Per-face visibility culling: checks 6 adjacent voxels, renders only exposed faces
- Directional lighting with ambient + diffuse, gamma correction
- Per-face color coding in the fragment shader
- Free-fly camera (WASD + arrows) with yaw/pitch, no gimbal lock
- Fixed 32×32×32 grid

### Stage 2 — `02_Constellation/constellation_system.py`
Multi-target organic shape morphing with a constellation layout.

- 6 primitive shapes: Sphere, Cube, Pyramid, Torus, Cylinder, Octahedron (all Numba-compiled)
- Shapes arranged in a circular constellation — system cycles through them as morph targets
- **Smoothstep easing** (`t² × (3 - 2t)`) for organic transitions between shapes
- Wave field + shape field composited with influence-weighted blending
- Camera-facing face culling via dot product test — only renders faces whose normals point toward the camera
- Dynamic grid sizes: 16–128 (adjustable quality/performance tradeoff)
- GPU compute shader infrastructure with graceful CPU fallback for compatibility

---

## Running it

```bash
pip install -r requirements.txt
```

**Stage 1 wireframe:**
```bash
cd 01_Voxel_Base
python voxel_morphing_system.py
```

**Stage 1 3D solid:**
```bash
cd 01_Voxel_Base
python voxel_morphing_3d.py
```

**Stage 2 constellation:**
```bash
cd 02_Constellation
python constellation_system.py
```

---

## Structure

```
01_Voxel_Base/
├── voxel_morphing_system.py   # 2D wireframe + sine wave field (408 lines)
├── voxel_morphing_3d.py       # 3D GPU instanced solid rendering (671 lines)
├── voxel_shaders.frag/.vert   # GLSL shaders for 3D version
└── requirements.txt

02_Constellation/
├── constellation_system.py    # Multi-target morphing engine (1,480 lines)
├── shape_library.py           # 6 Numba-compiled shape field functions (152 lines)
├── wireframe_generator.py     # Parametric wireframe geometry (198 lines)
├── compute_field_generator.py # GPU compute shader + CPU fallback (247 lines)
└── shaders/                   # GLSL compute + render shaders
```

**~3,150 lines total.**

---

## Requirements

- Python 3.8+
- ModernGL (OpenGL 3.3+)
- Numba + NumPy
- Pygame CE
