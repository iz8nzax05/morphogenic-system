#version 330 core

// Per-vertex attributes (cube geometry)
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in float aFaceID;  // Face identifier (0-5)

// Per-instance attributes (voxel position)
layout(location = 3) in vec3 aInstancePos;

// Uniforms
uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

// Outputs to fragment shader
out vec3 FragPos;
out vec3 FragNormal;
out float FaceID;

void main() {
    // Transform vertex position by instance position
    vec3 worldPos = aPos + aInstancePos;  // Full size cubes (1.0 unit) - no gaps between cubes
    vec4 pos = vec4(worldPos, 1.0);
    
    // Transform to screen space
    gl_Position = projection * view * model * pos;
    
    // Pass through normal and face ID
    FragPos = worldPos;
    FragNormal = aNormal;
    FaceID = aFaceID;
}

