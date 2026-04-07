#version 330 core

// Inputs from vertex shader
in vec3 FragPos;
in vec3 FragNormal;
in float FaceID;

// Output
out vec4 FragColor;

// Uniforms
uniform vec3 lightDir;      // Direction to light (normalized)
uniform vec3 lightColor;    // Light color (white)
uniform float ambientStrength;
uniform float diffuseStrength;

// Face colors (red, green, blue variations)
vec3 getFaceColor(float faceID) {
    if (faceID < 0.5) {        // Top face (0)
        return vec3(1.0, 0.3, 0.3);  // Red
    } else if (faceID < 1.5) { // Bottom face (1)
        return vec3(0.3, 0.1, 0.1);  // Dark red
    } else if (faceID < 2.5) { // Front face (2)
        return vec3(0.3, 1.0, 0.3);  // Green
    } else if (faceID < 3.5) { // Back face (3)
        return vec3(0.1, 0.3, 0.1);  // Dark green
    } else if (faceID < 4.5) { // Right face (4)
        return vec3(0.3, 0.3, 1.0);  // Blue
    } else {                    // Left face (5)
        return vec3(0.1, 0.1, 0.3);  // Dark blue
    }
}

void main() {
    // Get face color
    vec3 faceColor = getFaceColor(FaceID);
    
    // Normalize normal vector
    vec3 norm = normalize(FragNormal);
    vec3 lightDirection = normalize(lightDir);
    
    // Ambient lighting
    vec3 ambient = ambientStrength * lightColor;
    
    // Diffuse lighting (face orientation to light)
    float diff = max(dot(norm, lightDirection), 0.0);
    vec3 diffuse = diffuseStrength * diff * lightColor;
    
    // Combine lighting with face color
    vec3 result = (ambient + diffuse) * faceColor;
    
    // Gamma correction (optional, makes colors brighter)
    result = pow(result, vec3(1.0/2.2));
    
    FragColor = vec4(result, 1.0);
}
