#version 450 core

in vec3 v_bitangent;
in vec3 v_position;
out vec4 out_color;

void main(void) {
    out_color = vec4(normalize(v_bitangent), 1.0);
}
