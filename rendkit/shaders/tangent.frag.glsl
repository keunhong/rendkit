#version 450 core

in vec3 v_tangent;
in vec3 v_position;
out out_color;

void main(void) {
    out_color = vec4(normalize(v_tangent), 1.0);
}
