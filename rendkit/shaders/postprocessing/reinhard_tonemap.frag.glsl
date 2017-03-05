#version 450 core

uniform sampler2D u_rendtex;
uniform float u_thres;

in vec2 v_uv;
out vec4 out_color;

void main() {
    vec3 L = texture2D(u_rendtex, v_uv).rgb;
    float thres2 = u_thres * u_thres;
    L = (L * (1 + L / thres2)) / (vec3(1.0) + L);
    out_color = vec4(L, 1.0);
}
