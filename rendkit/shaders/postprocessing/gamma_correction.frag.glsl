#version 450 core

uniform sampler2D u_rendtex;
uniform float u_gamma;

in vec2 v_uv;
out vec4 out_color;

void main() {
    vec3 color = texture2D(u_rendtex, v_uv).rgb;
    color = pow(max(vec3(0), color), vec3(1.0/u_gamma));
    out_color = vec4(color, 1.0);
}
