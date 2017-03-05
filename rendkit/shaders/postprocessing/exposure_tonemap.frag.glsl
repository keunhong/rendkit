#version 450 core

uniform sampler2D u_rendtex;
uniform float u_exposure;

in vec2 v_uv;
out vec4 out_color;

void main() {
    vec3 color = texture2D(u_rendtex, v_uv).rgb;
    color = vec3(1.0) - exp(-color * u_exposure);
    out_color = vec4(color, 1.0);
}
