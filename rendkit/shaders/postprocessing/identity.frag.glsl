#version 450 core

uniform sampler2D u_rendtex;

in vec2 v_uv;
out vec4 out_color;

void main() {
    vec3 color = texture2D(u_rendtex, v_uv).rgb;
    out_color = vec4(color, 1.0);
}
