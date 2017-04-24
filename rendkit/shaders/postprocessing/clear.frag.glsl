#version 450 core

uniform sampler2D u_rendtex;
uniform vec4 u_clear_color;

in vec2 v_uv;
out vec4 out_color;

void main() {
    vec4 color = texture2D(u_rendtex, v_uv);
    color.rgb = u_clear_color.rgb * (1.0 - color.w) + color.rgb * color.w;
    color.w = 1.0;
    out_color = color;
}
