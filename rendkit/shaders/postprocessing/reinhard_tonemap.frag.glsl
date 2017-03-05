#version 120

uniform sampler2D u_rendtex;
varying vec2 v_uv;

void main() {
    vec3 color = texture2D(u_rendtex, v_uv).rgb;
    color = color / (vec3(1.0) + color);
    gl_FragColor = vec4(color, 1.0);
}
