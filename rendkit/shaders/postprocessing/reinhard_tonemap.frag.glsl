#version 120

uniform sampler2D u_rendtex;
uniform float u_thres;
varying vec2 v_uv;

void main() {
    vec3 L = texture2D(u_rendtex, v_uv).rgb;
    float thres2 = u_thres * u_thres;
    L = (L * (1 + L / thres2)) / (vec3(1.0) + L);
    gl_FragColor = vec4(L, 1.0);
}
