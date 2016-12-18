#version 120

uniform sampler2D u_rendtex;
uniform float u_gamma;
varying vec2 v_texcoord;

void main() {
    vec3 color = texture2D(u_rendtex, v_texcoord).rgb;
    color = pow(max(vec3(0), color), vec3(1.0/u_gamma));
    gl_FragColor = vec4(color, 1.0);
}
