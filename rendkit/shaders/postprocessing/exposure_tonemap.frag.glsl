#version 120

uniform sampler2D u_rendtex;
uniform float u_exposure;
varying vec2 v_texcoord;

void main() {
    vec3 color = texture2D(u_rendtex, v_texcoord).rgb;
    color = vec3(1.0) - exp(-color * u_exposure);
    gl_FragColor = vec4(color, 1.0);
}
