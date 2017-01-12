#version 120

varying vec3 v_tangent;
varying vec3 v_position;

void main(void) {
    gl_FragColor = vec4(normalize(v_tangent), 1.0);
}
