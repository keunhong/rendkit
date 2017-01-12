#version 120

varying vec3 v_vector;
varying vec3 v_position;

void main(void) {
    gl_FragColor = vec4(normalize(v_vector), 1.0);
}
