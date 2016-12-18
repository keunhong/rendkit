#version 120

varying vec3 v_coord;

void main(void) {
    gl_FragColor = vec4(vec3(v_coord), 1.0);
}
