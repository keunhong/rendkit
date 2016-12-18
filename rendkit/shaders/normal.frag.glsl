#version 120

varying vec3 v_normal;

void main(void) {
    gl_FragColor = vec4(normalize(v_normal), 1.0);
}
