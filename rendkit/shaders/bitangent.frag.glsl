#version 120

varying vec3 v_bitangent;
varying vec3 v_position;

void main(void) {
    gl_FragColor = vec4(normalize(v_bitangent), 1.0);
}
