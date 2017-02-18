#version 120

attribute vec2 a_position;
attribute vec2 a_uv;
varying vec2 v_uv;

void main() {
    v_uv = a_position;
    gl_Position = vec4(a_position, 0.0, 1.0);
}
