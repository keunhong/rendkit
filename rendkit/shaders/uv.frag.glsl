#version 120

varying vec2 v_uv;

void main() {
	gl_FragColor = vec4(0.0, v_uv, 1.0);
}
