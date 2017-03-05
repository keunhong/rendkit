#version 150
#include "cubemap/dual_paraboloid.glsl"

uniform samplerCube u_cubemap;
uniform vec2 u_cubemap_size;
uniform int u_hemisphere;
in vec2 v_uv;

void main() {
  vec3 samp_vec = dualp_tex_to_world(v_uv, u_hemisphere, 1.2);
  gl_FragColor = vec4(texture(u_cubemap, samp_vec).xyz, 1.0);
}
