#version 450 core
#include "cubemap/spheremap.glsl"

uniform sampler2D u_panorama;
uniform int u_cube_face;

in vec2 v_uv;
out vec4 out_color;

vec2 samp(vec2 pos, int cube_face) {
  vec3 normal = normalize(vec3(pos.xy, 1) );
  switch(cube_face) {
    case 0: normal = normalize(vec3(1, pos.y, -pos.x)); break;
    case 1: normal = normalize(vec3(-1, pos.y, pos.x)); break;
    case 2: normal = normalize(vec3(pos.x, 1, -pos.y)); break;
    case 3: normal = normalize(vec3(pos.x, -1, pos.y)); break;
    case 4: normal = normalize(vec3(pos.xy, 1)); break;
    case 5: normal = normalize(vec3(-pos.x, pos.y, -1)); break;
  }

  return sphere_world_to_tex(normal);
}


void main() {
  vec2 pos = v_uv * 2.0 - 1.0;
  vec3 color = texture(u_panorama, samp(pos, u_cube_face)).xyz;
  out_color = vec4(color, 1.0);
}
