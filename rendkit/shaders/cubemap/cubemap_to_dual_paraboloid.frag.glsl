#version 150

uniform samplerCube u_cubemap;
uniform vec2 u_cubemap_size;
uniform int u_hemisphere;
in vec2 v_uv;

vec3 uv_to_world(vec2 uv, int hemisphere) {
  float b = 1.2;
  float b2 = b * b;
  float s = uv.x - 1/2;
  float t = uv.y - 1/2;
  float s2 = s * s;
  float t2 = t * t;
  float denom = (4*b2*s2 + 4*b2*t2 + 1);
  vec3 v;
  v.x = (4 * b * s) / denom;
  v.y = (4 * b * t) / denom;
  v.z = sqrt(1 - v.x*v.x - v.y*v.y);
  if (hemisphere == 1) {
    v.z *= -1;
  }
  return normalize(v);
}

void main() {
  vec3 samp_vec = uv_to_world(v_uv * 2.0 - 1.0, u_hemisphere);
//  gl_FragColor = vec4(v_uv, 0.0, 1.0);
  gl_FragColor = vec4(texture(u_cubemap, samp_vec).xyz, 1.0);
}
