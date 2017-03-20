#version 450 core
#include "utils/math.glsl"
#include "utils/sampling.glsl"
#include "envmap/cubemap.glsl"

uniform samplerCube u_radiance_map;
uniform vec2 u_cubemap_size;
uniform int u_cube_face;

in vec2 v_uv;
out vec4 out_color;


vec3 importance_sample(vec2 xi) {
  float phi = 2.0f * M_PI * xi.x;
  float cos_theta = 1.0 - xi.y / (2 * M_PI);
  float sin_theta = sqrt(1 - cos_theta * cos_theta);

  vec3 H;
  H.x = sin_theta * cos(phi);
  H.y = sin_theta * sin(phi);
  H.z = cos_theta;
  return H;
}


// Adapted from http://www.codinglabs.net/article_physically_based_rendering.aspx
vec4 samp(vec2 pos, int cube_face) {
  vec3 normal = cubemap_face_to_world(pos, cube_face);
  vec3 up = vec3(0, 1, 0);
  vec3 right = normalize(cross(up, normal));
  up = cross(normal, right);

  vec3 total_color = vec3(0.0);
  uint N_SAMPLES = 4096u;
  for (uint i = 0u; i < N_SAMPLES; i++) {
    vec2 xi = hammersley(i, N_SAMPLES); // Use psuedo-random point set.
    vec3 L = importance_sample(xi);
    L = sample_to_world(L, normal);
    float pdf = L.z;
		float lod = compute_lod(pdf, N_SAMPLES, u_cubemap_size.x, u_cubemap_size.y);
    vec3 light_color = textureLod(u_radiance_map, L, lod).rgb;
    total_color += 1.0 / float(N_SAMPLES) * light_color;
  }
  return vec4(total_color, 1.0);
}


void main() {
  out_color = vec4(samp(v_uv, u_cube_face));
}
