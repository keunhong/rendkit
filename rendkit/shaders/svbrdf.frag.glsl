#version 150
#include "brdf/aittala.glsl"
#include "math.glsl"

#define LIGHT_POINT 0
#define LIGHT_DIRECTIONAL 1
#define LIGHT_AMBIENT 2

uniform sampler2D u_diff_map;
uniform sampler2D u_spec_map;
uniform sampler2D u_spec_shape_map;
uniform sampler2D u_normal_map;
uniform sampler2D u_theta_cdf;
uniform vec2 u_sigma_range;
uniform vec3 u_cam_pos;
in vec3 v_position;
in vec3 v_normal;
in vec3 v_tangent;
in vec3 v_bitangent;
in vec2 v_uv;

uniform float alpha;
#if TPL.num_lights > 0
uniform float u_light_intensity[TPL.num_lights];
uniform vec3 u_light_position[TPL.num_lights];
uniform vec3 u_light_color[TPL.num_lights];
uniform int u_light_type[TPL.num_lights];
#endif

#if TPL.use_radiance_map
uniform samplerCube u_irradiance_map;
uniform samplerCube u_radiance_map;
#endif

const float NUM_LIGHTS = TPL.num_lights;
const float M_PI = 3.14;


highp float rand(vec2 co) {
    highp float a = 12.9898;
    highp float b = 78.233;
    highp float c = 43758.5453;
    highp float dt= dot(co.xy ,vec2(a,b));
    highp float sn= mod(dt,3.14);
    return fract(sin(sn) * c);
}

vec3 compute_irradiance(vec3 N, vec3 L, vec3 light_color) {
  float cosine_term = max(.0, dot(N, L));
  return cosine_term * max(vec3(0.0), light_color);
}

vec2 compute_sample_angles(float sigma, vec2 xi) {
  float phi = 2.0f * M_PI * xi.x;
  float sigma_samp = (sigma - u_sigma_range.x) / (u_sigma_range.y - u_sigma_range.x);
  float theta = texture2D(u_theta_cdf, vec2(xi.y, sigma_samp)).r;
//  float theta = M_PI/2 * xi.y;
  return vec2(phi, theta);
}

vec3 sample_to_world(float phi, float theta, vec3 N) {
    vec3 H;

    H.x = sin(theta) * cos(phi);
    H.y = sin(theta) * sin(phi);
    H.z = cos(theta);

    vec3 up_vec = abs(N.z) < 0.999 ? vec3(0,0,1) : vec3(1,0,0);
    vec3 tangent_x = normalize(cross(up_vec, N));
    vec3 tangent_y = cross(N, tangent_x);

    return tangent_x * H.x + tangent_y * H.y + N * H.z;
}


void main() {
  vec3 V = normalize(u_cam_pos - v_position);

  vec3 rho_d = texture2D(u_diff_map, v_uv).rgb;
  vec3 rho_s = texture2D(u_spec_map, v_uv).rgb;
  vec3 specv = texture2D(u_spec_shape_map, v_uv).rgb;

  mat3 TBN = mat3(v_tangent, v_bitangent, v_normal);
  vec3 N = normalize(TBN * texture2D(u_normal_map, v_uv).rgb);

  // Flip normal if back facing.
//  bool is_back_facing = dot(V, v_normal) < 0;
//  if (is_back_facing) {
//    N *= -1;
//  }

  mat2 S = mat2(specv.x, specv.z,
      specv.z, specv.y);

  vec3 total_radiance = vec3(0.0);

  #if TPL.use_radiance_map
  total_radiance += rho_d * texture(u_irradiance_map, N).rgb;

  vec3 specular = vec3(0);
  float r1 = rand(v_uv);
  float r2 = rand(vec2(r1));
  float sigma = tr(S) / 2;
  int N_SAMPLES = 128;
  for (int i = 0; i < N_SAMPLES; i++) {
    vec2 xi = vec2(rand(vec2(r1 + i, r2 + i)));
    vec2 sample_angle = compute_sample_angles(sigma, xi);
    float phi = sample_angle.x;
    float theta = sample_angle.y;
    vec3 H = sample_to_world(phi, theta, N);
    vec3 L = reflect(-V, H);
    vec3 light_color = texture(u_radiance_map, L).rgb;
    specular += compute_irradiance(N, L, light_color) *
      aittala_spec(N, V, L, rho_s, S, alpha);
  }
  specular /= N_SAMPLES;
  total_radiance += specular;
  #endif

  #if TPL.num_lights > 0
  for (int i = 0; i < NUM_LIGHTS; i++) {
    vec3 irradiance = vec3(0);
    if (u_light_type[i] == LIGHT_AMBIENT) {
      irradiance = u_light_intensity[i] * u_light_color[i];
    } else {
      vec3 L;
      float attenuation = 1.0;
      if (u_light_type[i] == LIGHT_POINT) {
        L = u_light_position[i] - v_position;
        attenuation = 1.0 / dot(L, L);
        L = normalize(L);
      } else if (u_light_type[i] == LIGHT_DIRECTIONAL) {
        L = normalize(u_light_position[i]);
      } else {
        continue;
      }
      bool is_light_visible = dot(L, N) >= 0;
      if (is_light_visible) {
        irradiance = compute_irradiance(N, L, u_light_intensity[i] * u_light_color[i]);
        total_radiance += aittala_spec(N, V, L, rho_s, S, alpha) * irradiance;
      }
    }
    total_radiance += rho_d * irradiance;
  }
  #endif

  gl_FragColor = vec4(max(vec3(.0), total_radiance), 1.0);    // rough gamma
}
