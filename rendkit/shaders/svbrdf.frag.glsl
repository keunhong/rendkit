#version 120

#define LIGHT_POINT 0
#define LIGHT_DIRECTIONAL 1
#define LIGHT_AMBIENT 2

uniform sampler2D diff_map;
uniform sampler2D spec_map;
uniform sampler2D spec_shape_map;
uniform sampler2D normal_map;
uniform vec3 u_cam_pos;
varying vec3 v_position;
varying vec3 v_normal;
varying vec3 v_tangent;
varying vec3 v_bitangent;
varying vec2 v_uv;

uniform float alpha;
#if TPL.num_lights > 0
uniform float u_light_intensity[TPL.num_lights];
uniform vec3 u_light_position[TPL.num_lights];
uniform vec3 u_light_color[TPL.num_lights];
uniform int u_light_type[TPL.num_lights];
#endif

#if TPL.use_radiance_map
uniform samplerCube u_irradiance_map;
#endif

const float NUM_LIGHTS = TPL.num_lights;
const float F0 = 0.04;


float fresnel_schlick(float F0, vec3 V, vec3 H) {
  float VdotH = max(0, dot(V, H));
  float F = F0 + (1.0 - F0) * pow(1.0 - VdotH, 5.0);
  return max(F, 0);
}


vec3 compute_irradiance(vec3 N, vec3 L, vec3 light_color) {
  float cosine_term = max(.0, dot(N, L));
  return cosine_term * max(vec3(0.0), light_color);
}


vec3 spec_reflectance(vec3 N, vec3 V, vec3 light_dir, vec3 alb_s, mat3 R, mat2 S) {
  vec3 H = normalize(light_dir + V);

  // Halfway vector in normal-oriented coordinates (so normal is [0,0,1])
  vec3 H_ = H + R * H + 1.0 / (N.z + 1.0) * (R * (R * H));
  // h = tangent plane parametrized half-vector
  vec2 h = H_.xy / H_.z;

  vec2 hT_S = h * S;
  float hT_S_h = dot(hT_S, h); // h^T S h

  // Aittala (2015) NDF.
  float D = exp(-pow(abs(hT_S_h), alpha / 2));

  float F = fresnel_schlick(F0, V, H);
  float spec_term = D * F / (F0 * max(.0, dot(light_dir, H)) );
  return spec_term * alb_s;
}


void main() {
  vec3 V = normalize(u_cam_pos - v_position);

  vec3 alb_d = texture2D(diff_map, v_uv).rgb;
  vec3 alb_s = texture2D(spec_map, v_uv).rgb;
  vec3 specv = texture2D(spec_shape_map, v_uv).rgb;

  mat3 TBN = mat3(v_tangent, v_bitangent, v_normal);
  vec3 N = normalize(TBN * texture2D(normal_map, v_uv).rgb);

  // Flip normal if back facing.
//  bool is_back_facing = dot(V, v_normal) < 0;
//  if (is_back_facing) {
//    N *= -1;
//  }

  mat3 R = mat3(0, 0, N.x,
      0, 0, N.y,
      -N.x, -N.y, 0);

  mat2 S = mat2(specv.x, specv.z,
      specv.z, specv.y);

  vec3 total_radiance = vec3(0.0);

  #if TPL.use_radiance_map
  total_radiance += alb_d * textureCube(u_irradiance_map, N).rgb;

//  vec3 Q = reflect(-V, v_normal);
//  float samp_radius = 0.5;
//  float spec_area = radiance_map_area / (4.0 * samp_radius * samp_radius);
//  for (float s = 0; s <= 1.0; s += 1.0/u_radiance_map_size.s) {
//    for (float t = 0; t <= 1.0; t += 1.0/u_radiance_map_size.t) {
//      float x = Q.x + (2 * s - 1) * samp_radius;
//      float z = Q.z + (2 * t - 1) * samp_radius;
//      if (x*x + z*z <= 1) {
//        vec2 samp_ind = vec2((z + 1.0) / 2.0,
//                             (x + 1.0) / 2.0);
//        vec3 L = vec3(x, sqrt(max(0, 1 - x*x - z*z)), z);
//        vec3 light_color = texture2D(u_radiance_map, samp_ind).rgb;
//        total_radiance += spec_reflectance(N, V, L, alb_s, R, S)
//          * compute_irradiance(N, L, light_color) / spec_area;
//      }
//    }
//  }
  // TODO: Lower hemisphere.
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
        total_radiance += spec_reflectance(N, V, L, alb_s, R, S) * irradiance;
      }
    }
    total_radiance += alb_d * irradiance;
  }
  #endif

  gl_FragColor = vec4(max(vec3(.0), total_radiance), 1.0);    // rough gamma
}
