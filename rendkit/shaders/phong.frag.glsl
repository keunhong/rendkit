#version 450 core
#include "utils/math.glsl"
#include "utils/sampling.glsl"
#include "envmap/dual_paraboloid.glsl"

#define LIGHT_POINT 0
#define LIGHT_DIRECTIONAL 1

out vec4 out_color;

in vec3 v_position;
in vec3 v_normal;

uniform vec3 u_cam_pos;
uniform vec3 u_diff;
uniform vec3 u_spec;
uniform float u_shininess;

#if TPL.num_lights > 0
uniform float u_light_intensity[TPL.num_lights];
uniform vec3 u_light_position[TPL.num_lights];
uniform vec3 u_light_color[TPL.num_lights];
uniform int u_light_type[TPL.num_lights];
#endif

#if TPL.num_shadow_sources > 0
#include "utils/shadow.glsl"
uniform sampler2DShadow u_shadow_depth[TPL.num_shadow_sources];
in vec4 v_position_shadow[TPL.num_shadow_sources];
#endif

#if TPL.use_radiance_map
uniform samplerCube u_irradiance_map;
uniform sampler2D u_radiance_upper;
uniform sampler2D u_radiance_lower;
uniform float u_radiance_scale;
uniform vec2 u_cubemap_size;
#endif

const float NUM_LIGHTS = TPL.num_lights;


vec3 importance_sample(vec2 xi, float shininess) {
  float phi = 2.0f * M_PI * xi.x;
  float cos_theta = pow(1 - xi.y, 1.0/(shininess + 1.0));
  float sin_theta = sqrt(1.0 - cos_theta * cos_theta);

  vec3 H;
  H.x = sin_theta * cos(phi);
  H.y = sin_theta * sin(phi);
  H.z = cos_theta;
  return H;
}


float compute_pdf(vec3 H, float shininess) {
  float D = (shininess + 2.0) / (2.0 * M_PI)
              * clamp(pow(H.z, shininess), 0.0, 1.0);
  return D * H.z;
}


void main() {
  vec3 V = normalize(u_cam_pos - v_position);

	vec3 total_radiance = vec3(0.0);

  float shadowness = 0.0;
	#if TPL.num_shadow_sources > 0
	for (int i = 0; i < TPL.num_shadow_sources; i++) {
    shadowness += compute_shadow(v_position, v_position_shadow[i], u_shadow_depth[i]);
	}
  shadowness /= TPL.num_shadow_sources * 2.0;
	#endif

  #if TPL.use_radiance_map
  total_radiance += u_diff * texture(u_irradiance_map, v_normal).rgb;
  vec3 specular = vec3(0);
  uint N_SAMPLES = 256u;
  for (uint i = 0u; i < N_SAMPLES; i++) {
    vec2 xi = hammersley(i, N_SAMPLES); // Use psuedo-random point set.
    vec3 H = importance_sample(xi, u_shininess);
    H = local_to_world(H, v_normal);
    float pdf = compute_pdf(H, u_shininess);
		float lod = compute_lod(pdf, N_SAMPLES, u_cubemap_size.x, u_cubemap_size.y);

    vec3 L = reflect(-V, H);
    vec2 dp_uv = dualp_world_to_tex(L, 1.2);
    vec3 light_color;
    if (L.y > 0) {
      light_color = textureLod(u_radiance_upper, dp_uv, lod).rgb;
    } else {
      light_color = textureLod(u_radiance_lower, dp_uv, lod).rgb;
    }
    specular += u_spec / float(N_SAMPLES) * light_color;
  }
  total_radiance += specular;
  total_radiance *= u_radiance_scale;
  #endif

	#if TPL.num_lights > 0
	for (int i = 0; i < TPL.num_lights; i++) {
		vec3 L;
		float attenuation = 1.0;
		if (u_light_type[i] == LIGHT_POINT) {
			L = u_light_position[i] - v_position;
		    attenuation = 1.0 / dot(L, L);
		    L = normalize(L);
		} else if (u_light_type[i] == LIGHT_DIRECTIONAL) {
			L = normalize(u_light_position[i]);
		}

		float ndotl = max(0.0, dot(v_normal, L));
		vec3 refl_dir = normalize(2.0 * ndotl * v_normal - L);
		float rdotv = max(0.0, dot(refl_dir, V));

		vec3 Id = u_diff;
		vec3 Is = u_spec * pow(rdotv, u_shininess);
		vec3 irradiance = attenuation * u_light_intensity[i] * u_light_color[i];
		vec3 radiance = (Is + Id) * ndotl * irradiance;
		total_radiance += radiance;
	}
	#endif
	total_radiance *= (1.0 - shadowness);
	out_color = vec4(total_radiance, 1.0);
}
