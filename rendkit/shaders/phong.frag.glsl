#version 150
#include "utils/math.glsl"
#include "utils/sampling.glsl"

#define LIGHT_POINT 0
#define LIGHT_DIRECTIONAL 1

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

#if TPL.use_radiance_map
uniform samplerCube u_irradiance_map;
uniform samplerCube u_radiance_map;
#endif

const float NUM_LIGHTS = TPL.num_lights;

vec3 irradiance(vec3 N, vec3 L, vec3 light_color) {
  float cosine_term = max(.0, dot(N, L));
  return cosine_term * light_color;
}

vec2 compute_sample_angles(vec2 xi) {
  float phi = 2.0f * M_PI * xi.x;
  float theta = acos(pow(1 - xi.y, 1.0/(u_shininess + 1)));
  return vec2(phi, theta);
}


void main() {
  vec3 V = normalize(u_cam_pos - v_position);

	bool is_back_facing = dot(v_normal, V) < 0;
	if (is_back_facing) {
		gl_FragColor = vec4(0.0);
		return;
	}

	vec3 total_radiance = vec3(0.0);

  #if TPL.use_radiance_map
  total_radiance += u_diff * texture(u_irradiance_map, v_normal).rgb;
  vec3 specular = vec3(0);
  float r1 = rand(v_position.xy);
  float r2 = rand(vec2(r1));
  int N_SAMPLES = 128;
  for (int i = 0; i < N_SAMPLES; i++) {
    vec2 xi = vec2(rand(vec2(r1 + i, r2 + i)));
    vec2 sample_angle = compute_sample_angles(xi);
    float phi = sample_angle.x;
    float theta = sample_angle.y;
    vec3 H = sample_to_world(phi, theta, v_normal);
    vec3 L = reflect(-V, H);
    vec3 light_color = texture(u_radiance_map, L).rgb;
    float ndotl = max(0.0, dot(v_normal, L));
		vec3 refl_dir = normalize(2.0 * ndotl * v_normal - L);
		float rdotv = max(0.0, dot(refl_dir, V));
		vec3 Is = u_spec * pow(rdotv, u_shininess);
    specular += irradiance(v_normal, L, light_color) * Is * ndotl;
  }
  specular /= N_SAMPLES;
  total_radiance += specular;
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
	gl_FragColor = vec4(total_radiance, 1.0);
}
