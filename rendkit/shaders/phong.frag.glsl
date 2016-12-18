#version 120

#define LIGHT_POINT 0
#define LIGHT_DIRECTIONAL 1

varying vec3 v_position;
varying vec3 v_normal;

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
uniform sampler2D u_radiance_map;
uniform vec2 u_radiance_map_size;
#endif

const float NUM_LIGHTS = TPL.num_lights;

vec3 irradiance(vec3 N, vec3 L, vec3 light_color) {
  float cosine_term = max(.0, dot(N, L));
  return cosine_term * light_color;
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
  float radiance_map_area = u_radiance_map_size.x * u_radiance_map_size.y;
  for (float s = 0; s <= 1.0; s += 1.0/u_radiance_map_size.s) {
    for (float t = 0; t <= 1.0; t += 1.0/u_radiance_map_size.t) {
      float x = s * 2 - 1;
      float z = t * 2 - 1;
      if (x*x + z*z <= 1) {
        vec2 samp_ind = vec2((z+1)/2, (x+1)/2);
        vec3 L = vec3(x, sqrt(max(0, 1 - x*x - z*z)), z);
        vec3 light_color = texture2D(u_radiance_map, samp_ind).rgb;
        vec3 irradiance = irradiance(v_normal, L, light_color);
        vec3 radiance = u_diff * irradiance;
        total_radiance += radiance / radiance_map_area;
      }
    }
  }

  vec3 Q = reflect(-V, v_normal);
  float samp_radius = 0.25;
  float spec_area = radiance_map_area / (4.0 * samp_radius * samp_radius);
  for (float s = 0; s <= 1.0; s += 1.0/u_radiance_map_size.s) {
    for (float t = 0; t <= 1.0; t += 1.0/u_radiance_map_size.t) {
      float x = Q.x + (2 * s - 1) * samp_radius;
      float z = Q.z + (2 * t - 1) * samp_radius;
      if (x*x + z*z <= 1) {
        vec2 samp_ind = vec2((z + 1.0) / 2.0,
                             (x + 1.0) / 2.0);
        vec3 L = vec3(x, sqrt(max(0, 1 - x*x - z*z)), z);
        vec3 H = normalize(L + V);
        vec3 light_color = texture2D(u_radiance_map, samp_ind).rgb;
        float ndoth = max(0.0, dot(v_normal, H));
        vec3 Is = u_spec * pow(ndoth, u_shininess);
        vec3 irradiance = irradiance(v_normal, L, light_color);
        vec3 radiance = Is * irradiance;
        total_radiance += radiance / spec_area;
      }
    }
  }
  // TODO: Lower hemisphere.
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
