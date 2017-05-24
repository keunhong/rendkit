#version 450 core
#include "utils/math.glsl"
#include "brdf/fresnel.glsl"
#include "utils/sampling.glsl"
#include "envmap/dual_paraboloid.glsl"

#define LIGHT_POINT 0
#define LIGHT_DIRECTIONAL 1
#define LIGHT_AMBIENT 2

uniform sampler2D u_diff_map;
uniform sampler2D u_spec_map;
uniform sampler2D u_rough_map;
uniform sampler2D u_aniso_map;
uniform sampler2D u_normal_map;

uniform sampler2D u_cdf_sampler;
uniform sampler2D u_pdf_sampler; // Normalization factor for PDF.
uniform vec3 u_cam_pos;

#if TPL.num_lights > 0
uniform float u_light_intensity[TPL.num_lights];
uniform vec3 u_light_position[TPL.num_lights];
uniform vec3 u_light_color[TPL.num_lights];
uniform int u_light_type[TPL.num_lights];
#endif

#if TPL.use_radiance_map
uniform samplerCube u_irradiance_map;
uniform sampler2D u_radiance_upper;
uniform sampler2D u_radiance_lower;
uniform float u_radiance_scale;
uniform vec2 u_cubemap_size;
#endif

#if TPL.num_shadow_sources > 0
#include "utils/shadow.glsl"
uniform sampler2DShadow u_shadow_depth[TPL.num_shadow_sources];
in vec4 v_position_shadow[TPL.num_shadow_sources];
#endif

const float NUM_LIGHTS = TPL.num_lights;

in vec3 v_position;
in vec3 v_normal;
in vec3 v_tangent;
in vec3 v_bitangent;
in vec2 v_uv;

out vec4 out_color;


vec3 compute_irradiance(vec3 N, vec3 L, vec3 light_color) {
  float cosine_term = max(.0, dot(N, L));
  return cosine_term * max(vec3(0.0), light_color);
}


/**
 * Beckmann importance sampling code ripped from Blender.
 */
float bsdf_beckmann_aniso_G1(float alpha_x, float alpha_y, vec3 L_local) {
  float cos_phi = L_local.x;
  float sin_phi = L_local.y;
  float cos_n = L_local.z;
	float cos_n2 = cos_n * cos_n;
	float sin_phi2 = sin_phi * sin_phi;
	float cos_phi2 = cos_phi * cos_phi;
	float alpha_x2 = alpha_x * alpha_x;
	float alpha_y2 = alpha_y * alpha_y;

	float alphaO2 = (cos_phi2*alpha_x2 + sin_phi2*alpha_y2) / (cos_phi2 + sin_phi2);
	float invA = safe_sqrt(alphaO2 * (1 - cos_n2) / cos_n2);
	if(invA < 0.625f) {
		return 1.0f;
	}

	float a = 1.0f / invA;
	return ((2.181f*a + 3.535f)*a) / ((2.577f*a + 2.276f)*a + 1.0f);
}


vec3 microfacet_beckmann_sample_slopes(
	float cos_theta_i, float sin_theta_i, float randu, float randv) {

	/* special case (normal incidence) */
	if(cos_theta_i >= 0.99999f) {
		const float r = sqrt(-log(randu));
		const float phi = M_2PI * randv;
		return vec3(r * cos(phi), r * sin(phi), 1.0);
	}

	/* precomputations */
	const float tan_theta_i = sin_theta_i/cos_theta_i;
	const float inv_a = tan_theta_i;
	const float cot_theta_i = 1.0f/tan_theta_i;
	const float erf_a = fast_erff(cot_theta_i);
	const float exp_a2 = exp(-cot_theta_i*cot_theta_i);
	const float SQRT_PI_INV = 0.56418958354f;
	const float Lambda = 0.5f*(erf_a - 1.0f) + (0.5f*SQRT_PI_INV)*(exp_a2*inv_a);
	const float G1 = 1.0f/(1.0f + Lambda); /* masking */

	/* Based on paper from Wenzel Jakob
	 * An Improved Visible Normal Sampling Routine for the Beckmann Distribution
	 *
	 * http://www.mitsuba-renderer.org/~wenzel/files/visnormal.pdf
	 *
	 * Reformulation from OpenShadingLanguage which avoids using inverse
	 * trigonometric functions.
	 */

	/* Sample slope X.
	 *
	 * Compute a coarse approximation using the approximation:
	 *   exp(-ierf(x)^2) ~= 1 - x * x
	 *   solve y = 1 + b + K * (1 - b * b)
	 */
	float K = tan_theta_i * SQRT_PI_INV;
	float y_approx = randu * (1.0f + erf_a + K * (1 - erf_a * erf_a));
	float y_exact  = randu * (1.0f + erf_a + K * exp_a2);
	float b = K > 0 ? (0.5f - sqrt(K * (K - y_approx + 1.0f) + 0.25f)) / K : y_approx - 1.0f;

	/* Perform newton step to refine toward the true root. */
	float inv_erf = fast_ierff(b);
	float value  = 1.0f + b + K * exp(-inv_erf * inv_erf) - y_exact;
	/* Check if we are close enough already,
	 * this also avoids NaNs as we get close to the root.
	 */
  float slope_x, slope_y;
	if(abs(value) > 1e-6f) {
		b -= value / (1.0f - inv_erf * tan_theta_i); /* newton step 1. */
		inv_erf = fast_ierff(b);
		value  = 1.0f + b + K * exp(-inv_erf * inv_erf) - y_exact;
		b -= value / (1.0f - inv_erf * tan_theta_i); /* newton step 2. */
		/* Compute the slope from the refined value. */
		slope_x = fast_ierff(b);
	} else {
		/* We are close enough already. */
		slope_x = inv_erf;
	}
	slope_y = fast_ierff(2.0f*randv - 1.0f);

	return vec3(slope_x, slope_y, G1);
}


vec4 microfacet_sample_stretched(float alpha_x, float alpha_y, vec3 omega_i, float randu, float randv) {
  /* 1. stretch omega_i */
	vec3 omega_i_ = vec3(alpha_x * omega_i.x, alpha_y * omega_i.y, omega_i.z);
	omega_i_ = normalize(omega_i_);

	/* get polar coordinates of omega_i_ */
	float costheta_ = 1.0f;
	float sintheta_ = 0.0f;
	float cosphi_ = 1.0f;
	float sinphi_ = 0.0f;

	if(omega_i_.z < 0.99999f) {
		costheta_ = omega_i_.z;
		sintheta_ = safe_sqrt(1.0f - costheta_*costheta_);

		float invlen = 1.0f/sintheta_;
		cosphi_ = omega_i_.x * invlen;
		sinphi_ = omega_i_.y * invlen;
	}

	/* 2. sample P22_{omega_i}(x_slope, y_slope, 1, 1) */
  vec3 slope = microfacet_beckmann_sample_slopes(costheta_, sintheta_, randu, randv);
  float slope_x = slope.x;
  float slope_y = slope.y;
  float G1o = slope.z;

	/* 3. rotate */
	float tmp = cosphi_*slope_x - sinphi_*slope_y;
	slope_y = sinphi_*slope_x + cosphi_*slope_y;
	slope_x = tmp;

	/* 4. unstretch */
	slope_x = alpha_x * slope_x;
	slope_y = alpha_y * slope_y;

	/* 5. compute normal */
	return vec4(normalize(vec3(-slope_x, -slope_y, 1.0f)), G1o);
}

float get_pdf_value(float alpha_x, float alpha_y, vec3 H) {
  float slope_x = H.x/(H.z*alpha_x);
  float slope_y = H.y/(H.z*alpha_y);
  return exp(-pow(slope_x, 2) - pow(slope_y, 2)) / (alpha_x * alpha_y * M_PI);
}


void main() {
  vec3 V = normalize(u_cam_pos - v_position);

  vec3 rho_d = texture(u_diff_map, v_uv).rgb;
  vec3 rho_s = texture(u_spec_map, v_uv).rgb / (M_PI * 4.0);
  float roughness = texture(u_rough_map, v_uv).r;
  float aniso = texture(u_aniso_map, v_uv).r;
  float alpha_x, alpha_y;
  if(aniso < 0.0f) {
    alpha_x = roughness/(1.0f + aniso);
    alpha_y = roughness*(1.0f + aniso);
  }
  else {
    alpha_x = roughness*(1.0f - aniso);
    alpha_y = roughness/(1.0f - aniso);
  }

  mat3 TBN = mat3(v_tangent, v_bitangent, v_normal);
  vec3 N = normalize(TBN * texture(u_normal_map, v_uv).rgb);

  float shadowness = 0.0;
	#if TPL.num_shadow_sources > 0
	for (int i = 0; i < TPL.num_shadow_sources; i++) {
    shadowness += compute_shadow(v_position, v_position_shadow[i], u_shadow_depth[i]);
	}
  shadowness /= TPL.num_shadow_sources * 2.0;
	#endif

  vec3 total_radiance = vec3(0.0);

  #if TPL.use_radiance_map
  total_radiance += rho_d * texture(u_irradiance_map, N).rgb;

  vec3 specular = vec3(0);
  uint N_SAMPLES = 200u;

  float NdotV = dot(N, V);
  vec3 Z = N;
  vec3 up_vec = abs(N.z) < 0.999 ? vec3(0,0,1) : vec3(1,0,0);
  vec3 X = normalize(cross(up_vec, N));
  vec3 Y = cross(N, X);
  vec3 local_V = world_to_local(V, N);

  for (uint i = 0u; i < N_SAMPLES; i++) {
    vec2 xi = hammersley(i, N_SAMPLES); // Use psuedo-random point set.
    vec4 sample_result = microfacet_sample_stretched(alpha_x, alpha_y, local_V, xi.x, xi.y);
    vec3 H_local = sample_result.xyz;
    float G1o = sample_result.w;
    vec3 H_world = X*H_local.x + Y * H_local.y + Z * H_local.z;
    vec3 L = reflect(-V, H_world);
    vec3 L_local = world_to_local(L, N);

    float slope_x = -H_local.x/(H_local.z*alpha_x);
    float slope_y = -H_local.y/(H_local.z*alpha_y);

    float cosThetaM = H_local.z;
    float cosThetaM2 = cosThetaM * cosThetaM;
    float cosThetaM4 = cosThetaM2 * cosThetaM2;

    float D = exp(-slope_x*slope_x - slope_y*slope_y) / (M_PI * alpha_x * alpha_y * cosThetaM4);

    float G1i = bsdf_beckmann_aniso_G1(alpha_x, alpha_y, L_local);
    float G = G1i * G1o;

    float spec_common = D * 0.25f / dot(N, V);
    float spec_out = G * spec_common;
    float pdf = G1o * spec_common;

		float lod = compute_lod(pdf, N_SAMPLES, u_cubemap_size.x, u_cubemap_size.y);
    vec3 light_color;
    vec2 dp_uv = dualp_world_to_tex(L, 1.2);
    if (L.y > 0) {
      light_color = textureLod(u_radiance_upper, dp_uv, lod).rgb;
    } else {
      light_color = textureLod(u_radiance_lower, dp_uv, lod).rgb;
    }

//    float F = fresnel_schlick(F0, V, H_world) / F0;
    float VdotH = abs(dot(V, H_world));
    float NdotH = abs(dot(N, H_world));
    specular += rho_s * compute_irradiance(N, L, light_color);
  }
  total_radiance += specular / float(N_SAMPLES);
  total_radiance *= u_radiance_scale;
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
        total_radiance += aittala_spec(N, V, L, rho_s, S, u_alpha) * irradiance;
      }
    }
    total_radiance += rho_d * irradiance;
  }
  #endif

	total_radiance *= (1.0 - shadowness);
  out_color = vec4(max(vec3(.0), total_radiance), 1.0);    // rough gamma
}
