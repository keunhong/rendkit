
highp float rand(vec2 co) {
    highp float a = 12.9898;
    highp float b = 78.233;
    highp float c = 43758.5453;
    highp float dt= dot(co.xy ,vec2(a,b));
    highp float sn= mod(dt,3.14);
    return fract(sin(sn) * c);
}


vec3 angle_to_vec(float phi, float theta) {
  vec3 H;
  float sin_theta = sin(theta);
  H.x = sin(theta) * cos(phi);
  H.y = sin(theta) * sin(phi);
  H.z = cos(theta);
  return H;
}


// From https://github.com/thefranke/dirtchamber/blob/master/shader/importance.hlsl.
vec3 sample_to_world(vec3 H, vec3 N) {
  vec3 up_vec = abs(N.z) < 0.999 ? vec3(0,0,1) : vec3(1,0,0);
  vec3 tangent_x = normalize(cross(up_vec, N));
  vec3 tangent_y = cross(N, tangent_x);
  return normalize(tangent_x * H.x + tangent_y * H.y + N * H.z);
}


// From http://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html.
float radical_inverse_vdc(uint bits) {
  bits = (bits << 16u) | (bits >> 16u);
  bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
  bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
  bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
  bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
  return float(bits) * 2.3283064365386963e-10; // / 0x10000000:
}

vec2 hammersley(uint i, uint N) {
  return vec2(float(i) / float(N), radical_inverse_vdc(i));
}


float compute_lod(float pdf, uint n_samples, float width, float height) {
    float lod = (0.5 * log2((width * height)/float(n_samples))) - 0.5 * log2(pdf);
    return max(lod, 0);
}
