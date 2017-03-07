

float compute_shadow(vec4 shadow_pos, sampler2D depth) {
  float shadow;
  vec3 shadow_proj = shadow_pos.xyz / shadow_pos.w;
  shadow_proj = shadow_proj * 0.5 + 0.5;
  if (shadow_proj.z > 1.0) {
    return 0.0;
  }

  float current_depth = shadow_proj.z;
  float bias = 0.007;
  vec2 texel_size = 1.0 / textureSize(depth, 0);

  float pcf_depth = texture(depth, shadow_proj.xy).r;
//  return current_depth - bias > pcf_depth ? 1.0 : 0.0;
  for(int x = -1; x <= 1; ++x) {
      for(int y = -1; y <= 1; ++y) {
          float pcf_depth = texture(depth, shadow_proj.xy + vec2(x, y) * texel_size).r;
          shadow += current_depth - bias > pcf_depth ? 1.0 : 0.0;
      }
  }
  shadow /= 9.0;

  return shadow;
}

