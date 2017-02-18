#version 120

#define M_PI 3.1415926535897932384626433832795


uniform samplerCube u_cubemap;
uniform int u_cube_face;
varying vec2 v_uv;


// Adapted from http://www.codinglabs.net/article_physically_based_rendering.aspx
vec4 samp(vec2 pos, int cube_face) {
  vec3 normal = normalize(vec3(pos.xy, 1) );
  switch(cube_face) {
    case 0: normal = normalize(vec3(1, pos.y, -pos.x)); break;
    case 1: normal = normalize(vec3(-1, pos.y, pos.x)); break;
    case 2: normal = normalize(vec3(pos.x, 1, -pos.y)); break;
    case 3: normal = normalize(vec3(pos.x, -1, pos.y)); break;
    case 4: normal = normalize(vec3(pos.xy, 1)); break;
    case 5: normal = normalize(vec3(-pos.x, pos.y, -1)); break;
  }

  vec3 up = vec3(0, 1, 0);
  vec3 right = normalize(cross(up, normal));
  up = cross(normal, right);

  vec3 samp_color = vec3(0, 0, 0);
  float num_samples = 0;
  for (float phi = 0; phi < 2 * M_PI; phi += 0.025) {
    for (float theta = 0; theta < M_PI / 2; theta += 0.1) {
      vec3 temp = cos(phi) * right + sin(phi) * up;
      vec3 samp_vec = cos(theta) * normal + sin(theta) * temp;
      samp_color += textureCube(u_cubemap, samp_vec).rgb * cos(theta) * sin(theta);
      num_samples++;
    }
  }

  return vec4(M_PI * samp_color / num_samples, 1.0);
}


void main() {
  gl_FragColor = vec4(samp(v_uv, u_cube_face));
}
