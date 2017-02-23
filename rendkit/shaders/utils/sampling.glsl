
highp float rand(vec2 co) {
    highp float a = 12.9898;
    highp float b = 78.233;
    highp float c = 43758.5453;
    highp float dt= dot(co.xy ,vec2(a,b));
    highp float sn= mod(dt,3.14);
    return fract(sin(sn) * c);
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

