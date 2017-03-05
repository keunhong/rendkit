#version 450 core
uniform mat4 u_view;
uniform mat4 u_model;
uniform mat4 u_projection;

in vec3 a_position;
in vec2 a_uv;

#if TPL.use_normals
in vec3 a_normal;
out vec3 v_normal;
#endif

#if TPL.use_tangents
in vec3 a_tangent;
in vec3 a_bitangent;
out vec3 v_tangent;
out vec3 v_bitangent;
#endif

out vec3 v_position;
out vec2 v_uv;

void main() {
    gl_Position = u_projection * u_view * u_model * vec4(a_position, 1.0);

    v_position = a_position.xyz;

    #if TPL.use_normals
    v_normal = a_normal;
    #endif

    #if TPL.use_tangents
    v_tangent = a_tangent;
    v_bitangent = a_bitangent;
    #endif

    v_uv  = a_uv;
}
