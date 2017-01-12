#version 120
uniform mat4 u_view;
uniform mat4 u_model;
uniform mat4 u_projection;

attribute vec3 a_position;
attribute vec2 a_uv;

#if TPL.use_normals
attribute vec3 a_normal;
varying vec3 v_normal;
#endif

#if TPL.use_tangents
attribute vec3 a_tangent;
attribute vec3 a_bitangent;
varying vec3 v_tangent;
varying vec3 v_bitangent;
#endif

varying vec3 v_position;
varying vec2 v_uv;

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
