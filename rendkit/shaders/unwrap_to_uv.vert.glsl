#version 120
uniform mat4 u_view;
uniform mat4 u_model;
uniform mat4 u_projection;
uniform float u_near;
uniform float u_far;
attribute vec3 a_position;
attribute vec2 a_uv;
attribute vec3 a_normal;
attribute vec3 a_tangent;
attribute vec3 a_bitangent;
varying vec3 v_position;
varying vec3 v_normal;
varying vec3 v_tangent;
varying vec3 v_bitangent;
varying vec2 v_uv;
varying vec3 v_pos_clip_space;
varying float v_depth;


void main() {
    vec4 pos_clip_space = u_projection * u_view * u_model * vec4(a_position, 1.0);
    vec2 uv_ndc = a_uv * 2 - 1.0;
    gl_Position = vec4(uv_ndc, 0.0, 1.0);

    vec4 point_3d = u_view * u_model * vec4(a_position,1.0);
    v_depth = (point_3d.z - u_far) / (u_near - u_far);
    v_position = a_position.xyz;
    v_normal = a_normal;
    v_tangent = a_tangent;
    v_bitangent = a_bitangent;
    v_uv  = a_uv;
    v_pos_clip_space = pos_clip_space.xyz / pos_clip_space.w;
}
