#version 120

uniform sampler2D u_rendtex;
uniform vec2 u_texture_shape;
uniform vec4 u_aa_kernel;
varying vec2 v_texcoord;

void main() {
    vec2 pos = v_texcoord.xy;
    vec3 color = vec3(0.0);

    float dx = 1.0 / u_texture_shape.y;
    float dy = 1.0 / u_texture_shape.x;

    // Convolve
    int sze = 3;
    for (int y=-sze; y<sze+1; y++)
    {
        for (int x=-sze; x<sze+1; x++)
        {
            float k = u_aa_kernel[int(abs(float(x)))]
            			* u_aa_kernel[int(abs(float(y)))];
            vec2 dpos = vec2(float(x)*dx, float(y)*dy);
            color += texture2D(u_rendtex, pos + dpos).rgb * k;
        }
    }

    // Determine final color
    gl_FragColor = vec4(color, 1.0);
}
