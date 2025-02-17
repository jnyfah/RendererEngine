#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable

struct MaterialData
{
    vec4     Ambient;
    vec4     Emissive;
    vec4     Albedo;
    vec4     Specular;
    vec4     Roughness;
    vec4     Factors; // {x : transparency, y : Metallic, z : AlphaTest, w : _padding}

    uint64_t EmissiveMap;
    uint64_t AlbedoMap;
    uint64_t SpecularMap;
    uint64_t NormalMap;
    uint64_t OpacityMap;
    uint64_t _padding;
};

struct DirectionalLight
{
    vec4 Direction;
    vec4 Ambient;
    vec4 Diffuse;
    vec4 Specular;
};

struct PointLight
{
    vec4  Position;
    vec4  Ambient;
    vec4  Diffuse;
    vec4  Specular;

    float Constant;
    float Linear;
    float Quadratic;
    float _padding;
};

struct SpotLight
{
    vec4  Position;
    vec4  Direction;
    vec4  Ambient;
    vec4  Diffuse;
    vec4  Specular;

    float CutOff;
    float Constant;
    float Linear;
    float Quadratic;
};

#define INVALID_MAP_HANDLE 0xFFFFFFFFu

layout(std140, set = 0, binding = 5) readonly buffer MatSB
{
    MaterialData Data[];
}
MaterialDataBuffer;

layout(set = 0, binding = 9) uniform sampler2D TextureArray[];

MaterialData FetchMaterial(uint dataIndex)
{
    return MaterialDataBuffer.Data[dataIndex];
}

// http://www.thetenthplanet.de/archives/1180
// modified to fix handedness of the resulting cotangent frame
mat3 cotangentFrame(vec3 N, vec3 p, vec2 uv)
{
    // get edge vectors of the pixel triangle
    vec3  dp1     = dFdx(p);
    vec3  dp2     = dFdy(p);
    vec2  duv1    = dFdx(uv);
    vec2  duv2    = dFdy(uv);

    // solve the linear system
    vec3  dp2perp = cross(dp2, N);
    vec3  dp1perp = cross(N, dp1);
    vec3  T       = dp2perp * duv1.x + dp1perp * duv2.x;
    vec3  B       = dp2perp * duv1.y + dp1perp * duv2.y;

    // construct a scale-invariant frame
    float invmax  = inversesqrt(max(dot(T, T), dot(B, B)));

    // calculate handedness of the resulting cotangent frame
    float w       = (dot(cross(N, T), B) < 0.0) ? -1.0 : 1.0;

    // adjust tangent if needed
    T             = T * w;

    return mat3(T * invmax, B * invmax, N);
}

vec3 perturbNormal(vec3 n, vec3 v, vec3 normalSample, vec2 uv)
{
    vec3 map = normalize(2.0 * normalSample - vec3(1.0));
    mat3 TBN = cotangentFrame(n, v, uv);
    return normalize(TBN * map);
}

void runAlphaTest(float alpha, float alphaThreshold)
{
    if (alphaThreshold > 0.0)
    {
        // http://alex-charlton.com/posts/Dithering_on_the_GPU/
        // https://forums.khronos.org/showthread.php/5091-screen-door-transparency
        mat4 thresholdMatrix = mat4(1.0 / 17.0, 9.0 / 17.0, 3.0 / 17.0, 11.0 / 17.0, 13.0 / 17.0, 5.0 / 17.0, 15.0 / 17.0, 7.0 / 17.0, 4.0 / 17.0, 12.0 / 17.0, 2.0 / 17.0, 10.0 / 17.0, 16.0 / 17.0, 8.0 / 17.0, 14.0 / 17.0, 6.0 / 17.0);

        alpha                = clamp(alpha - 0.5 * thresholdMatrix[int(mod(gl_FragCoord.x, 4.0))][int(mod(gl_FragCoord.y, 4.0))], 0.0, 1.0);

        if (alpha < alphaThreshold)
            discard;
    }
}
