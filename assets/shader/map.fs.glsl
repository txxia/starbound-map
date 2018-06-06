#version 330

// Config
#define GRID_INTENSITY 0.0001
// #define DEBUG_TILE_ID
// #define DEBUG_REGION_ID

// Macros
#define FDIV(a, b) (float(a) / float(b))

#define RGB16_R(u16) (FDIV(((u16) >> 11) & 0x1FU, 31))
#define RGB16_G(u16) (FDIV(((u16) >> 5) & 0x3FU, 63))
#define RGB16_B(u16) (FDIV((u16) & 0x1FU, 31))
#define RGB16(u16) (vec3(RGB16_R(u16), RGB16_G(u16), RGB16_B(u16)))

#define HUE8(u8) (FDIV(u8, 360))

// Constants
#define GRID_DIM 7
#define GRID_DIM_INV FDIV(1, GRID_DIM)
#define REGION_COUNT (GRID_DIM*GRID_DIM)

#define REGION_DIM 32
#define REGION_TILES ((REGION_DIM)*(REGION_DIM))
#define TILE_SIZE 32
#define REGION_BYTES ((REGION_TILES)*(TILE_SIZE))
#define LAYERS_PER_REGION 2

// Tile Data
#define FG_MAT(d) ((d)[0].x >> 16)
#define FG_HUE(d) (HUE8(((d)[0].x >> 8) & 0xFFU))
#define FG_VAR(d) ((d)[0].x & 0xFU)

#define BG_MAT(d) ((d)[0].z >> 16)
#define BG_HUE(d) (HUE8(((d)[0].z >> 8) & 0xFFU))
#define BG_VAR(d) ((d)[0].z & 0xFFU)

#define COLL(d) (((d)[1].z >> 8) & 0xFFU)
#define COLL_EMPTY      1U
#define COLL_PLATFORM   2U
#define COLL_DYNAMIC    3U
#define COLL_SOLID      5U

uniform vec2 iResolution;
uniform float iTime;
uniform mat3 iFragProjection;
uniform bool iRegionValid[REGION_COUNT];
uniform usamplerBuffer iRegionLayer[REGION_COUNT * LAYERS_PER_REGION];
uniform struct Config {
    bool showGrid;
} iConfig;

out vec4 outputColor;

// http://lolengine.net/blog/2013/07/27/rgb-to-hsv-in-glsl
vec3 rgb2hsv(vec3 c)
{
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));

    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}
vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

// Get the linear tile id in a region, given its 2D coordinate.
int getTileId(in vec2 normalCoord01){
    // transform to [0..TILE_SIZE]^2
    ivec2 t = ivec2(normalCoord01 * TILE_SIZE) + 1;
    return (t.y - 1) * REGION_DIM + t.x - 1;
}

// Get region number in [0, REGION_COUNT) based on pixel coordinate in [0, 1]^2.
int getRegionId(in vec2 normalCoord01) {
    ivec2 r01 = ivec2(normalCoord01 * float(GRID_DIM));
    return r01.y * GRID_DIM + r01.x;
}

void getTile(in int regionId, in int id, inout uvec4 tileData[2]){
    int regionBase = regionId * 2;
    tileData[0] = texelFetch(iRegionLayer[regionBase], id);
    tileData[1] = texelFetch(iRegionLayer[regionBase+1], id);
}

vec3 invalidRegionColor(in vec2 normalCoord01){
    return vec3(0.0, 0.5, 0.7) * sqrt(sin(normalCoord01.y * iResolution.y));
}

vec3 tileColor(in int regionId, in int tileId) {
    uvec4 tileData[2];
    getTile(regionId, tileId, tileData);

    uint fgId = FG_MAT(tileData);
    float fgHueShift01 = FG_HUE(tileData);
    uint bgId = BG_MAT(tileData);
    float bgHueShift01 = BG_HUE(tileData);
    float coll = float(COLL(tileData) != COLL_EMPTY);

    return mix(
        hsv2rgb(vec3(bgHueShift01, 1.0, 0.2)),
        hsv2rgb(vec3(fgHueShift01, 1.0, coll)),
        coll
    );
}

vec3 gridColor(in vec2 normalCoord01) {
    vec2 grid_fract = fract(normalCoord01 * GRID_DIM);
    vec2 grid_proximity = 1.0 / abs(round(grid_fract) - grid_fract);
    vec3 grid = vec3(GRID_INTENSITY * (grid_proximity.x * grid_proximity.y));
    return grid;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord){
    // r in [-1, +1]^2
    vec3 r01aug = iFragProjection * vec3(fragCoord, 1.0);
    vec2 r01 = r01aug.xy / r01aug.z;
    vec2 r = r01.xy * 2.0 - 1.0;
    vec3 pixel;

    int regionId = getRegionId(r01);
    if (iRegionValid[regionId]) {
        vec2 regionCoord01 = mod(r01, GRID_DIM_INV) * GRID_DIM;
        int tileId = getTileId(regionCoord01);
        pixel = tileColor(regionId, tileId);

        #ifdef DEBUG_TILE_ID
        pixel = vec3(
            FDIV(mod(tileId, REGION_DIM), REGION_DIM),
            FDIV(FDIV(tileId, REGION_DIM), REGION_DIM),
            0.0
        );
        #endif
    } else {
        pixel = invalidRegionColor(r01);
    }
    if (iConfig.showGrid){
        pixel += gridColor(r01);
    }

    #ifdef DEBUG_REGION_ID
    pixel = vec3(
        FDIV(mod(regionId, GRID_DIM), GRID_DIM),
        FDIV(FDIV(regionId, GRID_DIM), GRID_DIM),
        0.0
    );
    #endif

    fragColor = vec4(pixel, 1.0);
}

void main()
{
    vec2 fragCoord = gl_FragCoord.xy;
    mainImage(outputColor, fragCoord);
}