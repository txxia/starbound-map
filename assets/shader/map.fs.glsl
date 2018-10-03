#version 430

// Config
#define GRID_INTENSITY 0.0001
#define INVALID_REGION_STRIP_SPEED 10.0
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
#define BLACK (vec3(0.0))

#define GRID_DIM 7
#define GRID_DIM_INV FDIV(1, GRID_DIM)
#define REGION_COUNT (GRID_DIM*GRID_DIM)

#define REGION_DIM 32
#define TILES_PER_REGION ((REGION_DIM)*(REGION_DIM))
#define TILE_SIZE 32
#define REGION_BYTES ((TILES_PER_REGION)*(TILE_SIZE))

// Tile data access
#define MAT(u) ((u) >> 16)
#define HUE(u) (HUE8(((u) >> 8) & 0xFFU))
#define VAR(u) ((u) & 0xFU)

#define COLL(u) (((u) >> 8) & 0xFFU)
#define COLL_UNKNOWN    0U
#define COLL_EMPTY      1U
#define COLL_PLATFORM   2U
#define COLL_DYNAMIC    3U
#define COLL_SOLID      5U

uniform vec2 iResolution;
uniform float iTime;
uniform mat3 iFragProjection;
uniform struct Config {
    bool showGrid;
} iConfig;

struct Tile {
    uint fg_mat2_hue_var;
    uint fg_mod2_hueShift_validity;
    uint bg_mat2_hue_var;
    uint bg_mod2_hueShift;
    uint liquidLevel;
    uint liquidPressure;
    uint liquid_collision_dungeon2;
    uint biome2_indestructible;
};
struct Region {
    Tile tiles[TILES_PER_REGION];
};
layout(std430, binding = 1) buffer World {
    Region gRegions[];
};

out vec4 outputColor;

///////////////////////////////////////////////////////////////////////////////

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

///////////////////////////////////////////////////////////////////////////////

bool Tile_valid(in Tile tile){ return (tile.fg_mod2_hueShift_validity & 0xF) != 0U; }
uint Tile_fgMat(in Tile tile){ return MAT(tile.fg_mat2_hue_var); }
uint Tile_bgMat(in Tile tile){ return MAT(tile.bg_mat2_hue_var); }
float Tile_fgHue(in Tile tile){ return HUE(tile.fg_mat2_hue_var); }
float Tile_bgHue(in Tile tile){ return HUE(tile.bg_mat2_hue_var); }
uint Tile_collision(in Tile tile){ return COLL(tile.liquid_collision_dungeon2); }

///////////////////////////////////////////////////////////////////////////////

// Get the linear tile id in a region, given its 2D coordinate.
int getTileId(in vec2 normalCoord01){
    // transform to [0..REGION_DIM]^2
    ivec2 t = ivec2(normalCoord01 * REGION_DIM);
    return t.y * REGION_DIM + t.x;
}

// Get region number in [0, REGION_COUNT) based on pixel coordinate in [0, 1]^2.
int getRegionId(in vec2 normalCoord01) {
    ivec2 r01 = ivec2(normalCoord01 * float(GRID_DIM));
    return r01.y * GRID_DIM + r01.x;
}

void getTile(in int regionId, in int tileId, inout Tile tile) {
    tile = gRegions[regionId].tiles[tileId];
}

vec3 unknownRegionColor(in vec2 normalCoord01){
    float timeOffset = iTime * INVALID_REGION_STRIP_SPEED;
    float pixel = normalCoord01.x * iResolution.x + normalCoord01.y * iResolution.y;
    return vec3(0.5) * (sin(pixel + timeOffset) + 1) * 0.5;
}

vec3 tileColor(in int regionId, in int tileId, in vec2 normalCoord01) {
    Tile tile;
    getTile(regionId, tileId, tile);
    float fgHueShift01 = Tile_fgHue(tile);
    float bgHueShift01 = Tile_bgHue(tile);
    uint coll = Tile_collision(tile);
    bool validTile = Tile_valid(tile);

    bool knownTile = coll != COLL_UNKNOWN;
    float collision = float(coll != COLL_EMPTY);
    vec3 unknownColor = unknownRegionColor(normalCoord01);
    vec3 color;
    if (validTile){
        color = mix(
            hsv2rgb(vec3(bgHueShift01, 1.0, 0.2)),
            hsv2rgb(vec3(fgHueShift01, 1.0, collision)),
            collision
        );
    } else {
        color = BLACK;
    }

    return mix(unknownColor, color, max(0.5, round(float(knownTile))));
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
    vec2 regionCoord01 = mod(r01, GRID_DIM_INV) * GRID_DIM;
    int tileId = getTileId(regionCoord01);
    pixel = tileColor(regionId, tileId, r01);

    #ifdef DEBUG_TILE_ID
    pixel = vec3(
        FDIV(mod(tileId, REGION_DIM), REGION_DIM),
        FDIV(FDIV(tileId, REGION_DIM), REGION_DIM),
        0.0
    );
    #endif
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