#version 430

// Constants
#define BLACK (vec3(0.0))
#define WHITE (vec3(1.0))
#define RED (vec3(1.0, 0.0, 0.0))
#define GREEN (vec3(0.0, 1.0, 0.0))
#define BLUE (vec3(0.0, 0.0, 1.0))

#define REGION_DIM 32
#define REGION_DIM_INV (1.0/REGION_DIM)
#define TILES_PER_REGION ((REGION_DIM)*(REGION_DIM))
#define TILE_SIZE 32
#define REGION_BYTES ((TILES_PER_REGION)*(TILE_SIZE))

// Config (unit of length is tile, unless specified)
#define GRID_THICKNESS 0.3
#define GRID_CROSS_LENGTH 4
#define UNKNOWN_REGION_STRIP_SPEED 10.0
#define TILE_HIGHLIGHT_RADIUS 0.5
#define OO_BOUND_FADE_ALPHA 0.1
#define OO_BOUND_FADE_DISTANCE (REGION_DIM * 10)

// Debug Config
#define DEBUG_COLOR_ALPHA 0.12345
// #define DEBUG_WORLD_COORD_VALIDITY
// #define DEBUG_WORLD_REGION_COORD
// #define DEBUG_WORLD_REGION_ID
// #define DEBUG_WORLD_REGION_TILE_COORD

// Macros
#define SET_DEBUG_COLOR(v3) (output_color = vec4((v3), DEBUG_COLOR_ALPHA))
#define IS_DEBUG_COLOR_SET() (abs(output_color.a - DEBUG_COLOR_ALPHA) < 0.001)
#define GET_DEBUG_COLOR() (vec4(output_color.xyz, 1.0))

#define FDIV(a, b) (float(a) / float(b))

#define RGB16_R(u16) (FDIV(((u16) >> 11) & 0x1FU, 31))
#define RGB16_G(u16) (FDIV(((u16) >> 5) & 0x3FU, 63))
#define RGB16_B(u16) (FDIV((u16) & 0x1FU, 31))
#define RGB16(u16) (vec3(RGB16_R(u16), RGB16_G(u16), RGB16_B(u16)))

#define HUE8(u8) (FDIV(u8, 360))

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

struct Rect {
    vec2 position;
    vec2 size;
};
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

uniform vec2 iResolution;
uniform float iTime;
uniform mat3 iFragProjection;
uniform struct View {
    ivec2 worldRSize;
    ivec2 worldTSize;
    Rect clipRect;
} iView;
uniform struct Config {
    bool showGrid;
} iConfig;
uniform ivec2 iTileSelected;

layout(std430, binding = 0) buffer World {
    Region worldRegions[];
};

out vec4 output_color;

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

// validity: Non-negative -> invalid, otherwise valid
void locateTile(in vec2 norm01, out float validity, out vec2 world_coord, out ivec2 region_coord, out ivec2 tile_coord){
    vec2 unwrapped_world_coord = iView.clipRect.position + iView.clipRect.size * norm01;
    vec2 min_dist_to_boundary = min(unwrapped_world_coord, iView.worldTSize - unwrapped_world_coord);
    validity = min(
        min_dist_to_boundary.x,
        min_dist_to_boundary.y);
    world_coord = mod(unwrapped_world_coord, iView.worldTSize);   // wrap around regions that are out of boundary
    region_coord = ivec2(world_coord / REGION_DIM);
    tile_coord = ivec2(mod(world_coord, REGION_DIM));
}

Tile getWorldTile(in ivec2 region_coord, in ivec2 tile_coord) {
    int linear_region_id = region_coord.y * iView.worldRSize.x + region_coord.x;
    int linear_tile_id = tile_coord.y * REGION_DIM + tile_coord.x;

    #if defined(DEBUG_WORLD_REGION_COORD)
    SET_DEBUG_COLOR(vec3(
        FDIV(region_coord.x, iView.worldRSize.x),
        FDIV(region_coord.y, iView.worldRSize.y),
        0.0
    ));
    #elif defined(DEBUG_WORLD_REGION_ID)
    SET_DEBUG_COLOR(vec3(
        FDIV(linear_region_id, iView.worldRSize.x * iView.worldRSize.y))
    );
    #elif defined(DEBUG_WORLD_REGION_TILE_COORD)
    SET_DEBUG_COLOR(vec3(
        tile_coord / float(REGION_DIM),
        0.0
    ));
    #endif

    return worldRegions[linear_region_id].tiles[linear_tile_id];
}

vec3 unknownRegionColor(in vec2 norm01){
    float time_offset = iTime * UNKNOWN_REGION_STRIP_SPEED;
    float pixel = norm01.x * iResolution.x + norm01.y * iResolution.y;
    return vec3(0.5) * (sin(pixel + time_offset) + 1) * 0.5;
}

vec3 tileColor(in Tile tile, in vec2 norm01) {
    float fg_hue_shift_01 = Tile_fgHue(tile);
    float bg_hue_shift_01 = Tile_bgHue(tile);
    uint coll = Tile_collision(tile);
    bool valid_tile = Tile_valid(tile);

    bool known_tile = coll != COLL_UNKNOWN;
    float collision = float(coll != COLL_EMPTY);
    vec3 unknown_color = unknownRegionColor(norm01);
    vec3 color;
    if (valid_tile){
        color = mix(
            hsv2rgb(vec3(bg_hue_shift_01, 1.0, 0.2)),
            hsv2rgb(vec3(fg_hue_shift_01, 1.0, collision)),
            collision
        );
    } else {
        color = BLACK;
    }

    return mix(unknown_color, color, max(0.5, round(float(known_tile))));
}

vec3 boxShadow(in vec3 color, in vec2 world_coord, in vec2 position, in vec2 size, in float radius) {
    vec2 half_size = size / 2;
    vec2 box_center = position + half_size;
    vec2 dist_to_box = abs(world_coord - box_center) - half_size;
    float max_dist_to_box = max(dist_to_box.x, dist_to_box.y);
    float intensity = 1.0 - smoothstep(0, radius, max_dist_to_box);
    intensity *= step(0.0, max_dist_to_box); // outer shadow only
    return intensity * color;
}

vec3 regionGridColor(in vec2 world_coord, float zoom) {
    vec2 region_coord = world_coord / REGION_DIM;
    vec2 dist_to_grid = abs(round(region_coord) - region_coord) * REGION_DIM;
    float max_dist_to_grid = max(dist_to_grid.x, dist_to_grid.y);
    float min_dist_to_grid = min(dist_to_grid.x, dist_to_grid.y);
    float intensity = 1.0 - smoothstep(0.0, 1.0, min_dist_to_grid / GRID_THICKNESS);
    intensity *= step(max_dist_to_grid, GRID_CROSS_LENGTH); // limited to a circle from the corner
    return vec3(intensity * pow(min(zoom, 1.0), 5.0));
}

vec3 boundaryGridColor(in float world_coord_validity) {
    float intensity = 1.0 - smoothstep(0.0, 1.0, abs(world_coord_validity) / GRID_THICKNESS);
    return vec3(intensity);
}

vec4 mainImage(in vec2 frag_coord){

    vec3 n01aug = iFragProjection * vec3(frag_coord, 1.0);
    vec2 n01 = n01aug.xy / n01aug.z;    // Normalized coord in [0, 1]^2
    float zoom = FDIV(iResolution.x, iView.clipRect.size.x);
    vec3 frag_color;

    float world_coord_validity;
    vec2 world_coord;
    ivec2 region_coord;
    ivec2 tile_coord;
    locateTile(n01, world_coord_validity, world_coord, region_coord, tile_coord);

    #if defined(DEBUG_WORLD_COORD_VALIDITY)
        SET_DEBUG_COLOR(vec3(world_coord_validity / max(iView.worldTSize.x, iView.worldTSize.y)));
    #else
    if (iView.clipRect.size.x == 0) {
        frag_color = mix(BLACK, unknownRegionColor(n01), 0.5);
    } else {
        Tile tile = getWorldTile(region_coord, tile_coord);
        frag_color = tileColor(tile, n01);

        frag_color += boxShadow(WHITE, world_coord, iTileSelected, vec2(1.0), TILE_HIGHLIGHT_RADIUS);

        // Fade out when the tile is out of boundary
        frag_color = mix(frag_color, BLACK,
                clamp(-world_coord_validity / OO_BOUND_FADE_DISTANCE, 0, 1));

        if (iConfig.showGrid){
            frag_color += regionGridColor(world_coord, zoom);
            frag_color += boundaryGridColor(world_coord_validity);
        }
    }
    #endif
    return vec4(frag_color, 1.0);
}

void main()
{
    vec2 frag_coord = gl_FragCoord.xy;
    vec4 color = mainImage(frag_coord);

    if (IS_DEBUG_COLOR_SET()) {
        output_color = GET_DEBUG_COLOR();
    } else {
        output_color = color;
    }
}