#ifndef SRC_SHADERS_AUTOTUNE_HLSL_
#define SRC_SHADERS_AUTOTUNE_HLSL_

/// AutoTune: in-shader scene statistics collection for HDR auto-suggestion.
///
/// Opt-in per game:
///   #define RENODX_AUTOTUNE 1
///   #include "../../shaders/autotune.hlsl"
/// then call inside the tonemap/output pixel shader (scene color, pre-tonemap):
///   RENODX_AUTOTUNE_ACCUMULATE(untonemapped, SV_Position);
///
/// The CPU-side counterpart (mods/autotune.hpp) binds the stats UAV,
/// reads it back with a 3-frame delay and derives slider suggestions.
///
/// Buffer layout (uints):
///   [0..63]  log2-luminance histogram (subsampled 1/16 pixels)
///   [64]     total samples
///   [65]     samples outside BT.709 gamut (any negative channel)
///   [66]     max luminance this frame (float bits, InterlockedMax works
///            because positive IEEE754 floats compare as uints)
///   [67]     reserved

#if RENODX_AUTOTUNE && (__SHADER_TARGET_MAJOR >= 6)

#ifndef RENODX_AUTOTUNE_UAV_SLOT
#define RENODX_AUTOTUNE_UAV_SLOT u0
#endif
#ifndef RENODX_AUTOTUNE_UAV_SPACE
#define RENODX_AUTOTUNE_UAV_SPACE space50
#endif

RWBuffer<uint> renodx_autotune_stats
    : register(RENODX_AUTOTUNE_UAV_SLOT, RENODX_AUTOTUNE_UAV_SPACE);

namespace renodx {
namespace autotune {

static const uint HISTOGRAM_BINS = 64u;
static const uint OFFSET_SAMPLE_COUNT = 64u;
static const uint OFFSET_WIDE_GAMUT_COUNT = 65u;
static const uint OFFSET_MAX_LUMINANCE = 66u;

// log2 luminance range covered by the histogram.
// 2^-14 (deep shadow) .. 2^6 (64x diffuse white ~ 6400 nits scene-referred)
static const float LOG2_LUM_MIN = -14.f;
static const float LOG2_LUM_MAX = 6.f;

void Accumulate(float3 color_bt709, float4 sv_position) {
  // Subsample 1/16 pixels (every 4th pixel in x and y) to keep
  // atomic contention negligible.
  uint2 pixel = uint2(sv_position.xy);
  if (((pixel.x | pixel.y) & 3u) != 0u) return;

  const float3 clamped = max(0, color_bt709);
  const float y = dot(clamped, float3(0.2126f, 0.7152f, 0.0722f));

  const float log_y = log2(max(y, 1e-10f));
  const float normalized = saturate((log_y - LOG2_LUM_MIN) / (LOG2_LUM_MAX - LOG2_LUM_MIN));
  const uint bin = min((uint)(normalized * HISTOGRAM_BINS), HISTOGRAM_BINS - 1u);

  uint ignore;
  InterlockedAdd(renodx_autotune_stats[bin], 1u, ignore);
  InterlockedAdd(renodx_autotune_stats[OFFSET_SAMPLE_COUNT], 1u, ignore);

  // Out-of-BT.709 detection: scene value with a negative channel can only
  // be represented in a wider gamut.
  const float min_channel = min(color_bt709.r, min(color_bt709.g, color_bt709.b));
  if (min_channel < -1e-4f) {
    InterlockedAdd(renodx_autotune_stats[OFFSET_WIDE_GAMUT_COUNT], 1u, ignore);
  }

  InterlockedMax(renodx_autotune_stats[OFFSET_MAX_LUMINANCE], asuint(y), ignore);
}

}  // namespace autotune
}  // namespace renodx

#define RENODX_AUTOTUNE_ACCUMULATE(color, sv_pos) \
  renodx::autotune::Accumulate((color), (sv_pos))

#else  // !RENODX_AUTOTUNE

#define RENODX_AUTOTUNE_ACCUMULATE(color, sv_pos)

#endif  // RENODX_AUTOTUNE

#endif  // SRC_SHADERS_AUTOTUNE_HLSL_
