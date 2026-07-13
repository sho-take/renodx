/*
 * Copyright (C) 2026 RenoDX AutoTune contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <functional>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <deps/imgui/imgui.h>
#include <include/reshade.hpp>

#include "../utils/data.hpp"
#include "../utils/settings.hpp"
#include "./shader.hpp"

// AutoTune: reads back per-frame scene statistics accumulated by
// shaders/autotune.hlsl and derives suggested slider values ("Suggest mode").
//
// v1 scope:
//   - D3D12 only (other APIs silently disable themselves)
//   - Suggest-only: values are shown in the overlay; the user applies them
//     with a button. Nothing is written automatically.
//   - Suggestions are never persisted unless the user applies them
//     (applying goes through utils::settings::UpdateSettings like any
//     manual slider change).
//
// Usage (game addon):
//   renodx::mods::autotune::SetupCustomShaders(custom_shaders);  // before shader::Use
//   for (auto* s : renodx::mods::autotune::NewSettings({
//            {"ColorGradeHighlights", [](auto& r) { return r.highlights; }},
//            {"ColorGradeShadows",    [](auto& r) { return r.shadows; }},
//            {"ColorGradeContrast",   [](auto& r) { return r.contrast; }},
//            {"ColorGradeBlowout",    [](auto& r) { return r.dechroma; }},
//        })) settings.push_back(s);
//   renodx::mods::autotune::Use(fdw_reason);  // in DllMain

namespace renodx::mods::autotune {

static uint32_t stats_uav_slot = 0u;    // register(u0, ...)
static uint32_t stats_uav_space = 50u;  // register(..., space50)

constexpr uint32_t HISTOGRAM_BINS = 64u;
constexpr uint32_t OFFSET_SAMPLE_COUNT = 64u;
constexpr uint32_t OFFSET_WIDE_GAMUT_COUNT = 65u;
constexpr uint32_t OFFSET_MAX_LUMINANCE = 66u;
constexpr uint32_t TOTAL_DWORDS = 68u;
constexpr uint64_t BUFFER_SIZE = TOTAL_DWORDS * sizeof(uint32_t);
constexpr int READBACK_COUNT = 3;  // frames of latency before mapping

// Must match autotune.hlsl
constexpr float LOG2_LUM_MIN = -14.f;
constexpr float LOG2_LUM_MAX = 6.f;

struct SceneStats {
  float p50 = 0.f;
  float p95 = 0.f;
  float p999 = 0.f;
  float max_luminance = 0.f;
  float wide_gamut_ratio = 0.f;
  uint32_t samples = 0u;
  bool valid = false;
};

struct Recommendation {
  float highlights = 50.f;
  float shadows = 50.f;
  float contrast = 50.f;
  float dechroma = 50.f;
  const char* scene_label = "n/a";
};

namespace internal {

struct __declspec(uuid("5d1c6a44-9f3e-4b8a-b1c7-00a070d00001")) DeviceData {
  reshade::api::resource stats_buffer = {0};
  reshade::api::resource_view stats_uav = {0};
  std::array<reshade::api::resource, READBACK_COUNT> readback = {};
  uint64_t frame = 0;
  bool ready = false;
};

static std::mutex stats_mutex;
static SceneStats raw_stats;       // last frame, unsmoothed
static SceneStats smooth_stats;    // asymmetric EMA
static Recommendation recommendation;
static float setting_enabled = 1.f;

// Asymmetric smoothing: adapt fast when the scene gets darker (attack),
// slowly when it gets brighter (release) - mirrors human adaptation and
// avoids pumping. Suggest-mode only displays values, so smoothing here is
// purely for readable, stable numbers.
inline float Ema(float previous, float current, float attack, float release) {
  const float alpha = (current < previous) ? attack : release;
  return previous + ((current - previous) * alpha);
}

inline float BinCenterLuminance(uint32_t bin) {
  const float t = (static_cast<float>(bin) + 0.5f) / static_cast<float>(HISTOGRAM_BINS);
  return exp2f(LOG2_LUM_MIN + (t * (LOG2_LUM_MAX - LOG2_LUM_MIN)));
}

inline float Percentile(const uint32_t* histogram, uint32_t total, float fraction) {
  if (total == 0u) return 0.f;
  const auto target = static_cast<uint64_t>(static_cast<double>(total) * fraction);
  uint64_t cumulative = 0;
  for (uint32_t bin = 0; bin < HISTOGRAM_BINS; ++bin) {
    cumulative += histogram[bin];
    if (cumulative >= target) return BinCenterLuminance(bin);
  }
  return BinCenterLuminance(HISTOGRAM_BINS - 1u);
}

inline void Analyze(const uint32_t* dwords) {
  SceneStats stats;
  stats.samples = dwords[OFFSET_SAMPLE_COUNT];
  if (stats.samples == 0u) return;

  stats.p50 = Percentile(dwords, stats.samples, 0.50f);
  stats.p95 = Percentile(dwords, stats.samples, 0.95f);
  stats.p999 = Percentile(dwords, stats.samples, 0.999f);
  float max_lum;
  static_assert(sizeof(max_lum) == sizeof(uint32_t));
  std::memcpy(&max_lum, &dwords[OFFSET_MAX_LUMINANCE], sizeof(max_lum));
  stats.max_luminance = max_lum;
  stats.wide_gamut_ratio = static_cast<float>(dwords[OFFSET_WIDE_GAMUT_COUNT])
                           / static_cast<float>(stats.samples);
  stats.valid = true;

  const std::unique_lock lock(stats_mutex);
  raw_stats = stats;

  if (!smooth_stats.valid) {
    smooth_stats = stats;
  } else {
    smooth_stats.p50 = Ema(smooth_stats.p50, stats.p50, 0.10f, 0.02f);
    smooth_stats.p95 = Ema(smooth_stats.p95, stats.p95, 0.10f, 0.02f);
    smooth_stats.p999 = Ema(smooth_stats.p999, stats.p999, 0.10f, 0.02f);
    smooth_stats.max_luminance = Ema(smooth_stats.max_luminance, stats.max_luminance, 0.10f, 0.02f);
    smooth_stats.wide_gamut_ratio = Ema(smooth_stats.wide_gamut_ratio, stats.wide_gamut_ratio, 0.05f, 0.05f);
    smooth_stats.samples = stats.samples;
  }

  // --- Scene classification -> recommendation ---------------------------
  // Stage 1: classify. Stage 2: map class to slider deltas. Keeping these
  // separate makes the heuristics tunable without touching the readback.
  // All values are v1 placeholders to be tuned on real content.
  Recommendation rec;
  const float eps = 1e-6f;
  const float key = smooth_stats.p50;                                       // scene key (median luminance)
  const float dynamic_range = log2f((smooth_stats.p999 + eps) / (smooth_stats.p50 + eps));

  const bool low_key = key < 0.02f;
  const bool high_key = key > 0.30f;
  const bool flat = dynamic_range < 3.0f;
  const bool extreme_dr = dynamic_range > 8.0f;

  if (low_key && extreme_dr) {
    rec.scene_label = "dark, high-DR (night + lights)";
    rec.highlights = 55.f;  // keep highlights, they carry the scene
    rec.shadows = 56.f;     // lift floor slightly to avoid crush
    rec.contrast = 48.f;    // engine contrast already extreme
  } else if (low_key) {
    rec.scene_label = "dark (cave/interior)";
    rec.highlights = 50.f;
    rec.shadows = 58.f;
    rec.contrast = 50.f;
  } else if (high_key && flat) {
    rec.scene_label = "bright, flat (overcast/menu)";
    rec.highlights = 58.f;  // expand: headroom is unused
    rec.shadows = 48.f;
    rec.contrast = 54.f;
  } else if (extreme_dr) {
    rec.scene_label = "high dynamic range (sunlit)";
    rec.highlights = 52.f;
    rec.shadows = 52.f;
    rec.contrast = 50.f;
  } else {
    rec.scene_label = "balanced";
    rec.highlights = 52.f;
    rec.shadows = 50.f;
    rec.contrast = 51.f;
  }

  // Gamut: lots of out-of-709 color -> protect (more dechroma/blowout
  // restraint); almost none -> safe to keep vivid.
  if (smooth_stats.wide_gamut_ratio > 0.05f) {
    rec.dechroma = 56.f;
  } else if (smooth_stats.wide_gamut_ratio < 0.005f) {
    rec.dechroma = 48.f;
  } else {
    rec.dechroma = 50.f;
  }

  recommendation = rec;
}

inline void OnInitDevice(reshade::api::device* device) {
  if (device->get_api() != reshade::api::device_api::d3d12) {
    std::stringstream s;
    s << "mods::autotune disabled (unsupported API "
      << static_cast<int>(device->get_api()) << ", v1 is D3D12-only)";
    reshade::log::message(reshade::log::level::warning, s.str().c_str());
    return;
  }

  auto* data = renodx::utils::data::Create<DeviceData>(device);

  reshade::api::resource_desc buffer_desc(
      BUFFER_SIZE,
      reshade::api::memory_heap::gpu_only,
      reshade::api::resource_usage::unordered_access
          | reshade::api::resource_usage::copy_source);

  if (!device->create_resource(
          buffer_desc, nullptr,
          reshade::api::resource_usage::unordered_access,
          &data->stats_buffer)) {
    reshade::log::message(reshade::log::level::error, "mods::autotune failed to create stats buffer");
    return;
  }

  if (!device->create_resource_view(
          data->stats_buffer,
          reshade::api::resource_usage::unordered_access,
          // Typed buffer view: ReShade's D3D12 backend does not set
          // StructureByteStride, so structured views are unusable here.
          // buffer.offset/size are in ELEMENTS for typed views.
          reshade::api::resource_view_desc(
              reshade::api::resource_view_type::buffer,
              reshade::api::format::r32_uint,
              0, TOTAL_DWORDS),
          &data->stats_uav)) {
    reshade::log::message(reshade::log::level::error, "mods::autotune failed to create stats UAV");
    return;
  }

  const reshade::api::resource_desc readback_desc(
      BUFFER_SIZE,
      reshade::api::memory_heap::gpu_to_cpu,
      reshade::api::resource_usage::copy_dest);
  for (auto& readback : data->readback) {
    if (!device->create_resource(
            readback_desc, nullptr,
            reshade::api::resource_usage::copy_dest,
            &readback)) {
      reshade::log::message(reshade::log::level::error, "mods::autotune failed to create readback buffer");
      return;
    }
  }

  data->ready = true;
  reshade::log::message(reshade::log::level::info, "mods::autotune initialized (D3D12)");
}

inline void OnDestroyDevice(reshade::api::device* device) {
  auto* data = renodx::utils::data::Get<DeviceData>(device);
  if (data == nullptr) return;
  // NOTE: ReShade exposes wait_idle() only on command_queue; GPU is already drained at destroy_device time.
  if (data->stats_uav.handle != 0u) device->destroy_resource_view(data->stats_uav);
  if (data->stats_buffer.handle != 0u) device->destroy_resource(data->stats_buffer);
  for (auto& readback : data->readback) {
    if (readback.handle != 0u) device->destroy_resource(readback);
  }
  renodx::utils::data::Delete<DeviceData>(device);
}

inline reshade::api::resource_view GetStatsView(reshade::api::command_list* cmd_list) {
  auto* data = renodx::utils::data::Get<DeviceData>(cmd_list->get_device());
  if (data == nullptr || !data->ready) return {0};
  return data->stats_uav;
}

inline void OnPresent(
    reshade::api::command_queue* queue,
    reshade::api::swapchain* /*swapchain*/,
    const reshade::api::rect*, const reshade::api::rect*,
    uint32_t, const reshade::api::rect*) {
  if (setting_enabled == 0.f) return;

  auto* device = queue->get_device();
  auto* data = renodx::utils::data::Get<DeviceData>(device);
  if (data == nullptr || !data->ready) return;

  auto* cmd_list = queue->get_immediate_command_list();
  if (cmd_list == nullptr) return;

  const int slot = static_cast<int>(data->frame % READBACK_COUNT);

  // Snapshot this frame's stats, then reset for the next frame.
  cmd_list->barrier(
      data->stats_buffer,
      reshade::api::resource_usage::unordered_access,
      reshade::api::resource_usage::copy_source);
  cmd_list->copy_buffer_region(data->stats_buffer, 0, data->readback[slot], 0, BUFFER_SIZE);
  cmd_list->barrier(
      data->stats_buffer,
      reshade::api::resource_usage::copy_source,
      reshade::api::resource_usage::unordered_access);
  const uint32_t zeros[4] = {0u, 0u, 0u, 0u};
  cmd_list->clear_unordered_access_view_uint(data->stats_uav, zeros);
  queue->flush_immediate_command_list();

  // Map the oldest snapshot - (READBACK_COUNT - 1) frames old, guaranteed
  // GPU-complete, so this map never stalls. Skip the first cycle entirely:
  // the frame-0 snapshot predates the first clear and contains undefined
  // buffer contents.
  if (data->frame >= static_cast<uint64_t>(READBACK_COUNT)) {
    const int oldest = static_cast<int>((data->frame + 1) % READBACK_COUNT);
    void* mapped = nullptr;
    if (device->map_buffer_region(
            data->readback[oldest], 0, BUFFER_SIZE,
            reshade::api::map_access::read_only, &mapped)
        && mapped != nullptr) {
      uint32_t dwords[TOTAL_DWORDS];
      std::memcpy(dwords, mapped, BUFFER_SIZE);
      device->unmap_buffer_region(data->readback[oldest]);
      Analyze(dwords);
    }
  }

  data->frame++;
}

}  // namespace internal

// Attach the stats UAV binding to every custom shader. Only shaders whose
// HLSL actually declares the UAV will write to it; for the rest the push
// descriptor is harmless. Call before renodx::mods::shader::Use().
inline void SetupCustomShaders(renodx::mods::shader::CustomShaders& custom_shaders) {
  for (auto& [hash, custom_shader] : custom_shaders) {
    custom_shader.views.push_back({
        .type = reshade::api::descriptor_type::buffer_unordered_access_view,
        .slot = stats_uav_slot,
        .space = stats_uav_space,
        .get_view = &internal::GetStatsView,
    });
  }
}

inline SceneStats GetStats() {
  const std::unique_lock lock(internal::stats_mutex);
  return internal::smooth_stats;
}

inline Recommendation GetRecommendation() {
  const std::unique_lock lock(internal::stats_mutex);
  return internal::recommendation;
}

using RecommendationGetter = std::function<float(const Recommendation&)>;

// Builds the "Auto Tune" settings section.
// `apply_map` pairs an existing setting key with the recommendation field
// that should be written into it when the user clicks Apply.
inline std::vector<renodx::utils::settings::Setting*> NewSettings(
    std::vector<std::pair<std::string, RecommendationGetter>> apply_map) {
  auto* shared_map = new std::vector<std::pair<std::string, RecommendationGetter>>(std::move(apply_map));

  return {
      new renodx::utils::settings::Setting{
          .key = "AutoTuneEnabled",
          .binding = &internal::setting_enabled,
          .value_type = renodx::utils::settings::SettingValueType::BOOLEAN,
          .default_value = 1.f,
          .label = "Scene Analysis",
          .section = "Auto Tune (Suggest)",
          .tooltip = "Measures scene luminance/gamut statistics and suggests slider values. Nothing is applied automatically.",
      },
      new renodx::utils::settings::Setting{
          .value_type = renodx::utils::settings::SettingValueType::CUSTOM,
          .section = "Auto Tune (Suggest)",
          .on_draw = []() {
            if (internal::setting_enabled == 0.f) {
              ImGui::TextDisabled("Scene analysis is off.");
              return false;
            }
            const auto stats = GetStats();
            const auto rec = GetRecommendation();
            if (!stats.valid) {
              ImGui::TextDisabled("Waiting for frame statistics...");
              return false;
            }
            ImGui::Text("Scene: %s", rec.scene_label);
            ImGui::Text("Median: %.4f | P95: %.3f | P99.9: %.3f | Max: %.2f",
                        stats.p50, stats.p95, stats.p999, stats.max_luminance);
            ImGui::Text("Outside BT.709: %.2f%%", stats.wide_gamut_ratio * 100.f);
            ImGui::Separator();
            ImGui::Text("Suggested: Highlights %.0f | Shadows %.0f | Contrast %.0f | Blowout %.0f",
                        rec.highlights, rec.shadows, rec.contrast, rec.dechroma);
            return false;
          },
      },
      new renodx::utils::settings::Setting{
          .value_type = renodx::utils::settings::SettingValueType::BUTTON,
          .label = "Apply Suggestions",
          .section = "Auto Tune (Suggest)",
          .tooltip = "Writes the suggested values to the sliders above (same as moving them by hand).",
          .on_click = [shared_map]() {
            const auto rec = GetRecommendation();
            std::vector<std::pair<std::string, float>> pairs;
            pairs.reserve(shared_map->size());
            for (const auto& [key, getter] : *shared_map) {
              pairs.emplace_back(key, getter(rec));
            }
            renodx::utils::settings::UpdateSettings(pairs);
            return true;
          },
      },
  };
}

inline void Use(DWORD fdw_reason) {
  switch (fdw_reason) {
    case DLL_PROCESS_ATTACH:
      reshade::register_event<reshade::addon_event::init_device>(internal::OnInitDevice);
      reshade::register_event<reshade::addon_event::destroy_device>(internal::OnDestroyDevice);
      reshade::register_event<reshade::addon_event::present>(internal::OnPresent);
      break;
    case DLL_PROCESS_DETACH:
      reshade::unregister_event<reshade::addon_event::init_device>(internal::OnInitDevice);
      reshade::unregister_event<reshade::addon_event::destroy_device>(internal::OnDestroyDevice);
      reshade::unregister_event<reshade::addon_event::present>(internal::OnPresent);
      break;
  }
}

}  // namespace renodx::mods::autotune
