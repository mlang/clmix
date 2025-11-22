// Simple command-line tool to mix electronic music
//
// At least C++23 is required to compile this program.
// No need for defensive programming.
// We aim for beautiful code.

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cctype>
#include <charconv>
#include <cmath>
#include <concepts>
#include <deque>
#include <cstdlib>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string_view>
#include <map>
#include <memory>
#include <numbers>
#include <numeric>
#include <optional>
#include <print>
#include <random>
#include <ranges>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <sstream>
#include <type_traits>
#include <utility>
#include <vector>

#include "vendor/mdspan.hpp"

// We rely on mdspan's C++23 multi-arg operator[] for indexing (e.g., out[i, ch]).
template<typename T>
using multichannel = Kokkos::mdspan<T, Kokkos::dextents<std::size_t, 2>>;

#include <readline/history.h>
#include <readline/readline.h>

#include <sndfile.hh>

#include <samplerate.h>

extern "C" {
#include <aubio/aubio.h>
}

#define MINIAUDIO_IMPLEMENTATION
#include <miniaudio.h>

namespace {

constexpr float kHeadroomDB = -2.0f;

template<std::floating_point T>
[[nodiscard]] constexpr T dbamp(T db) noexcept
{
  return std::pow(T(10.0), db * T(0.05));
}

template<std::floating_point T>
[[nodiscard]] constexpr T ampdb(T amp) noexcept
{
  return T(20.0) * std::log10(amp);
}

template<typename T>
requires (std::is_integral_v<T> || std::is_floating_point_v<T>)
[[nodiscard]] std::expected<T, std::string> parse_number(std::string_view s)
{
  static_assert(!std::is_same_v<T, bool>, "parse_number<bool> is not supported");
  T v{};
  const char* b = s.data();
  const char* e = b + s.size();

  auto to_msg = [](std::errc ec) -> std::string {
    assert(ec != std::errc());
    if (ec == std::errc::invalid_argument) return "not a number";
    if (ec == std::errc::result_out_of_range) return "out of range";
    return "parse error";
  };

  std::from_chars_result r = [&]{
    if constexpr (std::is_floating_point_v<T>)
      return std::from_chars(b, e, v, std::chars_format::general);
    return std::from_chars(b, e, v);
  }();

  constexpr std::errc ok{};
  if (r.ec == ok) {
    if (r.ptr != e) return std::unexpected(std::string("trailing characters"));
    return v;
  }
  return std::unexpected(to_msg(r.ec));
}

template<typename T>
class Interleaved {
  std::vector<T> storage;
public:
  uint32_t sample_rate = 0;
  multichannel<T> audio{}; // non-ownning view over 'storage'

  Interleaved() = default;

  Interleaved(uint32_t sr, std::size_t ch, std::size_t frames)
  : sample_rate(sr), storage(frames * ch), audio(storage.data(), frames, ch)
  { assert(ch > 0); }

  // move-only
  Interleaved(const Interleaved&) = delete;
  Interleaved& operator=(const Interleaved&) = delete;

  Interleaved(Interleaved&&) noexcept = default;
  Interleaved& operator=(Interleaved&&) noexcept = default;

  [[nodiscard]] std::size_t frames() const noexcept { return audio.extent(0); }
  [[nodiscard]] std::size_t channels() const noexcept { return audio.extent(1); }
  [[nodiscard]] std::size_t samples() const noexcept { return frames() * channels(); }
  [[nodiscard]] T* data() noexcept { return storage.data(); }
  [[nodiscard]] const T* data() const noexcept { return storage.data(); }

  [[nodiscard]] T peak() const noexcept {
    T p = T(0);
    for (const T& s : storage) {
      T a = std::abs(s);
      if constexpr (std::is_floating_point_v<T>) {
        if (!std::isfinite(a)) continue;
      }
      if (a > p) p = a;
    }
    return p;
  }

  void resize(std::size_t new_frames) {
    const std::size_t ch = channels();
    storage.resize(new_frames * ch);
    audio = multichannel<T>(storage.data(), new_frames, ch);
  }

  // Scale all samples in-place by gain.
  template<typename U> requires std::is_arithmetic_v<U>
  Interleaved& operator*=(U gain) noexcept
  {
    const T g = static_cast<T>(gain);
    for (T& s: storage) s *= g;
    return *this;
  }
};

[[nodiscard]] Interleaved<float> change_tempo(
  const Interleaved<float>& in,
  double from_bpm, double to_bpm,
  uint32_t to_rate,
  int converter_type
) {
  const std::size_t channels = in.channels();
  const std::size_t in_frames_sz = in.frames();

  if (channels == 0 || in_frames_sz == 0)
    return {};

  if (from_bpm <= 0.0 || to_bpm <= 0.0 || in.sample_rate == 0 || to_rate == 0)
    throw std::invalid_argument("BPM and sample rates must be positive.");

  if (!std::in_range<long>(in_frames_sz))
    throw std::overflow_error("Input too large for libsamplerate (frame count exceeds 'long').");

  const long in_frames = static_cast<long>(in_frames_sz);

  // Resampling ratio so that when played at to_rate, tempo becomes to_bpm.
  // Derivation: tempo_out = tempo_in * (to_rate / (ratio * from_rate))
  // -> ratio = (to_rate/from_rate) * (from_bpm/to_bpm)
  const double ratio =
      (static_cast<double>(to_rate) / static_cast<double>(in.sample_rate)) *
      (from_bpm / to_bpm);

  if (!(ratio > 0.0) || !std::isfinite(ratio))
    throw std::invalid_argument("Invalid resampling ratio derived from inputs.");

  // Estimate output frames (add 1 for safety).
  const double est_out_frames_d = std::ceil(static_cast<double>(in_frames) * ratio) + 1.0;
  if (!std::in_range<long>(static_cast<std::size_t>(est_out_frames_d)))
    throw std::overflow_error("Output too large for libsamplerate (frame count exceeds 'long').");

  const long out_frames_est = static_cast<long>(est_out_frames_d);

  Interleaved<float> out(to_rate, channels, static_cast<std::size_t>(out_frames_est));

  SRC_DATA data{};
  data.data_in = in.data();
  data.data_out = out.data();
  data.input_frames = in_frames;
  data.output_frames = out_frames_est;
  data.end_of_input = 1;
  data.src_ratio = ratio;

  if (!std::in_range<int>(channels))
    throw std::invalid_argument("Channel count too large for libsamplerate.");
  const auto ch = static_cast<int>(channels);

  const int err = src_simple(&data, converter_type, ch);
  if (err != 0)
    throw std::runtime_error(src_strerror(err));

  out.resize(static_cast<std::size_t>(data.output_frames_gen));

  return out;
}

// Encapsulated metronome state and processing
struct Metronome {
  std::atomic<double> bpm{120.0};
  std::atomic<unsigned> bpb{4u};

  // runtime state
  uint64_t lastBeatIndex = 0;
  int clickSamplesLeft = 0;
  int clickLen = 0; // in device samples
  float clickPhase = 0.f;
  float clickAmp = 0.f;
  float clickFreqCurHz = 0.f;

  // click parameters
  float clickFreqHzBeat = 1000.f;
  float clickFreqHzDownbeat = 1600.f; // higher pitch for downbeat
  float downbeatAmp = 0.35f;
  float beatAmp = 0.18f;

  void reset_runtime() {
    lastBeatIndex = std::numeric_limits<uint64_t>::max();
    clickSamplesLeft = 0;
    clickLen = 0;
    clickPhase = 0.f;
    clickAmp = 0.f;
    clickFreqCurHz = clickFreqHzBeat;
  }

  void prepare_after_seek(double posSrcFrames, double framesPerBeatSrc) {
    reset_runtime();
    const double q = std::max(0.0, posSrcFrames) / framesPerBeatSrc;
    const double qFloor = std::floor(q);
    const uint64_t bi = static_cast<uint64_t>(qFloor);
    const double frac = q - qFloor;
    if (std::abs(frac) < 1e-9) {
      lastBeatIndex = bi - 1; // prime to trigger click at boundary (wraps for beat 0)
    } else {
      lastBeatIndex = bi;
    }
  }

  [[nodiscard]] float process(double posSrcFrames, double framesPerBeatSrc, uint32_t devRate) {
    if (clickLen == 0) clickLen = std::max(1, (int)(devRate / 100)); // ~10ms
    uint64_t beatIndex = (uint64_t)std::floor(std::max(0.0, posSrcFrames) / framesPerBeatSrc);
    if (beatIndex != lastBeatIndex) {
      lastBeatIndex = beatIndex;
      clickSamplesLeft = clickLen;
      clickPhase = 0.f;
      unsigned curBpb = std::max(1u, bpb.load());
      bool downbeat = (beatIndex % static_cast<uint64_t>(curBpb)) == 0;
      clickAmp = downbeat ? downbeatAmp : beatAmp;
      clickFreqCurHz = downbeat ? clickFreqHzDownbeat : clickFreqHzBeat;
    }
    float click = 0.f;
    if (clickSamplesLeft > 0) {
      float env = (float)clickSamplesLeft / (float)clickLen; // linear decay
      clickPhase += 2.0f * std::numbers::pi_v<float> * clickFreqCurHz / (float)devRate;
      click = clickAmp * std::sinf(clickPhase) * env;
      --clickSamplesLeft;
    }
    return click;
  }
};

// Track metadata persisted in the DB
struct TrackInfo {
  std::filesystem::path filename;
  unsigned beats_per_bar = 4;
  double bpm = 120.0; // required > 0
  double upbeat_beats = 0.0;
  double time_offset_sec = 0.0;
  std::vector<int> cue_bars; // 1-based bar numbers
  std::set<std::string> tags; // unique tags
};

 // Simple text DB:
 // Each line: "filename with quotes" <space> <beats_per_bar> <space> <bpm> <space> <upbeat_beats> <space> <time_offset_sec> <space> <cues_csv_or_-> <space> <tags_csv_or_->
 // Lines starting with '#' or blank lines are ignored.
struct TrackDB {
  std::map<std::filesystem::path, TrackInfo> items;

  static std::filesystem::path norm(const std::filesystem::path& p) {
    return p.lexically_normal();
  }

  [[nodiscard]] TrackInfo* find(const std::filesystem::path& file) {
    auto it = items.find(norm(file));
    return (it == items.end()) ? nullptr : &it->second;
  }

  void upsert(const TrackInfo& info) {
    items[norm(info.filename)] = info;
  }

  bool load(const std::filesystem::path& dbfile) {
    items.clear();
    std::ifstream in(dbfile);
    if (!in.is_open()) {
      // Not an error if file is missing: treat as empty DB
      return false;
    }
    std::string line;
    while (std::getline(in, line)) {
      // Trim leading spaces
      auto it = std::find_if_not(line.begin(), line.end(),
                                 [](unsigned char ch){ return std::isspace(ch); });
      if (it == line.end() || *it == '#') continue; // blank or comment

      std::istringstream iss(line);
      std::string fname;
      int beats = 0;
      std::string bpm_tok, upbeat_tok, toffs_tok, cues_tok, tags_tok;

      // Require all fields in order; skip malformed lines
      if (!(iss >> std::ws >> std::quoted(fname)
                >> beats >> bpm_tok >> upbeat_tok >> toffs_tok >> cues_tok)) {
        std::println(std::cerr,
                     "Warning: failed to parse trackdb line (missing required fields): {}",
                     line);
        continue;
      }

      // tags are optional; if missing, tags_tok stays empty
      if (!(iss >> tags_tok)) {
        tags_tok.clear();
      }

      auto bpm_v    = parse_number<double>(bpm_tok);
      auto upbeat_v = parse_number<double>(upbeat_tok);
      auto toffs_v  = parse_number<double>(toffs_tok);

      if (!bpm_v || *bpm_v <= 0.0 || beats <= 0 || !upbeat_v || !toffs_v) {
        std::string err;
        if (!bpm_v)          err = "bpm: " + bpm_v.error();
        else if (beats <= 0) err = "beats_per_bar: must be > 0";
        else if (!upbeat_v)  err = "upbeat_beats: " + upbeat_v.error();
        else if (!toffs_v)   err = "time_offset_sec: " + toffs_v.error();
        else                 err = "unknown numeric error";

        std::println(std::cerr,
                     "Warning: invalid numeric fields in trackdb line ({}): {}",
                     err, line);
        continue;
      }

      std::vector<int> cues;
      if (cues_tok != "-") {
        std::stringstream ss(cues_tok);
        std::string tok;
        bool ok = true;
        while (std::getline(ss, tok, ',')) {
          auto bar = parse_number<int>(tok);
          if (!bar || *bar <= 0) { ok = false; break; }
          cues.push_back(*bar);
        }
        if (!ok) {
          std::println(std::cerr,
                       "Warning: invalid cue list in trackdb line: {}",
                       line);
          continue;
        }
        std::sort(cues.begin(), cues.end());
        cues.erase(std::unique(cues.begin(), cues.end()), cues.end());
      }

      // Parse tags (optional, "-" or empty => no tags)
      std::set<std::string> tags;
      if (!tags_tok.empty() && tags_tok != "-") {
        std::stringstream ss(tags_tok);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
          if (tok.empty()) continue;
          // trim simple leading/trailing spaces
          auto first = std::find_if_not(tok.begin(), tok.end(),
                                        [](unsigned char c){ return std::isspace(c); });
          auto last  = std::find_if_not(tok.rbegin(), tok.rend(),
                                        [](unsigned char c){ return std::isspace(c); }).base();
          if (first >= last) continue;
          tags.emplace(first, last); // std::set deduplicates
        }
      }

      TrackInfo ti;
      ti.filename = fname;
      ti.beats_per_bar = static_cast<unsigned>(beats);
      ti.bpm = *bpm_v;
      ti.upbeat_beats = *upbeat_v;
      ti.time_offset_sec = *toffs_v;
      ti.cue_bars = std::move(cues);
      ti.tags = std::move(tags);
      upsert(ti);
    }
    return true;
  }

  bool save(const std::filesystem::path& dbfile) const {
    std::ofstream out(dbfile, std::ios::trunc);
    if (!out.is_open()) {
      return false;
    }
    out << "# clmix track db: \"filename\" beats_per_bar bpm upbeat_beats time_offset_sec cues_csv_or_- tags_csv_or_-\n";
    for (const auto& [key, ti] : items) {
      std::string cues_tok;
      if (!ti.cue_bars.empty()) {
        for (size_t i = 0; i < ti.cue_bars.size(); ++i) {
          if (i) cues_tok.push_back(',');
          cues_tok += std::to_string(ti.cue_bars[i]);
        }
      } else {
        cues_tok = "-";
      }

      std::string tags_tok;
      if (!ti.tags.empty()) {
        size_t i = 0;
        for (const auto& tag : ti.tags) {
          if (i++) tags_tok.push_back(',');
          tags_tok += tag;
        }
      } else {
        tags_tok = "-";
      }

      out << std::quoted(ti.filename.generic_string()) << ' '
          << ti.beats_per_bar << ' '
          << std::to_string(ti.bpm) << ' '
          << std::to_string(ti.upbeat_beats) << ' '
          << std::to_string(ti.time_offset_sec) << ' '
          << cues_tok << ' '
          << tags_tok << '\n';
    }
    return true;
  }
};

struct PlayerState {
  std::atomic<bool> playing{false};
  std::shared_ptr<Interleaved<float>> track; // set before play; not swapped while playing
  std::atomic<float> trackGainDB{0.f}; // Track gain in dB (0 = unity; negative attenuates)
  std::atomic<double> upbeatBeats{0.0};
  std::atomic<double> timeOffsetSec{0.0};

  // Seek control (source frames)
  std::atomic<bool> seekPending{false};
  std::atomic<double> seekTargetFrames{0.0};

  // Playback runtime (audio thread)
  double srcPos = 0.0;              // in source frames (fractional)

  Metronome metro;
};

PlayerState g_player;
TrackDB g_db;

uint32_t g_device_rate = 44100;
uint32_t g_device_channels = 2;

std::vector<std::filesystem::path> g_mix_tracks;

struct MixCue {
  double frame;  // absolute cue frame in mix timeline
  long   bar;    // 1-based global bar number in the mix
};

std::vector<MixCue> g_mix_cues;

unsigned g_mix_bpb = 4;
double g_mix_bpm = 120.0;

[[nodiscard]] Interleaved<float> load_track(std::filesystem::path file)
{
  SndfileHandle sf(file.string());
  if (sf.error()) {
    throw std::runtime_error("Failed to open audio file: " + file.string());
  }

  const sf_count_t frames = sf.frames();
  const int sr = sf.samplerate();
  if (sr <= 0) {
    throw std::runtime_error("Invalid sample rate in file: " + file.string());
  }
  Interleaved<float> track(static_cast<uint32_t>(sr), static_cast<std::size_t>(sf.channels()), static_cast<std::size_t>(frames));

  const sf_count_t read_frames = sf.readf(track.data(), frames);
  if (read_frames < 0) {
    throw std::runtime_error("Failed to read audio data from file: " + file.string());
  }
  if (read_frames != frames) {
    track.resize(static_cast<std::size_t>(read_frames));
  }

  return track;
}

[[nodiscard]] float detect_bpm(const Interleaved<float>& track)
{
  if (track.sample_rate == 0 || track.channels() == 0 || track.frames() == 0) {
    throw std::invalid_argument("detect_bpm: invalid or empty track");
  }

  const uint_t win_s = 1024;
  const uint_t hop_s = 512;
  const uint_t samplerate = static_cast<uint_t>(track.sample_rate);

  using tempo_ptr = std::unique_ptr<aubio_tempo_t, decltype(&del_aubio_tempo)>;
  using fvec_ptr  = std::unique_ptr<fvec_t,        decltype(&del_fvec)>;

  tempo_ptr tempo{ new_aubio_tempo((char*)"default", win_s, hop_s, samplerate), &del_aubio_tempo };
  if (!tempo) {
    throw std::runtime_error("aubio: failed to create tempo object");
  }

  fvec_ptr inbuf{ new_fvec(hop_s), &del_fvec };
  fvec_ptr out{ new_fvec(1), &del_fvec };
  if (!inbuf || !out) {
    throw std::runtime_error("aubio: failed to allocate buffers");
  }

  const std::size_t channels = track.channels();
  const std::size_t total_frames = track.frames();

  for (std::size_t frame = 0; frame < total_frames; frame += hop_s) {
    for (uint_t j = 0; j < hop_s; ++j) {
      const std::size_t fr = frame + j;
      float v = 0.f;
      if (fr < total_frames) {
        float sum = 0.f;
        for (std::size_t c = 0; c < channels; ++c) {
          sum += track.audio[fr, c];
        }
        v = sum / static_cast<float>(channels);
      }
      inbuf->data[j] = v;
    }
    aubio_tempo_do(tempo.get(), inbuf.get(), out.get());
  }

  const float bpm = aubio_tempo_get_bpm(tempo.get());

  return bpm;
}

void apply_two_pass_limiter_db(Interleaved<float>& buf,
                               float ceiling_dB = -1.0f,
                               float max_attack_db_per_s = 200.0f,
                               float max_release_db_per_s = 40.0f)
{
  const uint32_t sr = buf.sample_rate;
  const size_t frames = buf.frames();
  const size_t ch = buf.channels();
  if (sr == 0 || frames == 0 || ch == 0) return;

  assert(max_attack_db_per_s > 0.0f);
  assert(max_release_db_per_s > 0.0f);

  // 1) Required attenuation (dB) to meet ceiling at each frame (computed on demand)
  auto required_att_dB = [&](size_t f) -> float {
    float pk = 0.f;
    for (size_t c = 0; c < ch; ++c) {
      float v = buf.audio[f, c];
      if (!std::isfinite(v)) continue;
      v = std::fabs(v);
      if (v > pk) pk = v;
    }
    // attenuation needed in dB (>= 0)
    return (pk > 0.f) ? std::max(0.f, ampdb(pk) - ceiling_dB) : 0.f;
  };

  // 2) Backward pass: limit how fast attenuation may increase (attack slope)
  const float attack_step  = max_attack_db_per_s / static_cast<float>(sr);
  std::vector<float> att(frames, 0.f);
  att[frames - 1] = required_att_dB(frames - 1);
  for (size_t i = frames - 1; i-- > 0; ) {
    att[i] = std::max(required_att_dB(i), att[i + 1] - attack_step);
  }

  // 3) Forward pass: limit how fast attenuation may decrease (release slope)
  const float release_step = max_release_db_per_s / static_cast<float>(sr);
  for (size_t i = 1; i < frames; ++i) {
    att[i] = std::max(att[i], att[i - 1] - release_step);
  }

  // 4) Apply gain: g = dbamp(-att_dB) clamped to [0,1]
  for (size_t f = 0; f < frames; ++f) {
    float g = std::clamp(dbamp(-att[f]), 0.0f, 1.0f);
    for (size_t c = 0; c < ch; ++c) {
      buf.audio[f, c] *= g;
    }
  }
}

// Build a rendered mix as a single Track at device rate/channels.
// Aligns last cue of A to first cue of B. Applies fade-in from start->first cue,
// unity between cues, fade-out from last cue->end. Accumulates global cue frames.
[[nodiscard]] std::shared_ptr<Interleaved<float>> build_mix_track(
  const std::vector<std::filesystem::path>& files,
  std::optional<double> force_bpm = std::nullopt,
  int converter_type = SRC_LINEAR
) {
  if (files.empty()) return {};

  // Collect TrackInfo and ensure cues exist
  std::vector<TrackInfo> tis;
  tis.reserve(files.size());
  for (auto& f : files) {
    auto* ti = g_db.find(f);
    if (!ti || ti->cue_bars.empty()) {
      throw std::runtime_error("Track missing in DB or has no cues: " + f.generic_string());
    }
    tis.push_back(*ti);
  }

  // Mix BPM default: mean of track bpms (unless forced)
  auto bpm = force_bpm.value_or([&]{
    const auto bpms = tis | std::views::transform(&TrackInfo::bpm);
    return std::ranges::fold_left(bpms, 0.0, std::plus<double>{}) / static_cast<double>(tis.size());
  }());
  g_mix_bpm = bpm;
  g_mix_bpb = tis.front().beats_per_bar;

  const uint32_t outRate = g_device_rate;
  if (!std::in_range<int>(g_device_channels))
    throw std::invalid_argument("Device channel count not representable as int");
  const int outCh = static_cast<int>(g_device_channels);
  const double fpb = (double)outRate * 60.0 / bpm;

  struct Item {
    std::filesystem::path file;
    TrackInfo ti;
    Interleaved<float> res;
    double firstCue;
    double lastCue;
    double offset;
  };
  std::vector<Item> items;
  items.reserve(files.size());

  // Parallel per-track processing with std::async (exceptions propagate via future::get)
  std::vector<std::future<Item>> futs;
  futs.reserve(files.size());

  for (size_t i = 0; i < files.size(); ++i) {
    futs.push_back(std::async(std::launch::async, [&, i]() -> Item {
      Interleaved<float> t = load_track(files[i]);
      const auto& ti = tis[i];

      {
        const float targetHeadroom = dbamp(kHeadroomDB);
        const float peak = t.peak();
        if (peak > 0.f && peak > targetHeadroom) {
          t *= targetHeadroom / peak;
        }
      }
      

      auto res = change_tempo(t, ti.bpm, bpm, outRate, converter_type);
      size_t frames = res.frames();

      int firstBar = ti.cue_bars.front();
      int lastBar  = ti.cue_bars.back();
      double shiftOut = ti.upbeat_beats * fpb + ti.time_offset_sec * (double)outRate;
      double firstCue = shiftOut + (double)(firstBar - 1) * (double)ti.beats_per_bar * fpb;
      double lastCue  = shiftOut + (double)(lastBar  - 1) * (double)ti.beats_per_bar * fpb;

      // Clamp
      if (frames == 0) { firstCue = lastCue = 0.0; }
      else {
        firstCue = std::clamp(firstCue, 0.0, (double)(frames - 1));
        lastCue  = std::clamp(lastCue,  0.0, (double)(frames - 1));
      }

      return Item{ files[i], ti, std::move(res), firstCue, lastCue, 0.0 };
    }));
  }

  for (auto& f : futs) {
    items.push_back(f.get());
  }

  // Offsets: align last cue of A with first cue of B
  if (!items.empty()) {
    items[0].offset = -items[0].firstCue;
    for (size_t i = 1; i < items.size(); ++i) {
      items[i].offset = items[i-1].offset + items[i-1].lastCue - items[i].firstCue;
    }
    double minOff = 0.0;
    for (auto& it : items) minOff = std::min(minOff, it.offset);
    if (minOff < 0.0) for (auto& it : items) it.offset -= minOff;
  }

  // Determine total frames
  size_t totalFrames = 0;
  for (auto& it : items) {
    totalFrames = std::max(totalFrames, (size_t)std::ceil(it.offset) + it.res.frames());
  }

  auto out = std::make_shared<Interleaved<float>>(outRate, (size_t)outCh, totalFrames);
  std::fill_n(out->data(), out->samples(), 0.0f);

  // Envelope: equal-power fade-in from start->firstCue, unity in [firstCue,lastCue], equal-power fade-out lastCue->end
  auto env = [](size_t f, size_t frames, double firstCue, double lastCue)->float {
    if (frames == 0) return 0.0f;
    if (lastCue < firstCue) std::swap(lastCue, firstCue);
    if ((double)f <= firstCue) {
      if (firstCue <= 0.0) return 1.0f;
      double p = (double)f / firstCue; // 0..1
      return (float)std::sin(0.5 * std::numbers::pi_v<double> * p);
    } else if ((double)f >= lastCue) {
      double denom = (double)frames - lastCue;
      if (denom <= 1e-12) return 0.0f;
      double p = ((double)f - lastCue) / denom; // 0..1
      return (float)std::cos(0.5 * std::numbers::pi_v<double> * p);
    } else {
      return 1.0f;
    }
  };


  // Mix down to out channels
  const size_t outChS = (size_t)outCh;
  for (auto& it : items) {
    const size_t inChS = it.res.channels();
    for (size_t f = 0; f < it.res.frames(); ++f) {
      double absF = it.offset + (double)f;
      if (absF < 0.0) continue;
      size_t outF = (size_t)absF;
      if (outF >= totalFrames) break;
      float a = env(f, it.res.frames(), it.firstCue, it.lastCue);
      if (a <= 0.0f) continue;

      for (size_t ch = 0; ch < outChS; ++ch) {
        size_t sC = ch % inChS;
        out->audio[outF, ch] += a * it.res.audio[f, sC];
      }
    }
  }

  // Final offline two-pass limiter for transparent ceiling control
  apply_two_pass_limiter_db(*out, -1.0f, 200.0f, 40.0f);

  // Accumulated cue frames and global bar numbers in mix timeline
  g_mix_cues.clear();
  for (auto& it : items) {
    for (int bar : it.ti.cue_bars) {
      double shiftOut = it.ti.upbeat_beats * fpb
                        + it.ti.time_offset_sec * (double)outRate;
      double local = shiftOut
                     + (double)(bar - 1) * (double)it.ti.beats_per_bar * fpb;
      double mixFrame = it.offset + local;

      // Compute global bar index from mixFrame using integer math on beats.
      double beatsFromZero = mixFrame / fpb;
      long barIdx = (long)std::floor(beatsFromZero / (double)g_mix_bpb) + 1;

      g_mix_cues.push_back(MixCue{mixFrame, barIdx});
    }
  }

  // Sort by frame, then by bar index for stability
  std::sort(g_mix_cues.begin(), g_mix_cues.end(),
            [](const MixCue& a, const MixCue& b) {
              if (a.frame < b.frame) return true;
              if (a.frame > b.frame) return false;
              return a.bar < b.bar;
            });

  // Remove near-duplicate cue frames caused by alignment overlaps
  {
    constexpr double eps = 1e-6;
    auto it = std::unique(g_mix_cues.begin(), g_mix_cues.end(),
                          [](const MixCue& a, const MixCue& b){
                            return std::abs(a.frame - b.frame) <= eps;
                          });
    g_mix_cues.erase(it, g_mix_cues.end());
  }

  return out;
}

void play(
  PlayerState &player, multichannel<float> output, uint32_t devRate
)
{
  if (player.track) {
    auto &track = *player.track;
    const auto bpm = std::max(1.0, player.metro.bpm.load());
    const float gainLin = dbamp(player.trackGainDB.load());
    const size_t srcCh = track.channels();
    const size_t totalSrcFrames = track.frames();
    if (totalSrcFrames == 0) return;


    const double framesPerBeatSrc = (double)track.sample_rate * 60.0 / (double)bpm;
    const double incrSrcPerOut = (double)track.sample_rate / (double)devRate;
    const double shiftSrc = player.upbeatBeats.load() * framesPerBeatSrc
                            + player.timeOffsetSec.load() * (double)track.sample_rate;

    double pos = player.srcPos;

    for (size_t i = 0; i < output.extent(0); ++i) {
      // Quantized seek: apply pending seek at the next bar boundary
      if (player.seekPending.load()) {
        unsigned bpbNow = std::max(1u, player.metro.bpb.load());
        const double adjNow  = std::max(0.0, pos - shiftSrc);
        const double adjNext = std::max(0.0, pos + incrSrcPerOut - shiftSrc);
        uint64_t beatNow  = (uint64_t)std::floor(adjNow / framesPerBeatSrc);
        uint64_t beatNext = (uint64_t)std::floor(adjNext / framesPerBeatSrc);
        bool crossesBeat = (beatNext != beatNow);
        bool nextIsBarStart = (beatNext % static_cast<uint64_t>(bpbNow)) == 0;
        if (crossesBeat && nextIsBarStart) {
          pos = player.seekTargetFrames.load();
          player.seekPending.store(false);
          player.metro.prepare_after_seek(pos - shiftSrc, framesPerBeatSrc);
        }
      }

      if (pos >= (double)(totalSrcFrames - 1)) {
        player.playing.store(false);
        break;
      }

      // Linear interpolation per channel
      size_t i0 = (size_t)pos;
      double frac = pos - (double)i0;
      size_t i1 = std::min(i0 + 1, totalSrcFrames - 1);

      float click = player.metro.process(pos - shiftSrc, framesPerBeatSrc, devRate);

      // Write each output channel from the corresponding source channel (wrap if more outs)
      for (size_t ch = 0; ch < output.extent(1); ++ch) {
        size_t srcC = ch % srcCh;
        float s0 = track.audio[i0, srcC];
        float s1 = track.audio[i1, srcC];
        float smp = std::lerp(s0, s1, static_cast<float>(frac));
        float mix = (smp * gainLin) + click;
        output[i, ch] = mix;
      }

      pos += incrSrcPerOut;
    }

    player.srcPos = pos;
  }
}

void callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount)
{
  PlayerState &player = *static_cast<PlayerState*>(pDevice->pUserData);

  if (player.playing.load()) {
    multichannel<float> output(static_cast<float*>(pOutput),
      frameCount, pDevice->playback.channels
    );
    play(player, output, pDevice->sampleRate);
  }
}

// Shell-style tokenizer supporting quotes and backslashes.
// - Whitespace splits args when not inside quotes.
// - Single quotes: literals (no escapes inside).
// - Double quotes: supports backslash escaping of \" and \\ (simple treatment).
[[nodiscard]] std::vector<std::string> parse_command_line(const std::string& s) {
  std::vector<std::string> out;
  std::string cur;
  bool in_single = false, in_double = false, escape = false;

  auto push = [&](){
    out.push_back(cur);
    cur.clear();
  };

  for (size_t i = 0; i < s.size(); ++i) {
    char ch = s[i];

    if (in_single) {
      if (ch == '\'') {
        in_single = false;
      } else {
        cur.push_back(ch);
      }
      continue;
    }

    if (escape) {
      cur.push_back(ch);
      escape = false;
      continue;
    }

    if (in_double) {
      if (ch == '\\') {
        escape = true;
      } else if (ch == '"') {
        in_double = false;
      } else {
        cur.push_back(ch);
      }
      continue;
    }

    // Outside quotes
    if (std::isspace(static_cast<unsigned char>(ch))) {
      if (!cur.empty()) push();
      continue;
    }
    if (ch == '\'') { in_single = true; continue; }
    if (ch == '"')  { in_double = true; continue; }
    if (ch == '\\') { escape = true; continue; }

    cur.push_back(ch);
  }

  if (escape) { cur.push_back('\\'); } // trailing backslash literal
  if (!cur.empty()) {
    push();
  }

  return out;
}

// Simple command registry.
using Command = std::move_only_function<void(std::span<const std::string>)>;

struct CommandEntry {
  std::string help;
  Command fn;
};

class REPL {
public:
  void register_command(std::string name, std::string help, Command fn) {
    commands_.emplace(std::move(name), CommandEntry{std::move(help), std::move(fn)});
  }

  void run(const char* prompt = "clmix> ") {
    running_ = true;
    while (running_) {
      char* line = readline(prompt);
      if (!line) break; // EOF (Ctrl-D)
      std::unique_ptr<char, decltype(&std::free)> guard(line, &std::free);

      std::string input(line);
      if (input.empty()) continue;

      add_history(line);
      auto args = parse_command_line(input);
      if (args.empty()) continue;

      auto it = commands_.find(args[0]);
      if (it == commands_.end()) {
        std::cerr << "Unknown command: " << args[0] << "\n";
        continue;
      }
      try {
        it->second.fn(std::span<const std::string>{args}.subspan(1));
      } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
      } catch (...) {
        std::cerr << "Unknown error.\n";
      }
    }
  }

  void stop() { running_ = false; }

  void print_help() const {
    std::cout << "Commands:\n";
    for (const auto& [name, entry] : commands_) {
      std::cout << "  " << name;
      if (!entry.help.empty()) std::cout << " - " << entry.help;
      std::cout << "\n";
    }
  }

private:
  std::map<std::string, CommandEntry> commands_;
  bool running_ = true;
};

void register_volume_command(REPL& repl, std::string label) {
  repl.register_command(
    "vol",
    "vol [dB] - get/set " + label + " volume in dB (0=unity, negative attenuates)",
    [label](std::span<const std::string> a){
      if (a.empty()) {
        float db = g_player.trackGainDB.load();
        float lin = dbamp(db);
        std::println(std::cout, "{} volume: {:.2f} dB (x{:.3f})", label, db, lin);
        return;
      }
      std::string s = a[0];
      if (s.size() >= 2) {
        std::string tail = s.substr(s.size() - 2);
        std::transform(tail.begin(), tail.end(), tail.begin(),
                       [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
        if (tail == "db") s.resize(s.size() - 2);
      }
      if (auto v = parse_number<float>(s)) {
        const float db = std::clamp(*v, -60.f, 12.f);
        g_player.trackGainDB.store(db);
        float lin = dbamp(db);
        std::println(std::cout, "{} volume set to {:.2f} dB (x{:.3f})", label, db, lin);
      } else {
        std::cerr << "Invalid dB value: " << v.error() << "\n";
      }
    }
  );
}

void run_track_info_shell(const std::filesystem::path& f, const std::filesystem::path& trackdb_path)
{
  Interleaved<float> t;
  try {
    t = load_track(f);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return;
  }
  auto tr = std::make_shared<Interleaved<float>>(std::move(t));

  double guessedBpm = 0.0;

  TrackInfo ti;
  if (auto* existing = g_db.find(f)) {
    ti = *existing;
  } else {
    ti.filename = f;
    ti.beats_per_bar = 4;
    ti.bpm = 120.f;
    try {
      guessedBpm = detect_bpm(*tr);
      if (guessedBpm > 0.f) {
        ti.bpm = guessedBpm;
      }
    } catch (const std::exception& e) {
      std::cerr << "BPM detection failed: " << e.what() << "\n";
    }
  }

  std::cout << "Opened " << f << "\n";
  if (guessedBpm > 0) std::println(std::cout, "Guessed BPM: {:.2f}", guessedBpm);
  std::println(std::cout, "BPM: {:.2f}", ti.bpm);
  std::cout << "Beats/bar: " << ti.beats_per_bar << "\n";
  std::println(std::cout, "Upbeat (beats): {:.3f}", ti.upbeat_beats);
  std::println(std::cout, "Time offset (s): {:.3f}", ti.time_offset_sec);

  // Initialize player state for this track (not playing yet)
  g_player.track = tr;
  g_player.srcPos = 0.0;
  g_player.seekTargetFrames.store(0.0);
  g_player.seekPending.store(false);
  // keep existing volume
  g_player.metro.reset_runtime();
  g_player.metro.bpm.store(ti.bpm);
  g_player.metro.bpb.store(std::max(1u, ti.beats_per_bar));
  g_player.upbeatBeats.store(ti.upbeat_beats);
  g_player.timeOffsetSec.store(ti.time_offset_sec);

  auto print_estimated_bars = [&](){
    auto bpmNow = std::max(1.0, g_player.metro.bpm.load());
    unsigned bpbNow = std::max(1u, g_player.metro.bpb.load());
    size_t totalFrames = tr->frames();
    double framesPerBeat = (double)tr->sample_rate * 60.0 / (double)bpmNow;
    double beats = (double)totalFrames / framesPerBeat;
    double bars = beats / static_cast<double>(bpbNow);
    std::println(std::cout, "Estimated bars: {:.2f}", bars);
  };
  print_estimated_bars();

  // Per-track subshell
  REPL sub;
  bool dirty = false;

  sub.register_command("help", "List commands", [&](std::span<const std::string>){ sub.print_help(); });

  sub.register_command("bpm", "bpm [value] - get/set BPM", [&](std::span<const std::string> a){
    if (a.empty()) {
      std::println(std::cout, "BPM: {:.2f}", g_player.metro.bpm.load());
      return;
    }
    if (auto v = parse_number<double>(a[0]); v) {
      if (*v <= 0.0) { std::cerr << "Invalid BPM: must be > 0\n"; return; }
      g_player.metro.bpm.store(*v);
      ti.bpm = *v;
      dirty = true;
      print_estimated_bars();
    } else {
      std::cerr << "Invalid BPM: " << v.error() << "\n";
    }
  });

  sub.register_command("bpb", "bpb [value] - get/set beats per bar", [&](std::span<const std::string> a){
    if (a.empty()) {
      std::cout << "Beats/bar: " << g_player.metro.bpb.load() << "\n";
      return;
    }
    if (auto v = parse_number<unsigned>(a[0]); v) {
      if (*v == 0u) { std::cerr << "Invalid beats-per-bar: must be > 0\n"; return; }
      g_player.metro.bpb.store(*v);
      ti.beats_per_bar = *v;
      dirty = true;
      print_estimated_bars();
    } else {
      std::cerr << "Invalid beats-per-bar: " << v.error() << "\n";
    }
  });

  sub.register_command("upbeat", "upbeat [beats] - get/set upbeat in beats (can be negative)", [&](std::span<const std::string> a){
    if (a.empty()) {
      std::println(std::cout, "Upbeat (beats): {:.3f}", ti.upbeat_beats);
      return;
    }
    if (auto v = parse_number<double>(a[0]); v) {
      ti.upbeat_beats = *v;
      g_player.upbeatBeats.store(*v);
      dirty = true;
      print_estimated_bars();
    } else {
      std::cerr << "Invalid upbeat value: " << v.error() << "\n";
    }
  });

  sub.register_command("offset", "offset [seconds] - get/set time offset in seconds (can be negative)", [&](std::span<const std::string> a){
    if (a.empty()) {
      std::println(std::cout, "Time offset (s): {:.3f}", ti.time_offset_sec);
      return;
    }
    if (auto v = parse_number<double>(a[0]); v) {
      ti.time_offset_sec = *v;
      g_player.timeOffsetSec.store(*v);
      dirty = true;
      print_estimated_bars();
    } else {
      std::cerr << "Invalid time offset: " << v.error() << "\n";
    }
  });

  sub.register_command("cue", "cue <bar> - add a cue at given bar (1-based)", [&](std::span<const std::string> a){
    if (a.size() != 1) {
      std::cerr << "Usage: cue <bar>\n";
      return;
    }
    if (auto bar = parse_number<int>(a[0]); bar) {
      if (*bar <= 0) { std::cerr << "Invalid bar: must be > 0\n"; return; }
      if (std::find(ti.cue_bars.begin(), ti.cue_bars.end(), *bar) == ti.cue_bars.end()) {
        ti.cue_bars.push_back(*bar);
        std::sort(ti.cue_bars.begin(), ti.cue_bars.end());
        dirty = true;
      }
      if (ti.cue_bars.empty()) {
        std::cout << "(no cues)\n";
      } else {
        std::cout << "Cues: ";
        for (size_t i = 0; i < ti.cue_bars.size(); ++i) {
          if (i) std::cout << ',';
          std::cout << ti.cue_bars[i];
        }
        std::cout << "\n";
      }
    } else {
      std::cerr << "Invalid bar: " << bar.error() << "\n";
    }
  });

  sub.register_command("uncue", "uncue <bar> - remove a cue", [&](std::span<const std::string> a){
    if (a.size() != 1) {
      std::cerr << "Usage: uncue <bar>\n";
      return;
    }
    if (auto bar = parse_number<int>(a[0]); bar) {
      if (std::erase(ti.cue_bars, *bar) > 0) {
        dirty = true;
      }
    } else {
      std::cerr << "Invalid bar: " << bar.error() << "\n";
    }
  });

  sub.register_command("cues", "List cue bars", [&](std::span<const std::string>){
    if (ti.cue_bars.empty()) {
      std::cout << "(no cues)\n";
      return;
    }
    for (size_t i = 0; i < ti.cue_bars.size(); ++i) {
      if (i) std::cout << ',';
      std::cout << ti.cue_bars[i];
    }
    std::cout << "\n";
  });

  register_volume_command(sub, "Track");

  sub.register_command("save", "Persist BPM/Beats-per-bar to trackdb", [&](std::span<const std::string>){
    g_db.upsert(ti);
    if (g_db.save(trackdb_path)) {
      std::cout << "Saved to " << trackdb_path << "\n";
      dirty = false;
    } else {
      std::cerr << "Failed to save DB to " << trackdb_path << "\n";
    }
  });

  sub.register_command("play", "Start playback with metronome overlay", [&](std::span<const std::string>){
    if (!g_player.track) {
      std::cerr << "No track loaded.\n";
      return;
    }
    g_player.seekPending.store(false);
    g_player.playing.store(true);
  });

  sub.register_command("stop", "Stop playback", [&](std::span<const std::string>){
    g_player.playing.store(false);
  });

  sub.register_command("seek", "seek <bar> - jump to given bar (1-based)", [&](std::span<const std::string> a){
    if (a.size() != 1) {
      std::cerr << "Usage: seek <bar>\n";
      return;
    }
    if (auto bar1 = parse_number<int>(a[0]); bar1) {
      int bar0 = std::max(0, *bar1 - 1);
      auto bpmNow = std::max(1.0, g_player.metro.bpm.load());
      unsigned bpbNow = std::max(1u, g_player.metro.bpb.load());
      double framesPerBeat = (double)tr->sample_rate * 60.0 / (double)bpmNow;
      double shift = ti.upbeat_beats * framesPerBeat + ti.time_offset_sec * (double)tr->sample_rate;
      double target = shift + (double)bar0 * static_cast<double>(bpbNow) * framesPerBeat;
      size_t totalFrames = tr->frames();
      if (target >= (double)totalFrames) target = (double)totalFrames - 1.0;
      if (target < 0.0) target = 0.0;
      if (!g_player.playing.load()) {
        g_player.srcPos = target;
        g_player.metro.prepare_after_seek(target - shift, framesPerBeat);
        g_player.seekTargetFrames.store(target);
        g_player.seekPending.store(false);
      } else {
        g_player.seekTargetFrames.store(target);
        g_player.seekPending.store(true);
      }
    } else {
      std::cerr << "Invalid bar number: " << bar1.error() << "\n";
    }
  });

  sub.register_command("exit", "Leave track shell", [&](std::span<const std::string>){
    sub.stop();
  });

  sub.run("track-info> ");
  g_player.playing.store(false);
}

}

int main(int argc, char** argv)
{
  if (argc < 2) {
    std::cerr << "Usage: clmix <trackdb.txt>\n"
                 "Provide the path to the track database file.\n";
    return 2;
  }
  const std::filesystem::path trackdb_path = argv[1];

  ma_device_config config = ma_device_config_init(ma_device_type_playback);
  config.playback.format   = ma_format_f32;
  config.playback.channels = 2;
  config.sampleRate        = 44100;
  config.noPreSilencedOutputBuffer = false;
  config.dataCallback      = callback;
  config.pUserData         = &g_player;
  ma_device device;
  ma_result res = ma_device_init(NULL, &config, &device);
  if (res != MA_SUCCESS) {
    std::cerr << "Audio device init failed: " << ma_result_description(res) << "\n";
    return 1;
  }

  res = ma_device_start(&device);
  if (res != MA_SUCCESS) {
    std::cerr << "Audio device start failed: " << ma_result_description(res) << "\n";
    ma_device_uninit(&device);
    return 1;
  }

  g_device_rate = device.sampleRate;
  g_device_channels = device.playback.channels;

  g_db.load(trackdb_path);

  REPL repl;

  repl.register_command("help", "List commands", [&](std::span<const std::string>){
    repl.print_help();
  });
  repl.register_command("exit", "Exit program", [&](std::span<const std::string>){
    repl.stop();
  });
  repl.register_command("quit", "Alias for exit", [&](std::span<const std::string>){
    repl.stop();
  });
  repl.register_command("echo", "Echo arguments; supports quoted args", [](std::span<const std::string> args){
    for (size_t i = 0; i < args.size(); ++i) {
      if (i) std::cout << ' ';
      std::cout << args[i];
    }
    std::cout << "\n";
  });
  repl.register_command("track-info", "track-info <file> - open per-track shell", [&](std::span<const std::string> args){
    if (args.size() != 1) {
      std::cerr << "Usage: track-info <file>\n";
      return;
    }
    run_track_info_shell(args[0], trackdb_path);
  });
  
  register_volume_command(repl, "Mix");
  
  // Mix commands
  repl.register_command("add", "add <file> - add track to mix (opens track-info if not in DB)", [&](std::span<const std::string> a){
    if (a.size() != 1) { std::cerr << "Usage: add <file>\n"; return; }
    std::filesystem::path f = a[0];
    if (!g_db.find(f)) {
      run_track_info_shell(f, trackdb_path);
    }
    if (!g_db.find(f)) {
      std::cerr << "Track still not in DB. Aborting.\n";
      return;
    }
    g_mix_tracks.push_back(f);
    try {
      g_player.playing.store(false);
      auto mixTrack = build_mix_track(g_mix_tracks);
      g_player.track = mixTrack;
      g_player.srcPos = 0.0;
      g_player.seekPending.store(false);
      g_player.seekTargetFrames.store(0.0);
      // keep existing volume (persist across mix rebuilds)
      g_player.metro.reset_runtime();
      g_player.metro.bpm.store(g_mix_bpm);
      g_player.metro.bpb.store(std::max(1u, g_mix_bpb));
      if (!g_mix_tracks.empty()) {
        if (auto* ti0 = g_db.find(g_mix_tracks.front())) {
          g_player.upbeatBeats.store(ti0->upbeat_beats);
          g_player.timeOffsetSec.store(ti0->time_offset_sec);
        } else {
          g_player.upbeatBeats.store(0.0);
          g_player.timeOffsetSec.store(0.0);
        }
      }
      std::cout << "Added. Mix size: " << g_mix_tracks.size()
                << ", BPM: " << g_mix_bpm
                << ", BPB: " << g_mix_bpb << "\n";
    } catch (const std::exception& e) {
      std::cerr << "Failed to build mix: " << e.what() << "\n";
    }
  });

  repl.register_command("bpm", "bpm [value] - show/set mix BPM (recomputes mix)", [&](std::span<const std::string> a){
    if (g_mix_tracks.empty()) { std::cerr << "No tracks in mix.\n"; return; }
    if (a.empty()) {
      std::println(std::cout, "Mix BPM: {:.2f}", g_mix_bpm);
      return;
    }
    if (auto v = parse_number<double>(a[0]); v) {
      if (*v <= 0.0) { std::cerr << "Invalid BPM: must be > 0\n"; return; }
      g_player.playing.store(false);
      auto mixTrack = build_mix_track(g_mix_tracks, *v);
      g_player.track = mixTrack;
      g_player.srcPos = 0.0;
      g_player.metro.reset_runtime();
      g_player.metro.bpm.store(g_mix_bpm);
      g_player.metro.bpb.store(std::max(1u, g_mix_bpb));
      if (!g_mix_tracks.empty()) {
        if (auto* ti0 = g_db.find(g_mix_tracks.front())) {
          g_player.upbeatBeats.store(ti0->upbeat_beats);
          g_player.timeOffsetSec.store(ti0->time_offset_sec);
        } else {
          g_player.upbeatBeats.store(0.0);
          g_player.timeOffsetSec.store(0.0);
        }
      }
      std::println(std::cout, "Mix BPM set to {:.2f} and recomputed.", g_mix_bpm);
    } else {
      std::cerr << "Invalid BPM: " << v.error() << "\n";
    }
  });

  repl.register_command("play", "Start playback (mix)", [&](std::span<const std::string>){
    if (!g_player.track) {
      if (g_mix_tracks.empty()) { std::cerr << "No tracks in mix.\n"; return; }
      try {
        g_player.track = build_mix_track(g_mix_tracks);
        g_player.srcPos = 0.0;
        g_player.metro.reset_runtime();
        g_player.metro.bpm.store(g_mix_bpm);
        g_player.metro.bpb.store(std::max(1u, g_mix_bpb));
        if (!g_mix_tracks.empty()) {
          if (auto* ti0 = g_db.find(g_mix_tracks.front())) {
            g_player.upbeatBeats.store(ti0->upbeat_beats);
            g_player.timeOffsetSec.store(ti0->time_offset_sec);
          } else {
            g_player.upbeatBeats.store(0.0);
            g_player.timeOffsetSec.store(0.0);
          }
        }
      } catch (const std::exception& e) {
        std::cerr << "Build mix failed: " << e.what() << "\n"; return;
      }
    }
    g_player.seekPending.store(false);
    g_player.playing.store(true);
  });

  repl.register_command("stop", "Stop playback", [&](std::span<const std::string>){
    g_player.playing.store(false);
  });

  repl.register_command("seek", "seek <bar> - jump to mix bar (1-based)", [&](std::span<const std::string> a){
    if (a.size() != 1 || !g_player.track) { std::cerr << "Usage: seek <bar>\n"; return; }
    if (auto bar1 = parse_number<int>(a[0]); bar1) {
      int bar0 = std::max(0, *bar1 - 1);
      double framesPerBeat = (double)g_player.track->sample_rate * 60.0 / g_mix_bpm;
      double shift = g_player.upbeatBeats.load() * framesPerBeat
                     + g_player.timeOffsetSec.load() * (double)g_player.track->sample_rate;
      double target = shift + (double)bar0 * (double)g_mix_bpb * framesPerBeat;
      size_t totalFrames = g_player.track->frames();
      if (target >= (double)totalFrames) target = (double)totalFrames - 1.0;
      if (target < 0.0) target = 0.0;
      if (!g_player.playing.load()) {
        g_player.srcPos = target;
        g_player.metro.prepare_after_seek(target - shift, framesPerBeat);
        g_player.seekTargetFrames.store(target, std::memory_order_relaxed);
        g_player.seekPending.store(false, std::memory_order_relaxed);
      } else {
        g_player.seekTargetFrames.store(target, std::memory_order_relaxed);
        g_player.seekPending.store(true, std::memory_order_release);
      }
    } else {
      std::cerr << "Invalid bar number: " << bar1.error() << "\n";
    }
  });

  repl.register_command("cue", "List all cue points in current mix", [&](std::span<const std::string>){
    if (g_mix_cues.empty()) {
      std::cout << "(no cues)\n";
      return;
    }
    for (const auto& c : g_mix_cues) {
      std::cout << "bar " << c.bar << "\n";
    }
  });

  repl.register_command("random", "random - build mix from all trackdb entries in random order", [&](std::span<const std::string>){
    if (g_db.items.empty()) {
      std::cerr << "Track DB is empty.\n";
      return;
    }
    std::vector<std::filesystem::path> all;
    all.reserve(g_db.items.size());
    for (const auto& kv : g_db.items) {
      const TrackInfo& ti = kv.second;
      if (!ti.cue_bars.empty()) all.push_back(ti.filename);
    }
    if (all.empty()) {
      std::cerr << "No tracks with cues in DB.\n";
      return;
    }
    std::mt19937 rng(std::random_device{}());
    std::shuffle(all.begin(), all.end(), rng);

    g_mix_tracks = std::move(all);
    std::cout << "Track order:\n";
    for (size_t i = 0; i < g_mix_tracks.size(); ++i) {
      std::cout << "  " << (i + 1) << ". " << g_mix_tracks[i].generic_string() << "\n";
    }
    try {
      g_player.playing.store(false);
      auto mixTrack = build_mix_track(g_mix_tracks);
      g_player.track = mixTrack;
      g_player.srcPos = 0.0;
      g_player.seekPending.store(false);
      g_player.seekTargetFrames.store(0.0);
      // keep existing volume (persist across mix rebuilds)
      g_player.metro.reset_runtime();
      g_player.metro.bpm.store(g_mix_bpm);
      g_player.metro.bpb.store(std::max(1u, g_mix_bpb));
      if (!g_mix_tracks.empty()) {
        if (auto* ti0 = g_db.find(g_mix_tracks.front())) {
          g_player.upbeatBeats.store(ti0->upbeat_beats);
          g_player.timeOffsetSec.store(ti0->time_offset_sec);
        } else {
          g_player.upbeatBeats.store(0.0);
          g_player.timeOffsetSec.store(0.0);
        }
      }
      std::cout << "Random mix created with " << g_mix_tracks.size()
                << " tracks. BPM: " << g_mix_bpm
                << ", BPB: " << g_mix_bpb << "\n";
    } catch (const std::exception& e) {
      std::cerr << "Failed to build random mix: " << e.what() << "\n";
    }
  });

  repl.register_command("export", "export <file.wav> - render mix to 24-bit WAV", [&](std::span<const std::string> a){
    if (a.size() != 1) { std::cerr << "Usage: export <file.wav>\n"; return; }
    if (g_mix_tracks.empty()) { std::cerr << "No tracks in mix.\n"; return; }

    const std::filesystem::path outPath = a[0];
    try {
      // Stop playback to avoid concurrent access while rendering/exporting
      g_player.playing.store(false);

      // Rebuild a fresh mix with current BPM and tracks
      auto mixTrack = build_mix_track(g_mix_tracks, g_mix_bpm, SRC_SINC_BEST_QUALITY);

      if (!std::in_range<sf_count_t>(mixTrack->frames())) {
        std::cerr << "Export failed: frame count too large for libsndfile.\n";
        return;
      }
      const sf_count_t frames = static_cast<sf_count_t>(mixTrack->frames());

      // Open 24-bit WAV for writing
      SndfileHandle sf(outPath.string(),
                       SFM_WRITE,
                       SF_FORMAT_WAV | SF_FORMAT_PCM_24,
                       static_cast<int>(mixTrack->channels()),
                       static_cast<int>(mixTrack->sample_rate));
      if (sf.error()) {
        std::cerr << "Failed to open output file: " << outPath << "\n";
        return;
      }

      // Write frames; libsndfile converts float -> PCM_24 and clips if needed
      const sf_count_t written = sf.writef(mixTrack->data(), frames);
      if (written != frames) {
        std::cerr << "Short write: wrote " << written << " of " << frames << " frames.\n";
      } else {
        std::cout << "Exported " << frames
                  << " frames (" << mixTrack->sample_rate << " Hz, "
                  << mixTrack->channels() << " ch) to " << outPath << "\n";
      }
    } catch (const std::exception& e) {
      std::cerr << "Export failed: " << e.what() << "\n";
    }
  });

  repl.run("clmix> ");

  ma_device_uninit(&device);

  return 0;
}
