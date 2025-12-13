// Simple command-line tool to mix electronic music

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cctype>
#include <charconv>
#include <cmath>
#include <concepts>
#include <cstring> // for std::memcpy
#include <cstdlib>
#include <cstdint>
#include <execution>
#include <expected>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
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
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "vendor/mdspan.hpp"

#include <boost/math/statistics/linear_regression.hpp>
#include <nlohmann/json.hpp>
#include <sndfile.hh>

#include <aubio/aubio.h>
#include <ebur128.h>
#include <getopt.h>
#include <miniaudio.h>
#include <readline/history.h>
#include <readline/readline.h>
#include <samplerate.h>

namespace {

// We rely on mdspan's C++23 multi-arg operator[] for indexing (e.g., out[i, ch]).
template<typename T>
using multichannel = Kokkos::mdspan<T, Kokkos::dextents<size_t, 2>>;

using boost::math::statistics::simple_ordinary_least_squares_with_R_squared;
using nlohmann::json;
using std::abs, std::ceil, std::clamp, std::floor, std::min, std::max;
using std::cerr, std::cout;
using std::expected, std::unexpected;
using std::filesystem::path;
using std::floating_point;
using std::in_range;
using std::is_arithmetic_v, std::is_floating_point_v, std::is_integral_v,
      std::is_same_v;
using std::optional, std::nullopt;
using std::println;
namespace ranges { using namespace std::ranges; }
namespace views { using namespace std::views; }
using std::runtime_error;
using std::shared_ptr, std::unique_ptr;
using std::string, std::string_view;
using std::vector;

template<floating_point T>
[[nodiscard]] constexpr T dbamp(T db) noexcept
{ return std::pow(T(10.0), db * T(0.05)); }

template<floating_point T>
[[nodiscard]] constexpr T ampdb(T amp) noexcept
{ return T(20.0) * std::log10(amp); }

template<typename T>
requires (is_integral_v<T> || is_floating_point_v<T>)
[[nodiscard]] expected<T, string> parse_number(string_view s)
{
  static_assert(!is_same_v<T, bool>, "parse_number<bool> is not supported");
  T v{};
  const char* b = s.data();
  const char* e = b + s.size();

  auto to_msg = [](std::errc ec) -> string {
    assert(ec != std::errc());
    if (ec == std::errc::invalid_argument) return "not a number";
    if (ec == std::errc::result_out_of_range) return "out of range";
    return "parse error";
  };

  std::from_chars_result r = [&]{
    if constexpr (is_floating_point_v<T>)
      return std::from_chars(b, e, v, std::chars_format::general);
    return std::from_chars(b, e, v);
  }();

  constexpr std::errc ok{};
  if (r.ec == ok) {
    if (r.ptr != e) return unexpected(string("trailing characters"));
    return v;
  }
  return unexpected(to_msg(r.ec));
}

template<typename T>
class interleaved {
  vector<T> storage;
  size_t frames_   = 0;
  size_t channels_ = 0;

public:
  uint32_t sample_rate = 0;

  interleaved() = default;

  interleaved(uint32_t sr, size_t ch, size_t frames)
  : storage(frames * ch), frames_(frames), channels_(ch), sample_rate(sr)
  { assert(ch > 0); }

  // move-only
  interleaved(const interleaved&) = delete;
  interleaved& operator=(const interleaved&) = delete;

  interleaved(interleaved&&) noexcept = default;
  interleaved& operator=(interleaved&&) noexcept = default;

  [[nodiscard]] size_t   frames()   const noexcept { return frames_; }
  [[nodiscard]] double   duration() const noexcept { return double(frames()) / sample_rate; }
  [[nodiscard]] size_t   channels() const noexcept { return channels_; }
  [[nodiscard]] size_t   samples()  const noexcept { return storage.size(); }
  [[nodiscard]] T*       data()       noexcept     { return storage.data(); }
  [[nodiscard]] const T* data() const noexcept     { return storage.data(); }

  template<typename Elem>
  class frame_view {
    std::span<Elem> row;

  public:
    frame_view(Elem* row, size_t ch) : row(row, ch) {}

    [[nodiscard]] T average() const noexcept {
      return ranges::fold_left(row, T(0), std::plus<T>{}) / row.size();
    }

    [[nodiscard]] T peak() const noexcept {
      return ranges::fold_left(
        row | views::transform([](auto v) { return abs(v); }),
        T(0), [](auto a, auto b) { return max(a, b); }
      );
    }

    frame_view& operator*=(T gain) noexcept
    {
      for (T &sample: row) sample *= gain;
      return *this;
    }
  };

  // 2D element access via multi-arg operator[]
  T& operator[](size_t frame, size_t ch) noexcept {
    assert(frame < frames_ && ch < channels_);
    return storage[frame * channels_ + ch];
  }
  const T& operator[](size_t frame, size_t ch) const noexcept
  {
    assert(frame < frames_ && ch < channels_);
    return storage[frame * channels_ + ch];
  }

  // 1D frame view
  frame_view<T> operator[](size_t frame) noexcept
  {
    assert(frame < frames_);
    return { storage.data() + frame * channels_, channels_ };
  }
  const frame_view<const T> operator[](size_t frame) const noexcept
  {
    assert(frame < frames_);
    return { storage.data() + frame * channels_, channels_ };
  }

  [[nodiscard]] T peak() const noexcept
  {
    return ranges::fold_left(
      storage | views::transform([](T v) { return abs(v); }),
      T(0), [](T a, T b) { return max(a, b); }
    );
  }

  void resize(size_t new_frames)
  {
    storage.resize(new_frames * channels_);
    frames_ = new_frames;
  }

  void clear()
  { resize(0); }
  void shrink_to_fit()
  { storage.shrink_to_fit(); }

  // Scale all samples in-place by gain.
  interleaved &operator*=(T gain) noexcept
  {
    for (T &sample: storage) sample *= gain;
    return *this;
  }
};

template<typename T>
void write_wav(interleaved<T> const &audio, path const &out_path)
{
  if (!in_range<sf_count_t>(audio.frames()))
    throw runtime_error("frame count too large for libsndfile");

  const auto frames = sf_count_t(audio.frames());

  SndfileHandle sf(out_path.string(), SFM_WRITE,
    SF_FORMAT_RF64 | SF_FORMAT_PCM_24,
    int(audio.channels()), int(audio.sample_rate)
  );

  if (sf.error() != SF_ERR_NO_ERROR) throw runtime_error(sf.strError());

  const sf_count_t written = sf.writef(audio.data(), frames);
  if (written != frames)
    throw runtime_error(
      std::format("Short write: wrote {} of {} frames", written, frames)
    );
}

template<floating_point T>
void ensure_headroom(interleaved<T> &audio, T headroom_dB)
{
  const T headroom_linear = dbamp(headroom_dB);
  const T peak = audio.peak();
  if (peak > T(0) && peak > headroom_linear) {
    audio *= headroom_linear / peak;
  }
}

struct ebur128_state_deleter {
  void operator()(ebur128_state *p) const noexcept { if (p) ebur128_destroy(&p); }
};

using ebur128_state_ptr = unique_ptr<ebur128_state, ebur128_state_deleter>;

[[nodiscard]] expected<double, string>
measure_lufs(const interleaved<float> &audio)
{
  ebur128_state_ptr state{
    ebur128_init(audio.channels(), audio.sample_rate, EBUR128_MODE_I)
  };
  if (!state) return unexpected("measure_lufs: ebur128_init failed");

  if (ebur128_add_frames_float(state.get(), audio.data(), audio.frames())
      != EBUR128_SUCCESS)
    return unexpected("measure_lufs: ebur128_add_frames_float failed");

  double lufs = 0.0;
  if (ebur128_loudness_global(state.get(), &lufs) != EBUR128_SUCCESS)
    return unexpected("measure_lufs: ebur128_loudness_global failed");

  return lufs;
}

[[nodiscard]] expected<interleaved<float>, string>
change_tempo(const interleaved<float>& in,
  double from_bpm, double to_bpm, uint32_t to_rate, int src_type
) {
  const size_t channels     = in.channels();
  const size_t in_frames_sz = in.frames();

  // Our invariants: we control all call sites.
  assert(channels > 0);
  assert(in_frames_sz > 0);
  assert(from_bpm > 0.0);
  assert(to_bpm   > 0.0);
  assert(in.sample_rate > 0);
  assert(to_rate        > 0);

  // libsamplerate constraints: still return error on overflow.
  if (!in_range<long>(in_frames_sz))
    return unexpected(
      "Input too large for libsamplerate (frame count exceeds 'long')."
    );

  const long in_frames = static_cast<long>(in_frames_sz);

  // Resampling ratio so that when played at to_rate, tempo becomes to_bpm.
  // Derivation: tempo_out = tempo_in * (to_rate / (ratio * from_rate))
  // -> ratio = (to_rate/from_rate) * (from_bpm/to_bpm)
  const auto ratio = (double(to_rate) / in.sample_rate) * (from_bpm / to_bpm);

  // With valid inputs, ratio must be finite and > 0.
  assert(ratio > 0.0);
  assert(std::isfinite(ratio));

  // Estimate output frames (add 1 for safety).
  const double est_out_frames_d = ceil(static_cast<double>(in_frames) * ratio) + 1.0;
  const auto   est_out_frames_sz = static_cast<size_t>(est_out_frames_d);

  if (!in_range<long>(est_out_frames_sz))
    return unexpected(
      "Output too large for libsamplerate (frame count exceeds 'long')."
    );

  const long out_frames_est = static_cast<long>(est_out_frames_d);

  // libsamplerate channel constraint: still return error.
  if (!in_range<int>(channels))
    return unexpected(
      "Channel count too large for libsamplerate."
    );
  const auto ch = static_cast<int>(channels);

  interleaved<float> out(to_rate, channels, static_cast<size_t>(out_frames_est));

  SRC_DATA data{
    .data_in       = in.data(),
    .data_out      = out.data(),
    .input_frames  = in_frames,
    .output_frames = out_frames_est,
    .end_of_input  = 1,
    .src_ratio     = ratio
  };

  if (const int err = src_simple(&data, src_type, ch); err != 0)
    return unexpected(src_strerror(err));

  out.resize(data.output_frames_gen);

  return out;
}

// Fade curve for envelopes / crossfades.
enum class fade_curve { Linear, Sine };

// Map a normalized 0..1 parameter to a gain using the chosen curve.
// For fade_curve::Sine we use a sine-shaped equal-power style curve.
[[nodiscard]] inline float apply_fade_curve(fade_curve curve, double x) noexcept
{
  x = clamp(x, 0.0, 1.0);
  switch (curve) {
    case fade_curve::Linear:
      return static_cast<float>(x);

    case fade_curve::Sine: {
      // Equal-power style fade: sin(pi/2 * x)
      return static_cast<float>(std::sin(0.5 * std::numbers::pi_v<double> * x));
    }
  }
  return static_cast<float>(x); // fallback
}

// Piecewise fade: optional fade-in from start->first_cue, unity between [first_cue,last_cue],
// optional fade-out from last_cue->end. Uses a chosen curve (typically Sine = equal-power style).
[[nodiscard]] inline float fade_for_frame(
  size_t frameIndex, size_t total_frames,
  optional<double> first_cue, optional<double> last_cue,
  fade_curve curve = fade_curve::Sine
) noexcept
{
  if (total_frames == 0) return 0.0f;

  // No cues at all: no envelope, just unity
  if (!first_cue && !last_cue) return 1.0f;

  double fc = first_cue.value_or(0.0);
  double lc = last_cue.value_or(static_cast<double>(total_frames));

  if (lc < fc) std::swap(lc, fc);

  const double f = static_cast<double>(frameIndex);

  // Fade-in region: [0, fc]
  if (first_cue && f <= fc) {
    if (fc <= 0.0) return 1.0f; // degenerate: no fade-in
    const double p = f / fc;    // 0..1
    return apply_fade_curve(curve, p);
  }

  // Fade-out region: [lc, total_frames)
  if (last_cue && f >= lc) {
    const double denom = static_cast<double>(total_frames) - lc;
    if (denom <= 1e-12) return 0.0f;  // degenerate: no tail
    const double p = (f - lc) / denom; // 0..1
    // For fade-out, invert the curve: 1 - curve(p)
    return 1.0f - apply_fade_curve(curve, p);
  }

  // Sustain region
  return 1.0f;
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
    const double q = max(0.0, posSrcFrames) / framesPerBeatSrc;
    const double qFloor = floor(q);
    const auto bi = static_cast<uint64_t>(qFloor);
    const double frac = q - qFloor;
    if (abs(frac) < 1e-9) {
      lastBeatIndex = bi - 1; // prime to trigger click at boundary (wraps for beat 0)
    } else {
      lastBeatIndex = bi;
    }
  }

  [[nodiscard]] float process(double posSrcFrames, double framesPerBeatSrc, uint32_t device_rate) {
    if (clickLen == 0) clickLen = max(1, (int)(device_rate / 100)); // ~10ms
    auto beatIndex = static_cast<uint64_t>(
      floor(max(0.0, posSrcFrames) / framesPerBeatSrc)
    );
    if (beatIndex != lastBeatIndex) {
      lastBeatIndex = beatIndex;
      clickSamplesLeft = clickLen;
      clickPhase = 0.f;
      unsigned curBpb = max(1u, bpb.load());
      bool downbeat = (beatIndex % static_cast<uint64_t>(curBpb)) == 0;
      clickAmp = downbeat ? downbeatAmp : beatAmp;
      clickFreqCurHz = downbeat ? clickFreqHzDownbeat : clickFreqHzBeat;
    }
    float click = 0.f;
    if (clickSamplesLeft > 0) {
      float env = (float)clickSamplesLeft / (float)clickLen; // linear decay
      clickPhase += 2.0f * std::numbers::pi_v<float> * clickFreqCurHz / (float)device_rate;
      click = clickAmp * std::sinf(clickPhase) * env;
      --clickSamplesLeft;
    }
    return click;
  }
};

// Track metadata persisted in the DB
struct track_info {
  path filename;
  unsigned beats_per_bar = 4;
  double bpm = 120.0; // required > 0
  double upbeat_beats = 0.0;
  double time_offset_sec = 0.0;
  vector<int> cue_bars; // 1-based bar numbers
  std::set<string> tags; // unique tags
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(track_info,
  filename, beats_per_bar, bpm, upbeat_beats, time_offset_sec, cue_bars, tags
)

// Compute cue frames for a track_info at a given sample rate.
// If bpm_override is not provided, track_info::bpm is used.
[[nodiscard]] vector<double>
cue_frames(track_info const& info,
  uint32_t sr, optional<double> bpm_override = nullopt
) {
  const auto frames_per_beat = 60.0 / bpm_override.value_or(info.bpm) * sr;
  auto bar_to_frame =
    [ shift = info.upbeat_beats * frames_per_beat + info.time_offset_sec * sr
    , frames_per_bar = frames_per_beat * info.beats_per_bar
    ](int bar) { return shift + (bar - 1) * frames_per_bar; };

  return info.cue_bars | views::transform(bar_to_frame)
       | ranges::to<vector<double>>();
}

class Matcher {
  struct Node;
  using Ptr = shared_ptr<const Node>;

  struct Node {
    struct Symbol { string name; };
    struct Not    { Ptr child; };
    struct And    { Ptr lhs; Ptr rhs; };
    struct Or     { Ptr lhs; Ptr rhs; };
    struct Compare {
      enum class Op { LT, LE, GT, GE, EQ };
      Op op;
      double value; // BPM value
    };

    std::variant<Symbol, Not, And, Or, Compare> v;

    explicit Node(Symbol s) : v(std::move(s)) {}
    explicit Node(Not n)     : v(std::move(n)) {}
    explicit Node(And a)     : v(std::move(a)) {}
    explicit Node(Or o)      : v(std::move(o)) {}
    explicit Node(Compare c) : v(std::move(c)) {}
  };

  Ptr root_;
  explicit Matcher(Ptr p) : root_(std::move(p)) {}

  static bool eval_node(const Node& n, const track_info& ti) {
    return std::visit([&](const auto& node) -> bool {
      using T = std::decay_t<decltype(node)>;
      if constexpr (is_same_v<T, Node::Symbol>) {
        return ti.tags.contains(node.name);
      } else if constexpr (is_same_v<T, Node::Not>) {
        return !eval_node(*node.child, ti);
      } else if constexpr (is_same_v<T, Node::And>) {
        if (!eval_node(*node.lhs, ti)) return false; // short-circuit
        return eval_node(*node.rhs, ti);
      } else if constexpr (is_same_v<T, Node::Or>) {
        if (eval_node(*node.lhs, ti)) return true;   // short-circuit
        return eval_node(*node.rhs, ti);
      } else if constexpr (is_same_v<T, Node::Compare>) {
        const double bpm = ti.bpm;
        switch (node.op) {
          case Node::Compare::Op::LT: return bpm <  node.value;
          case Node::Compare::Op::LE: return bpm <= node.value;
          case Node::Compare::Op::GT: return bpm >  node.value;
          case Node::Compare::Op::GE: return bpm >= node.value;
          case Node::Compare::Op::EQ: return bpm == node.value;
        }
        return false;
      }
    }, n.v);
  }

public:
  Matcher() = default; // an empty expr; operator() returns true

  // Create an atomic symbol/tag expression
  static Matcher tag(string name) {
    return Matcher(std::make_shared<Node>(Node::Symbol{std::move(name)}));
  }

  friend Matcher operator~(Matcher const &matcher)
  { return Matcher(std::make_shared<Node>(Node::Not{matcher.root_})); }

  friend Matcher operator&(Matcher const &lhs, Matcher const &rhs)
  { return Matcher(std::make_shared<Node>(Node::And{lhs.root_, rhs.root_})); }

  friend Matcher operator|(Matcher const &lhs, Matcher const &rhs)
  { return Matcher(std::make_shared<Node>(Node::Or{lhs.root_, rhs.root_})); }

  // Evaluate against a set of tags
  bool operator()(const track_info& ti) const {
    if (!root_) return true; // empty expression is vacuously true
    return eval_node(*root_, ti);
  }

  // Parse from a string with operators: ~ (NOT), & (AND), | (OR), and parentheses.
  // Precedence: ~ > & > |
  // Also supports BPM comparisons like ">=140bpm & <150bpm".
  static Matcher parse(string_view input) {
    struct Parser {
      string_view s;
      size_t i = 0;

      void skip_ws() {
        while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
      }

      bool consume(char c) {
        skip_ws();
        if (i < s.size() && s[i] == c) { ++i; return true; }
        return false;
      }

      static bool is_ident_char(char c) {
        // Identifier chars: anything except whitespace and operators/parens
        return !std::isspace(static_cast<unsigned char>(c))
            && c != '|' && c != '&' && c != '~' && c != '(' && c != ')';
      }

      [[noreturn]] void error(string_view msg) const {
        string err("Matcher parse error: ");
        err.append(msg);
        err.append(" at position ");
        err.append(std::to_string(i));
        err.append(" near '");
        const size_t start = i, end = min(i + 10, s.size());
        err.append(std::string(s.substr(start, end - start)));
        err.push_back('\'');
        throw std::invalid_argument(err);
      }

      // Try to parse a comparison operator: <, <=, >, >=, ==.
      bool parse_cmp_op(Node::Compare::Op& op) {
        skip_ws();
        if (i >= s.size()) return false;
        char c = s[i];
        if (c == '<') {
          if (i + 1 < s.size() && s[i + 1] == '=') {
            op = Node::Compare::Op::LE;
            i += 2;
          } else {
            op = Node::Compare::Op::LT;
            ++i;
          }
          return true;
        }
        if (c == '>') {
          if (i + 1 < s.size() && s[i + 1] == '=') {
            op = Node::Compare::Op::GE;
            i += 2;
          } else {
            op = Node::Compare::Op::GT;
            ++i;
          }
          return true;
        }
        if (c == '=') {
          if (i + 1 < s.size() && s[i + 1] == '=') {
            op = Node::Compare::Op::EQ;
            i += 2;
            return true;
          }
        }
        return false;
      }

      // Try to parse a numeric bpm literal: <number>bpm (case-insensitive).
      bool parse_bpm_literal(double& value) {
        skip_ws();
        size_t start_num = i;
        bool seen_digit = false;

        // integer part
        while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) {
          seen_digit = true;
          ++i;
        }
        // optional fractional part
        if (i < s.size() && s[i] == '.') {
          ++i;
          while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) {
            seen_digit = true;
            ++i;
          }
        }
        if (!seen_digit) {
          i = start_num;
          return false;
        }

        string num_str(s.substr(start_num, i - start_num));
        double v{};
        auto res = std::from_chars(num_str.data(), num_str.data() + num_str.size(), v);
        if (res.ec != std::errc{}) {
          i = start_num;
          return false;
        }

        auto match_suffix = [&](string_view suf) {
          if (i + suf.size() > s.size()) return false;
          for (size_t k = 0; k < suf.size(); ++k) {
            char c1 = static_cast<char>(std::tolower(static_cast<unsigned char>(s[i + k])));
            char c2 = suf[k];
            if (c1 != c2) return false;
          }
          i += suf.size();
          return true;
        };

        if (!match_suffix("bpm")) {
          i = start_num;
          return false;
        }

        value = v;
        return true;
      }

      Matcher parse_expr() {
        auto lhs = parse_and();
        skip_ws();
        while (consume('|')) {
          auto rhs = parse_and();
          lhs = lhs | rhs;
          skip_ws();
        }
        return lhs;
      }

      Matcher parse_and() {
        auto lhs = parse_unary();
        skip_ws();
        while (consume('&')) {
          auto rhs = parse_unary();
          lhs = lhs & rhs;
          skip_ws();
        }
        return lhs;
      }

      Matcher parse_unary() {
        skip_ws();
        if (consume('~')) {
          auto child = parse_unary();
          return ~child;
        }
        return parse_primary();
      }

      Matcher parse_primary() {
        skip_ws();
        if (consume('(')) {
          auto e = parse_expr();
          if (!consume(')')) error("expected ')'");
          return e;
        }
        if (i >= s.size()) error("unexpected end of input");

        // Try comparison: <op> <number>bpm
        {
          Node::Compare::Op op;
          size_t save = i;
          if (parse_cmp_op(op)) {
            double bpm_value{};
            if (!parse_bpm_literal(bpm_value)) {
              error("expected <number>bpm after comparison operator");
            }
            Node::Compare cmp{op, bpm_value};
            return Matcher(std::make_shared<Node>(std::move(cmp)));
          }
          i = save;
        }

        // Otherwise, parse identifier/tag
        if (!is_ident_char(s[i])) error("expected identifier or comparison");

        const size_t start = i;
        while (i < s.size() && is_ident_char(s[i])) ++i;
        string name(s.substr(start, i - start));

        // Restrict tags starting with digits: must be all digits
        if (!name.empty() && std::isdigit(static_cast<unsigned char>(name[0]))) {
          bool all_digits = ranges::all_of(name,
            [](unsigned char c){ return std::isdigit(c); }
          );
          if (!all_digits) {
            error("tag names starting with a digit must contain only digits");
          }
        }

        return Matcher::tag(std::move(name));
      }
    };

    Parser p{input};
    auto expr = p.parse_expr();
    p.skip_ws();
    if (p.i != input.size()) {
      throw std::invalid_argument(
          "Matcher parse error: trailing input at position " + std::to_string(p.i));
    }
    return expr;
  }
};

struct track_database {
  std::map<path, track_info> items;

  static path norm(const path& p) {
    return p.lexically_normal();
  }

  [[nodiscard]] const track_info* find(const path& file) const {
    auto it = items.find(norm(file));
    return (it == items.end()) ? nullptr : &it->second;
  }

  void upsert(const track_info& info) {
    items[norm(info.filename)] = info;
  }
};

track_database load_database(const path& dbfile)
{
  std::ifstream in;
  in.exceptions(std::ifstream::failbit | std::ifstream::badbit);
  in.open(dbfile);

  track_database database;
  json j;
  in >> j;

  if (!j.is_object()) throw runtime_error("root is not an object");

  int version = j.value("version", 1);
  if (version != 1) throw runtime_error("Unsupported trackdb version");

  if (!j.contains("tracks") || !j["tracks"].is_array())
    throw runtime_error("missing 'tracks' array");

  for (const auto& jti : j["tracks"]) database.upsert(jti.get<track_info>());

  return database;
}

void save(track_database const &database, const path& dbfile)
{
  auto to_json = [](const track_info &ti) { return json(ti); };
  auto tracks = json::array();
  ranges::move(
    database.items | views::values | views::transform(to_json),
    std::back_inserter(tracks)
  );

  json root = {
    {"version", 1},
    {"tracks", std::move(tracks)}
  };

  std::ofstream out;
  out.exceptions(std::ifstream::failbit | std::ifstream::badbit);
  out.open(dbfile, std::ios::trunc);

  out << root.dump(2); // pretty-print with 2-space indent
}

// Resolve mix track paths to track_info, ensuring they exist and have cues.
[[nodiscard]] vector<track_info>
resolve_mix_tracks(const track_database &database, const vector<path>& files)
{
  vector<track_info> tracks;
  tracks.reserve(files.size());
  for (auto const& file : files) {
    auto* info = database.find(file);
    if (!info || info->cue_bars.empty()) {
      throw runtime_error(
        "Track missing in DB or has no cues: " + file.generic_string()
      );
    }
    tracks.push_back(*info);
  }
  return tracks;
}

[[nodiscard]] vector<path>
match(track_database const &database, Matcher const &matcher)
{
  auto valid = [](const track_info &ti) { return !ti.cue_bars.empty(); };

  return database.items | views::values | views::filter(valid)
       | views::filter(matcher)
       | views::transform(&track_info::filename)
       | ranges::to<vector<path>>();
}

struct player_state {
  std::atomic<bool> playing{false};
  shared_ptr<interleaved<float>> track; // set before play; not swapped while playing
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

player_state g_player;

uint32_t g_device_rate = 44100;
uint32_t g_device_channels = 2;

struct mix_cue {
  unsigned   bar;        // 1-based global bar number in the mix
  path       track;      // which track this cue comes from
  unsigned   local_bar;  // bar number within that track (1-based)
  double     time_sec;   // absolute time in the rendered mix (seconds)
};

vector<mix_cue> g_mix_cues;

unsigned g_mix_bpb = 4;

// RAII wrapper around miniaudio playback device, templated on callback type.
template<class Callback>
class miniplayer {
public:
  miniplayer(uint32_t sample_rate, uint32_t channels, Callback cb)
  : callback_(std::move(cb))
  {
    auto config = ma_device_config_init(ma_device_type_playback);
    config.playback.format   = ma_format_f32;
    config.playback.channels = channels;
    config.sampleRate        = sample_rate;
    config.noPreSilencedOutputBuffer = false;
    config.dataCallback      = &miniplayer::ma_callback_trampoline;
    config.pUserData         = this;

    if (ma_result res = ma_device_init(nullptr, &config, &device_);
        res != MA_SUCCESS) {
      throw runtime_error(
        string("Audio device init failed: ") + ma_result_description(res)
      );
    }
  }

  miniplayer(const miniplayer&) = delete;
  miniplayer& operator=(const miniplayer&) = delete;

  ~miniplayer() {
    ma_device_uninit(&device_);
  }

  void start() {
    if (ma_result res = ma_device_start(&device_); res != MA_SUCCESS) {
      throw runtime_error(
        string("Audio device start failed: ") + ma_result_description(res)
      );
    }
  }

  void stop() noexcept {
    ma_device_stop(&device_); // ignore errors on stop
  }

  [[nodiscard]] uint32_t sample_rate() const noexcept {
    return device_.sampleRate;
  }

  [[nodiscard]] uint32_t channels() const noexcept {
    return device_.playback.channels;
  }

private:
  static void ma_callback_trampoline(ma_device* pDevice,
                                     void* pOutput,
                                     const void* pInput,
                                     ma_uint32 frameCount)
  {
    (void)pInput;
    auto* self = static_cast<miniplayer*>(pDevice->pUserData);
    if (!self) return;

    multichannel<float> out(
      static_cast<float*>(pOutput),
      static_cast<size_t>(frameCount),
      static_cast<size_t>(pDevice->playback.channels)
    );
    self->callback_(out, pDevice->sampleRate);
  }

  ma_device device_{};
  Callback  callback_;
};

[[nodiscard]] expected<interleaved<float>, string>
load_track(const path& file)
{
  SndfileHandle sf(file.string());
  if (sf.error()) {
    return unexpected("Failed to open audio file: " + file.generic_string());
  }

  const sf_count_t frames = sf.frames();
  const int sr = sf.samplerate();
  if (sr <= 0) {
    return unexpected("Invalid sample rate in file: " + file.generic_string());
  }

  interleaved<float> track(
    static_cast<uint32_t>(sr),
    static_cast<size_t>(sf.channels()),
    static_cast<size_t>(frames)
  );

  const sf_count_t read_frames = sf.readf(track.data(), frames);
  if (read_frames < 0) {
    return unexpected(
      "Failed to read audio data from file: " + file.generic_string()
    );
  }
  if (read_frames != frames) {
    track.resize(static_cast<size_t>(read_frames));
  }

  return track;
}

using aubio_tempo_ptr = unique_ptr<aubio_tempo_t, decltype(&del_aubio_tempo)>;
using aubio_onset_ptr = unique_ptr<aubio_onset_t, decltype(&del_aubio_onset)>;
using fvec_ptr        = unique_ptr<fvec_t,        decltype(&del_fvec)>;

template<typename F> void
for_each_mono_chunk(interleaved<float> const &audio, fvec_t *buffer, F &&f)
{
  auto mono = [&audio](size_t frame) { return audio[frame].average(); };
  for (auto frames: views::chunk(views::iota(size_t(0), audio.frames()), buffer->length)) {
    smpl_t *tail = ranges::transform(frames, buffer->data, mono).out;
    std::fill(tail, buffer->data + buffer->length, smpl_t(0));
    f(buffer);
  }
}

[[nodiscard]] float detect_bpm(const interleaved<float>& track)
{
  if (track.sample_rate == 0 || track.channels() == 0 || track.frames() == 0) {
    throw std::invalid_argument("detect_bpm: invalid or empty track");
  }

  const uint_t win_s = 1024;
  const uint_t hop_s = 512;
  const uint_t samplerate = track.sample_rate;

  aubio_tempo_ptr tempo{
    new_aubio_tempo("default", win_s, hop_s, samplerate), &del_aubio_tempo
  };
  if (!tempo) throw runtime_error("aubio: failed to create tempo object");

  fvec_ptr inbuf{ new_fvec(hop_s), &del_fvec };
  fvec_ptr out{ new_fvec(1), &del_fvec };
  if (!inbuf || !out) {
    throw runtime_error("aubio: failed to allocate buffers");
  }

  for_each_mono_chunk(track, inbuf.get(), [&](fvec_t *buffer) {
    aubio_tempo_do(tempo.get(), buffer, out.get());
  });

  return aubio_tempo_get_bpm(tempo.get());
}

void apply_two_pass_limiter_db(interleaved<float>& buf,
                               float ceiling_dB = -1.0f,
                               float max_attack_db_per_s = 200.0f,
                               float max_release_db_per_s = 40.0f)
{
  const uint32_t sr = buf.sample_rate;
  const size_t frames = buf.frames();
  if (sr == 0 || frames == 0) return;

  assert(max_attack_db_per_s > 0.0f);
  assert(max_release_db_per_s > 0.0f);

  // 1) Required attenuation (dB) to meet ceiling at each frame (computed on demand)
  auto required_att_dB = [&](size_t f) -> float {
    float pk = buf[f].peak();
    // attenuation needed in dB (>= 0)
    return (pk > 0.f) ? max(0.f, ampdb(pk) - ceiling_dB) : 0.f;
  };

  // 2) Backward pass: limit how fast attenuation may increase (attack slope)
  const float attack_step  = max_attack_db_per_s / static_cast<float>(sr);
  vector<float> att(frames, 0.f);
  att[frames - 1] = required_att_dB(frames - 1);
  for (size_t i = frames - 1; i-- > 0; ) {
    att[i] = max(required_att_dB(i), att[i + 1] - attack_step);
  }

  // 3) Forward pass: limit how fast attenuation may decrease (release slope)
  const float release_step = max_release_db_per_s / static_cast<float>(sr);
  for (size_t i = 1; i < frames; ++i) {
    att[i] = max(att[i], att[i - 1] - release_step);
  }

  // 4) Apply gain: g = dbamp(-att_dB) clamped to [0,1]
  for (size_t f = 0; f < frames; ++f) {
    buf[f] *= clamp(dbamp(-att[f]), 0.0f, 1.0f);
  }
}

// Transient detection method selector for autogrid
enum class TransientMethod {
  Beats,   // aubio_tempo (default)
  Onsets   // aubio_onset
};

// Onset detection for beat-grid fitting
[[nodiscard]] vector<double>
detect_onsets(const interleaved<float>& track)
{
  if (track.sample_rate == 0 || track.channels() == 0 || track.frames() == 0) {
    throw std::invalid_argument("detect_onsets: invalid or empty track");
  }

  const uint_t win_s = 1024;
  const uint_t hop_s = 512;
  const auto samplerate = static_cast<uint_t>(track.sample_rate);

  using fvec_ptr  = unique_ptr<fvec_t,        decltype(&del_fvec)>;

  aubio_onset_ptr onset{
    new_aubio_onset("hfc", win_s, hop_s, samplerate), &del_aubio_onset
  };
  if (!onset) throw runtime_error("aubio: failed to create onset object");

  fvec_ptr inbuf{ new_fvec(hop_s), &del_fvec };
  fvec_ptr outbuf{ new_fvec(1), &del_fvec };
  if (!inbuf || !outbuf) {
    throw runtime_error("aubio: failed to allocate onset buffers");
  }

  vector<double> result;

  for_each_mono_chunk(track, inbuf.get(), [&](fvec_t *buffer) {
    aubio_onset_do(onset.get(), buffer, outbuf.get());

    // onset detected in this hop?
    if (fvec_get_sample(outbuf.get(), 0) != smpl_t(0)) {
      const auto t_sec = double(aubio_onset_get_last_s(onset.get()));
      if (t_sec >= 0.0 && t_sec <= track.duration()) {
        result.push_back(t_sec);
      }
    }
  });

  std::sort(result.begin(), result.end());
  result.erase(std::unique(result.begin(), result.end()),
               result.end());
  return result;
}

// Beat detection using aubio_tempo (aubiotrack-style beat tracker).
[[nodiscard]] vector<double>
detect_beats(const interleaved<float>& track)
{
  const uint_t win_s = 1024;
  const uint_t hop_s = 512;
  const auto samplerate = static_cast<uint_t>(track.sample_rate);

  aubio_tempo_ptr tempo{
    new_aubio_tempo("default", win_s, hop_s, samplerate), &del_aubio_tempo
  };
  if (!tempo) throw runtime_error("aubio: failed to create tempo object");

  fvec_ptr tempo_out{ new_fvec(2), &del_fvec };
  fvec_ptr inbuf    { new_fvec(hop_s), &del_fvec };
  if (!inbuf || !tempo_out) {
    throw runtime_error("aubio: failed to allocate tempo buffers");
  }

  vector<double> result;

  for_each_mono_chunk(track, inbuf.get(), [&](fvec_t *buffer) {
    aubio_tempo_do(tempo.get(), inbuf.get(), tempo_out.get());

    // Beat detected in this hop?
    if (fvec_get_sample(tempo_out.get(), 0) != smpl_t(0)) {
      const auto t_sec = double(aubio_tempo_get_last_s(tempo.get()));
      if (t_sec >= 0.0 && t_sec <= track.duration()) {
        result.push_back(t_sec);
      }
    }
  });

  std::sort(result.begin(), result.end());
  result.erase(
    std::unique(result.begin(), result.end()),
    result.end()
  );

  return result;
}

struct BeatGridMatch {
  // For regression
  vector<double> beat_indices; // k
  vector<double> onset_times;  // t_k in seconds
};

[[nodiscard]] BeatGridMatch
match_beats(const track_info& ti, const interleaved<float>& track,
  TransientMethod method, double window_sec = 0.05
) {
  BeatGridMatch out;
  if (ti.cue_bars.size() < 2) return out; // need at least first & last cue bar

  const uint32_t sr = track.sample_rate;
  if (sr == 0 || track.frames() == 0) return out;

  const double bpm = ti.bpm;
  if (bpm <= 0.0) return out;

  const int first_bar = ti.cue_bars.front();
  const int last_bar  = ti.cue_bars.back();
  if (first_bar >= last_bar) return out;

  // Detect transients according to method (exceptions propagate to caller)
  vector<double> onsetTimes = (method == TransientMethod::Beats)
    ? detect_beats(track) : detect_onsets(track);

  // Sanity check: enough transients overall
  if (onsetTimes.size() < 4) {
    return out;
  }

  const double secondsPerBeat = 60.0 / bpm;
  const double shiftSec =
      ti.upbeat_beats * secondsPerBeat
    + ti.time_offset_sec;

  const double window = window_sec;

  // Global beat index k: 0 at start of first cue bar
  const double beats_before_first_bar =
      ti.upbeat_beats
    + (double)(first_bar - 1) * (double)ti.beats_per_bar;

  const auto duration = track.duration();

  for (int bar = first_bar; bar < last_bar; ++bar) {
    for (int b = 0; b < (int)ti.beats_per_bar; ++b) {
      const double globalBeatIndex =
          (double)b
        + (double)(bar - first_bar) * (double)ti.beats_per_bar;

      const double beatIndexFromZero =
          beats_before_first_bar + globalBeatIndex;

      const double gridTime =
          shiftSec + beatIndexFromZero * secondsPerBeat;

      if (gridTime < 0.0 || gridTime >= duration) {
        continue;
      }

      const double lo = gridTime - window;
      const double hi = gridTime + window;

      auto it = std::lower_bound(
        onsetTimes.begin(), onsetTimes.end(),
        max(0.0, lo)
      );

      double bestTime = -1.0;
      double bestAbs  = std::numeric_limits<double>::infinity();

      for (; it != onsetTimes.end() && *it <= hi; ++it) {
        double d = *it - gridTime;
        double a = abs(d);
        if (a < bestAbs) {
          bestAbs = a;
          bestTime = *it;
        }
      }

      if (bestTime >= 0.0) {
        out.beat_indices.push_back(globalBeatIndex);
        out.onset_times.push_back(bestTime);
      }
    }
  }

  return out;
}

struct GridFitResult {
  double offset_sec = 0.0;  // time of beat 0 (seconds)
  double beat_sec   = 0.0;  // seconds per beat
  size_t n          = 0;
  double R2         = 0.0;  // coefficient of determination
};

[[nodiscard]] GridFitResult
fit_grid(const BeatGridMatch& m)
{
  GridFitResult r{};
  const size_t n = m.beat_indices.size();
  if (n < 2 || m.onset_times.size() != n) return r;

  auto [A, B, R2] = simple_ordinary_least_squares_with_R_squared(
    m.beat_indices, m.onset_times
  );

  if (!(B > 0.0)) return r;

  r.offset_sec = A;
  r.beat_sec   = B;
  r.n          = n;
  r.R2         = R2;
  return r;
}

struct BPMOffsetCorrection {
  double new_bpm = 0.0;
  double new_time_offset_sec = 0.0;
};

[[nodiscard]] BPMOffsetCorrection
compute_bpm_offset_correction(const track_info& ti,
                              const GridFitResult& fit)
{
  BPMOffsetCorrection c{};
  if (fit.n < 2 || fit.beat_sec <= 0.0) return c;

  const double secondsPerBeat = fit.beat_sec;
  const double bpm = 60.0 / secondsPerBeat;

  const double A = fit.offset_sec; // time (sec) of beat 0 (start of first cue bar)

  const int first_bar = ti.cue_bars.front();
  const double beats_before_first_bar =
      ti.upbeat_beats
    + (double)(first_bar - 1) * (double)ti.beats_per_bar;

  const double shiftSec =
      A - beats_before_first_bar * secondsPerBeat;

  const double time_offset_sec =
      shiftSec - ti.upbeat_beats * secondsPerBeat;

  c.new_bpm = bpm;
  c.new_time_offset_sec = time_offset_sec;
  return c;
}

// Compute default mix BPM as mean of track BPMs for the given tracks.
[[nodiscard]] double compute_default_mix_bpm(
  const vector<track_info>& tracks
) {
  if (tracks.empty()) {
    throw runtime_error("No tracks to compute default BPM.");
  }
  const auto bpms = tracks | views::transform(&track_info::bpm);
  return ranges::fold_left(bpms, 0.0, std::plus<double>{}) / tracks.size();
}

[[nodiscard]] double mix_bpm(const track_database& database,
                            const vector<path>& mix_tracks,
                            optional<double> forced_bpm)
{
  if (forced_bpm) return *forced_bpm;
  auto tracks = resolve_mix_tracks(database, mix_tracks);
  return compute_default_mix_bpm(tracks);
}

struct MixResult {
  interleaved<float> audio;
  double bpm = 0.0;
  unsigned bpb = 4;
  vector<mix_cue> cues;
};

// Build a rendered mix as a single Track at given sample_rate/channels.
// Aligns last cue of A to first cue of B. Applies fade-in from start->first cue,
// unity between cues, fade-out from last cue->end. Accumulates global cue frames.
// 'bpm' is the mix BPM to use (no defaulting inside).
[[nodiscard]] MixResult build_mix(
  const vector<track_info>& tracks,
  double bpm,
  uint32_t sample_rate,
  uint32_t channels,
  int src_type = SRC_LINEAR
) {
  MixResult result{};
  if (tracks.empty()) return result;

  result.bpm = bpm;
  result.bpb = tracks.front().beats_per_bar;

  if (!in_range<int>(channels))
    throw std::invalid_argument("Device channel count not representable as int");
  const int outCh = static_cast<int>(channels);
  const uint32_t out_rate = sample_rate;
  const double fpb = static_cast<double>(out_rate) * 60.0 / bpm;

  struct Item {
    track_info info;
    interleaved<float> audio;
    double first_cue;
    double last_cue;
    double lufs;
    double offset = 0.0;
  };
  vector<expected<Item, string>> items_exp(tracks.size());

  // Parallel per-track processing
  std::transform(std::execution::par, tracks.begin(), tracks.end(), items_exp.begin(),
    [&](track_info const& info) -> expected<Item, string> {
      return load_track(info.filename).and_then(
        [&](interleaved<float> audio) {
          ensure_headroom(audio, -2.0f);
          return change_tempo(audio, info.bpm, bpm, out_rate, src_type);
        }
      ).and_then(
        [&](interleaved<float> audio) -> expected<Item, string> {
          size_t frames = audio.frames();

          auto cueFs = cue_frames(info, out_rate, bpm);
          double first_cue = 0.0;
          double last_cue  = 0.0;

          if (!cueFs.empty()) {
            first_cue = cueFs.front();
            last_cue  = cueFs.back();
          }

          if (frames == 0) {
            first_cue = last_cue = 0.0;
          } else {
            first_cue = clamp(first_cue, 0.0, static_cast<double>(frames - 1));
            last_cue  = clamp(last_cue,  0.0, static_cast<double>(frames - 1));
          }

          return measure_lufs(audio).and_then(
            [&](double lufs) -> expected<Item, string> {
              return Item{
                info, std::move(audio), first_cue, last_cue, lufs
              };
            }
          );
        }
      ).transform_error(
        [&](string error_msg) {
          return info.filename.generic_string() + ": " + error_msg;
        }
      );
    }
  );

  vector<Item> items;
  for (auto &item: items_exp) {
    if (!item) throw runtime_error(item.error());
    items.push_back(std::move(*item));
  }

  // Target LUFS = mean of track LUFS
  const auto target_lufs = ranges::fold_left(
    items | views::transform(&Item::lufs), 0.0, std::plus<double>{}
  ) / items.size();

  // Offsets: align last cue of A with first cue of B
  if (!items.empty()) {
    items[0].offset = -items[0].first_cue;
    for (size_t i = 1; i < items.size(); ++i) {
      items[i].offset = items[i-1].offset + items[i-1].last_cue - items[i].first_cue;
    }
    double minOff = 0.0;
    for (auto& it : items) minOff = min(minOff, it.offset);
    if (minOff < 0.0) for (auto& it : items) it.offset -= minOff;
  }

  // Determine total frames
  size_t total_frames = 0;
  for (auto& it : items) {
    const auto offsetFrames = static_cast<size_t>(ceil(it.offset));
    total_frames = max(total_frames, offsetFrames + it.audio.frames());
  }

  result.audio = interleaved<float>(
    out_rate, static_cast<size_t>(outCh), total_frames
  );

  // Mix down
  const auto outChS = static_cast<size_t>(outCh);
  for (size_t idx = 0; idx < items.size(); ++idx) {
    auto &it = items[idx];
    const size_t inChS = it.audio.channels();
    const auto gain_lin = dbamp(clamp(target_lufs - it.lufs, -12.0, 6.0));

    const bool is_first = (idx == 0);
    const bool is_last  = (idx + 1 == items.size());

    // Decide which fades to apply for this track
    optional<double> fade_in_cue;
    optional<double> fade_out_cue;

    if (!is_first) fade_in_cue  = it.first_cue; // no fade-in on first track
    if (!is_last)  fade_out_cue = it.last_cue;  // no fade-out on last track

    for (size_t f = 0; f < it.audio.frames(); ++f) {
      double absF = it.offset + static_cast<double>(f);
      if (absF < 0.0) continue;
      auto outF = static_cast<size_t>(absF);
      if (outF >= total_frames) break;
      const float fade = fade_for_frame(
        f, it.audio.frames(), fade_in_cue, fade_out_cue, fade_curve::Sine
      );
      const float a = fade * gain_lin;
      if (a <= 0.0f) continue;

      for (size_t ch = 0; ch < outChS; ++ch) {
        const size_t sC = ch % inChS;
        result.audio[outF, ch] += a * it.audio[f, sC];
      }
    }

    it.audio.clear();
    it.audio.shrink_to_fit();
  }

  apply_two_pass_limiter_db(result.audio, -1.0f, 200.0f, 40.0f);

  // Build cues
  result.cues.clear();
  for (auto& it : items) {
    auto cueFs = cue_frames(it.info, out_rate, bpm);
    for (size_t idx2 = 0; idx2 < cueFs.size(); ++idx2) {
      unsigned local_bar = it.info.cue_bars[idx2];
      double mix_frame = it.offset + cueFs[idx2];

      unsigned beats_from_zero = round(mix_frame / fpb);
      unsigned bar = (beats_from_zero / result.bpb) + 1;

      double time_sec = mix_frame / static_cast<double>(out_rate);

      result.cues.push_back(mix_cue{bar, it.info.filename, local_bar, time_sec});
    }
  }

  std::stable_sort(result.cues.begin(), result.cues.end(),
                   [](const mix_cue& a, const mix_cue& b) {
                     return a.bar < b.bar;
                   });

  {
    vector<mix_cue> deduped;
    deduped.reserve(result.cues.size());

    size_t i = 0;
    while (i < result.cues.size()) {
      size_t j = i + 1;
      while (j < result.cues.size() && result.cues[j].bar == result.cues[i].bar) {
        ++j;
      }
      deduped.push_back(result.cues[j - 1]);
      i = j;
    }

    result.cues.swap(deduped);
  }

  return result;
}

void play(player_state &player, multichannel<float> output, uint32_t device_rate)
{
  if (player.track) {
    auto &track = *player.track;
    const auto bpm = max(1.0, player.metro.bpm.load());
    const float gainLin = dbamp(player.trackGainDB.load());
    const size_t srcCh = track.channels();
    const size_t totalSrcFrames = track.frames();
    if (totalSrcFrames == 0) return;


    const double framesPerBeatSrc = (double)track.sample_rate * 60.0 / (double)bpm;
    const double incrSrcPerOut = (double)track.sample_rate / (double)device_rate;
    const double shiftSrc = player.upbeatBeats.load() * framesPerBeatSrc
                            + player.timeOffsetSec.load() * (double)track.sample_rate;

    double pos = player.srcPos;

    for (size_t i = 0; i < output.extent(0); ++i) {
      // Quantized seek: apply pending seek at the next bar boundary
      if (player.seekPending.load()) {
        unsigned bpbNow = max(1u, player.metro.bpb.load());
        const double adjNow  = max(0.0, pos - shiftSrc);
        const double adjNext = max(0.0, pos + incrSrcPerOut - shiftSrc);
        auto beatNow  = (uint64_t)floor(adjNow / framesPerBeatSrc);
        auto beatNext = (uint64_t)floor(adjNext / framesPerBeatSrc);
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
      auto i0 = (size_t)pos;
      double frac = pos - (double)i0;
      size_t i1 = min(i0 + 1, totalSrcFrames - 1);

      float click = player.metro.process(pos - shiftSrc, framesPerBeatSrc, device_rate);

      // Write each output channel from the corresponding source channel (wrap if more outs)
      for (size_t ch = 0; ch < output.extent(1); ++ch) {
        size_t srcC = ch % srcCh;
        float s0 = track[i0, srcC];
        float s1 = track[i1, srcC];
        float smp = std::lerp(s0, s1, static_cast<float>(frac));
        float mix = (smp * gainLin) + click;
        output[i, ch] = mix;
      }

      pos += incrSrcPerOut;
    }

    player.srcPos = pos;
  }
}

// Shell-style tokenizer supporting quotes and backslashes.
// - Whitespace splits args when not inside quotes.
// - Single quotes: literals (no escapes inside).
// - Double quotes: supports backslash escaping of \" and \\ (simple treatment).
[[nodiscard]] vector<string> parse_command_line(const string& s) {
  vector<string> out;
  string cur;
  bool in_single = false, in_double = false, escape = false;

  auto push = [&](){
    out.push_back(cur);
    cur.clear();
  };

  for (char ch: s) {
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
using command_args = std::span<const string>;
using command = std::move_only_function<void(command_args)>;
struct command_entry {
  string help;
  command fn;
};

class REPL {
public:
  void register_command(string name, string help, command fn) {
    commands_.emplace(std::move(name), command_entry{std::move(help), std::move(fn)});
  }

  void run(const char* prompt = "clmix> ") {
    running_ = true;
    while (running_) {
      char* line = readline(prompt);
      if (!line) break; // EOF (Ctrl-D)
      unique_ptr<char, decltype(&std::free)> guard(line, &std::free);

      string input(line);
      if (input.empty()) continue;

      add_history(line);
      auto args = parse_command_line(input);
      if (args.empty()) continue;

      auto it = commands_.find(args[0]);
      if (it == commands_.end()) {
        println(cerr, "Unknown command: {}", args[0]);
        continue;
      }
      try {
        it->second.fn(std::span<const string>{args}.subspan(1));
      } catch (const std::exception& e) {
        println(cerr, "Error: {}", e.what());
      } catch (...) {
        println(cerr, "Unknown error.");
      }
    }
  }

  void stop() { running_ = false; }

  void print_help() const {
    cout << "Commands:\n";
    for (const auto& [name, entry] : commands_) {
      cout << "  " << name;
      if (!entry.help.empty()) cout << " - " << entry.help;
      cout << "\n";
    }
  }

private:
  std::map<string, command_entry> commands_;
  bool running_ = true;
};

void register_volume_command(REPL& repl, string label) {
  repl.register_command(
    "vol",
    "vol [dB] - get/set " + label + " volume in dB (0=unity, negative attenuates)",
    [label](command_args a){
      if (a.empty()) {
        float db = g_player.trackGainDB.load();
        float lin = dbamp(db);
        println(cout, "{} volume: {:.2f} dB (x{:.3f})", label, db, lin);
        return;
      }
      string s = a[0];
      if (s.size() >= 2) {
        string tail = s.substr(s.size() - 2);
        std::transform(tail.begin(), tail.end(), tail.begin(),
                       [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
        if (tail == "db") s.resize(s.size() - 2);
      }
      if (auto v = parse_number<float>(s)) {
        const float db = clamp(*v, -60.f, 12.f);
        g_player.trackGainDB.store(db);
        float lin = dbamp(db);
        println(cout, "{} volume set to {:.2f} dB (x{:.3f})", label, db, lin);
      } else {
        println(cerr, "Invalid dB value: {}", v.error());
      }
    }
  );
}

void run_track_info_shell(track_database& database, const path& f, const path& trackdb_path)
{
  auto t_exp = load_track(f);
  if (!t_exp) {
    println(cerr, "Error: {}", t_exp.error());
    return;
  }
  auto tr = std::make_shared<interleaved<float>>(std::move(*t_exp));

  double guessedBpm = 0.0;

  track_info ti;
  if (auto* existing = database.find(f)) {
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
      println(cerr, "BPM detection failed: {}", e.what());
    }
  }

  println(cout, "Opened {}", f.generic_string());
  if (guessedBpm > 0)
    println(cout, "Guessed BPM: {:.2f}", guessedBpm);
  println(cout, "BPM: {:.2f}", ti.bpm);
  println(cout, "Beats/bar: {}", ti.beats_per_bar);
  println(cout, "Upbeat (beats): {:.3f}", ti.upbeat_beats);
  println(cout, "Time offset (s): {:.3f}", ti.time_offset_sec);

  // Initialize player state for this track (not playing yet)
  g_player.track = tr;
  g_player.srcPos = 0.0;
  g_player.seekTargetFrames.store(0.0);
  g_player.seekPending.store(false);
  // keep existing volume
  g_player.metro.reset_runtime();
  g_player.metro.bpm.store(ti.bpm);
  g_player.metro.bpb.store(max(1u, ti.beats_per_bar));
  g_player.upbeatBeats.store(ti.upbeat_beats);
  g_player.timeOffsetSec.store(ti.time_offset_sec);

  auto print_estimated_bars = [&](){
    auto bpmNow = max(1.0, g_player.metro.bpm.load());
    unsigned bpbNow = max(1u, g_player.metro.bpb.load());
    size_t total_frames = tr->frames();
    double framesPerBeat = (double)tr->sample_rate * 60.0 / (double)bpmNow;
    double beats = (double)total_frames / framesPerBeat;
    double bars = beats / static_cast<double>(bpbNow);
    println(cout, "Estimated bars: {:.2f}", bars);
  };
  print_estimated_bars();

  // Per-track subshell
  REPL sub;
  bool dirty = false;

  sub.register_command("help",
    "List commands",
    [&](command_args) {
      sub.print_help();
    }
  );

  sub.register_command("bpm",
    "bpm [value] - get/set BPM",
    [&](command_args a) {
      if (a.empty()) {
        println(cout, "BPM: {:.2f}", g_player.metro.bpm.load());
        return;
      }
      if (auto v = parse_number<double>(a[0]); v) {
        if (*v <= 0.0) {
          println(cerr, "Invalid BPM: must be > 0");
          return;
        }
        g_player.metro.bpm.store(*v);
        ti.bpm = *v;
        dirty = true;
        print_estimated_bars();
      } else {
        println(cerr, "Invalid BPM: {}", v.error());
      }
    }
  );

  sub.register_command("bpb",
    "bpb [value] - get/set beats per bar",
    [&](command_args a) {
      if (a.empty()) {
        println(cout, "Beats/bar: {}", g_player.metro.bpb.load());
        return;
      }
      if (auto v = parse_number<unsigned>(a[0]); v) {
        if (*v == 0u) {
          println(cerr, "Invalid beats-per-bar: must be > 0");
          return;
        }
        g_player.metro.bpb.store(*v);
        ti.beats_per_bar = *v;
        dirty = true;
        print_estimated_bars();
      } else {
        println(cerr, "Invalid beats-per-bar: {}", v.error());
      }
    }
  );

  sub.register_command("upbeat",
    "upbeat [beats] - get/set upbeat in beats (can be negative)",
    [&](command_args a) {
      if (a.empty()) {
        println(cout, "Upbeat (beats): {:.3f}", ti.upbeat_beats);
        return;
      }
      if (auto v = parse_number<double>(a[0]); v) {
        ti.upbeat_beats = *v;
        g_player.upbeatBeats.store(*v);
        dirty = true;
        print_estimated_bars();
      } else {
        println(cerr, "Invalid upbeat value: {}", v.error());
      }
    }
  );

  sub.register_command("offset",
    "offset [seconds] - get/set time offset in seconds (can be negative)",
    [&](command_args a) {
      if (a.empty()) {
        println(cout, "Time offset (s): {:.3f}", ti.time_offset_sec);
        return;
      }
      if (auto v = parse_number<double>(a[0]); v) {
        ti.time_offset_sec = *v;
        g_player.timeOffsetSec.store(*v);
        dirty = true;
        print_estimated_bars();
      } else {
        println(cerr, "Invalid time offset: {}", v.error());
      }
    }
  );

  sub.register_command("cue",
    "cue <bar> - add a cue at given bar (1-based)",
    [&](command_args a) {
      if (a.size() != 1) {
        println(cerr, "Usage: cue <bar>");
        return;
      }
      if (auto bar = parse_number<int>(a[0]); bar) {
        if (*bar <= 0) {
          println(cerr, "Invalid bar: must be > 0");
          return;
        }
        if (std::find(ti.cue_bars.begin(), ti.cue_bars.end(), *bar) == ti.cue_bars.end()) {
          ti.cue_bars.push_back(*bar);
          std::sort(ti.cue_bars.begin(), ti.cue_bars.end());
          dirty = true;
        }
        if (ti.cue_bars.empty()) {
          println(cout, "(no cues)");
        } else {
          cout << "Cues: ";
          for (size_t i = 0; i < ti.cue_bars.size(); ++i) {
            if (i) cout << ',';
            cout << ti.cue_bars[i];
          }
          cout << "\n";
        }
      } else {
        println(cerr, "Invalid bar: {}", bar.error());
      }
    }
  );

  sub.register_command("uncue",
    "uncue <bar> - remove a cue",
    [&](command_args a) {
      if (a.size() != 1) {
        println(cerr, "Usage: uncue <bar>");
        return;
      }
      if (auto bar = parse_number<int>(a[0]); bar) {
        if (std::erase(ti.cue_bars, *bar) > 0) {
          dirty = true;
        }
      } else {
        println(cerr, "Invalid bar: {}", bar.error());
      }
    }
  );

  sub.register_command("cues",
    "List cue bars",
    [&](command_args) {
      if (ti.cue_bars.empty()) {
        println(cout, "(no cues)");
        return;
      }
      for (size_t i = 0; i < ti.cue_bars.size(); ++i) {
        if (i) cout << ',';
        cout << ti.cue_bars[i];
      }
      cout << "\n";
    }
  );

  sub.register_command("tags",
    "List tags for this track",
    [&](command_args) {
      if (ti.tags.empty()) {
        println(cout, "(no tags)");
        return;
      }
      bool first = true;
      for (const auto& tag : ti.tags) {
        if (!first) cout << ", ";
        cout << tag;
        first = false;
      }
      cout << "\n";
    }
  );

  sub.register_command("tag",
    "tag <name> - add a tag to this track",
    [&](command_args a) {
      if (a.size() != 1) {
        println(cerr, "Usage: tag <name>");
        return;
      }
      string name = a[0];
      auto first = std::find_if_not(name.begin(), name.end(),
                                    [](unsigned char c){ return std::isspace(c); });
      auto last  = std::find_if_not(name.rbegin(), name.rend(),
                                    [](unsigned char c){ return std::isspace(c); }).base();
      if (first >= last) {
        println(cerr, "Empty tag name.");
        return;
      }
      string trimmed(first, last);
      if (trimmed.empty()) {
        println(cerr, "Empty tag name.");
        return;
      }
      ti.tags.insert(trimmed);
      dirty = true;
      cout << "Tags: ";
      bool firstOut = true;
      for (const auto& tag : ti.tags) {
        if (!firstOut) cout << ", ";
        cout << tag;
        firstOut = false;
      }
      cout << "\n";
    }
  );

  sub.register_command("autogrid",
    "autogrid [window_ms] [method] - refine BPM (±1%) and offset from "
    "transients between first and last cue bar; method = beats|onsets (default: beats)",
    [&](command_args a) {
      if (!g_player.track) {
        println(cerr, "No track loaded.");
        return;
      }
      if (ti.cue_bars.size() < 2) {
        println(cerr, "Need at least two cue bars for autogrid.");
        return;
      }

      double window_ms = 50.0;   // +/- 50 ms search window
      TransientMethod method = TransientMethod::Beats; // default

      if (a.size() >= 1) {
        if (auto v = parse_number<double>(a[0]); v && *v > 0.0) {
          window_ms = *v;
        } else {
          println(cerr, "Invalid window_ms: {}", v ? "must be > 0" : v.error());
          return;
        }
      }

      if (a.size() >= 2) {
        string m = a[1];
        ranges::transform(m, m.begin(),
          [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
        if (m == "beats") {
          method = TransientMethod::Beats;
        } else if (m == "onsets") {
          method = TransientMethod::Onsets;
        } else {
          println(cerr, "Invalid method '{}'; expected 'beats' or 'onsets'.", m);
          return;
        }
      }

      try {
        auto matches = match_beats(
          ti, *tr, method, window_ms * 1e-3
        );
        if (matches.beat_indices.size() < 4) {
          println(cerr, "Too few matched transients for reliable fit.");
          return;
        }

        auto fit = fit_grid(matches);
        if (fit.n < 4 || fit.beat_sec <= 0.0) {
          println(cerr, "Grid fit failed.");
          return;
        }

        println(cout,
          "Fit over {} beats: secondsPerBeat = {:.6f}, R^2 = {:.6f}",
          fit.n, fit.beat_sec, fit.R2
        );

        // R^2 sanity check: require a very straight line
        constexpr double min_R2 = 0.995;
        if (fit.R2 < min_R2) {
          println(cerr,
            "R^2 = {:.6f} is below the minimum {:.3f}; "
            "BPM or cues may be wrong. Not applying correction.",
            fit.R2, min_R2
          );
          return;
        }

        auto corr = compute_bpm_offset_correction(ti, fit);
        if (corr.new_bpm <= 0.0) {
          println(cerr, "Computed invalid BPM.");
          return;
        }

        // Enforce ±1% BPM adjustment
        const double old_bpm = ti.bpm;
        const double rel = abs(corr.new_bpm - old_bpm) / old_bpm;
        const double max_rel = 0.01; // 1%
        if (rel > max_rel) {
          println(cerr,
            "Computed BPM {:.4f} differs from current BPM {:.4f} by {:.2f}%, "
            "which exceeds the allowed ±1%. Not applying BPM change.",
            corr.new_bpm, old_bpm, rel * 100.0
          );
          return;
        }

        println(cout,
          "Old BPM: {:.4f}, new BPM: {:.4f}",
          ti.bpm, corr.new_bpm
        );
        println(cout,
          "Old time offset: {:.6f} s, new time offset: {:.6f} s",
          ti.time_offset_sec, corr.new_time_offset_sec
        );

        ti.bpm = corr.new_bpm;
        ti.time_offset_sec = corr.new_time_offset_sec;
        g_player.metro.bpm.store(ti.bpm);
        g_player.timeOffsetSec.store(ti.time_offset_sec);
        dirty = true;
        print_estimated_bars();
      } catch (const std::exception& e) {
        println(cerr, "autogrid failed: {}", e.what());
      }
    }
  );

  register_volume_command(sub, "Track");

  sub.register_command("save",
    "Persist BPM/Beats-per-bar to trackdb",
    [&](command_args) {
      database.upsert(ti);
      save(database, trackdb_path);
      println(cout, "Saved to {}", trackdb_path.generic_string());
      dirty = false;
    }
  );

  sub.register_command("play",
    "Start playback with metronome overlay",
    [&](command_args) {
      if (!g_player.track) {
        println(cerr, "No track loaded.");
        return;
      }
      g_player.seekPending.store(false);
      g_player.playing.store(true);
    }
  );

  sub.register_command("stop",
    "Stop playback",
    [&](command_args) {
      g_player.playing.store(false);
    }
  );

  sub.register_command("seek",
    "seek <bar> - jump to given bar (1-based)",
    [&](command_args a) {
      if (a.size() != 1) {
        println(cerr, "Usage: seek <bar>");
        return;
      }
      if (auto bar1 = parse_number<int>(a[0]); bar1) {
        int bar0 = max(0, *bar1 - 1);
        auto bpmNow = max(1.0, g_player.metro.bpm.load());
        unsigned bpbNow = max(1u, g_player.metro.bpb.load());
        double framesPerBeat = (double)tr->sample_rate * 60.0 / (double)bpmNow;
        double shift = ti.upbeat_beats * framesPerBeat + ti.time_offset_sec * (double)tr->sample_rate;
        double target = shift + (double)bar0 * static_cast<double>(bpbNow) * framesPerBeat;
        size_t total_frames = tr->frames();
        if (target >= (double)total_frames) target = (double)total_frames - 1.0;
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
        println(cerr, "Invalid bar number: {}", bar1.error());
      }
    }
  );

  sub.register_command("exit",
    "Leave track shell",
    [&](command_args) {
      sub.stop();
    }
  );

  sub.run("track-info> ");
  g_player.playing.store(false);
}

// Readline completion for track-info filenames (supports spaces via quoting)
static const track_database* g_db_for_completion = nullptr;

char** clmix_completion(const char* text, int start, int end) {
  (void)end;

  const char* line = rl_line_buffer;

  // Extract the first word (command) from the line up to 'start'
  string_view sv(line, static_cast<size_t>(start));
  auto it = std::find_if_not(sv.begin(), sv.end(),
                             [](unsigned char c){ return std::isspace(c); });
  if (it == sv.end()) {
    return nullptr;
  }
  auto cmd_start = it;
  auto cmd_end = std::find_if(cmd_start, sv.end(),
                              [](unsigned char c){ return std::isspace(c); });
  string cmd(cmd_start, cmd_end);

  // Only provide custom completion for the first argument of "track-info"
  if (cmd != "track-info") {
    return nullptr;
  }

  if (!g_db_for_completion) return nullptr;

  // Generator that iterates over g_db_for_completion->items and returns matching filenames.
  auto generator = [](const char* text, int state) -> char* {
    static vector<string> matches;
    static size_t index;

    if (state == 0) {
      matches.clear();
      index = 0;

      if (!g_db_for_completion) return nullptr;

      string prefix(text);
      for (const auto& [path, ti] : g_db_for_completion->items) {
        (void)ti;
        const string name = path.generic_string();
        if (name.starts_with(prefix)) {
          matches.push_back(name);
        }
      }
    }

    if (index >= matches.size()) {
      return nullptr;
    }

    const string& s = matches[index++];
    char* out = static_cast<char*>(std::malloc(s.size() + 1));
    if (!out) return nullptr;
    std::memcpy(out, s.c_str(), s.size() + 1);
    return out;
  };

  return rl_completion_matches(text, generator);
}

// Helper: rebuild mix into global player from mix_tracks, optionally forcing BPM.
void rebuild_mix_into_player(const track_database& database,
                             const vector<path>& mix_tracks,
                             optional<double> force_bpm = nullopt)
{
  if (mix_tracks.empty()) {
    throw runtime_error("No tracks in mix.");
  }

  auto tracks = resolve_mix_tracks(database, mix_tracks);
  double bpm = force_bpm.value_or(compute_default_mix_bpm(tracks));

  g_player.playing.store(false);
  g_player.track.reset();

  MixResult mix = build_mix(
    tracks, bpm, g_device_rate, g_device_channels
  );

  g_player.track = std::make_shared<interleaved<float>>(std::move(mix.audio));
  g_player.srcPos = 0.0;
  g_player.seekPending.store(false);
  g_player.seekTargetFrames.store(0.0);

  g_mix_bpb = mix.bpb;
  g_mix_cues = std::move(mix.cues);

  g_player.metro.reset_runtime();
  g_player.metro.bpm.store(mix.bpm);
  g_player.metro.bpb.store(max(1u, g_mix_bpb));
  if (!mix_tracks.empty()) {
    if (auto* ti0 = database.find(mix_tracks.front())) {
      g_player.upbeatBeats.store(ti0->upbeat_beats);
      g_player.timeOffsetSec.store(ti0->time_offset_sec);
    } else {
      g_player.upbeatBeats.store(0.0);
      g_player.timeOffsetSec.store(0.0);
    }
  }
}

[[nodiscard]] std::string format_cue_time(double t_sec)
{
  if (t_sec < 0.0) t_sec = 0.0;
  int total_cd_frames = static_cast<int>(std::round(t_sec * 75.0));
  if (total_cd_frames < 0) total_cd_frames = 0;

  int mm = total_cd_frames / (75 * 60);
  int ss = (total_cd_frames / 75) % 60;
  int ff = total_cd_frames % 75;

  return std::format("{:02d}:{:02d}:{:02d}", mm, ss, ff);
}

void export_current_mix(const track_database& database,
                        const vector<path>& mix_tracks,
                        const path& out_path,
                        optional<double> force_bpm = nullopt
)
{
  assert(mix_tracks.empty() == false);

  auto tracks = resolve_mix_tracks(database, mix_tracks);
  double bpm = force_bpm.value_or(compute_default_mix_bpm(tracks));

  // Rebuild a fresh mix with current BPM and tracks, best quality SRC
  MixResult mix = build_mix(
    tracks, bpm, g_device_rate, g_device_channels, SRC_SINC_BEST_QUALITY
  );

  write_wav(mix.audio, out_path);

  // Write CUE sheet next to the WAV
  path cue_path = out_path;
  cue_path.replace_extension(".cue");

  {
    std::ofstream cue(cue_path, std::ios::trunc);
    if (!cue) {
      println(cerr, "Warning: failed to write CUE file {}", cue_path.generic_string());
    } else {
      println(cue, "REM Generated by clmix");
      println(cue, "FILE \"{}\" WAVE", out_path.filename().generic_string());

      // Always create TRACK 01 starting at 00:00:00
      println(cue, "  TRACK 01 AUDIO");
      println(cue, "    INDEX 01 00:00:00");

      // For each internal cuepoint (skip first and last), create a new TRACK
      // starting at that cue's time.
      if (mix.cues.size() > 2) {
        int track_no = 2;
        for (size_t i = 1; i + 1 < mix.cues.size(); ++i, ++track_no) {
          const auto& c = mix.cues[i];
          std::string idx = format_cue_time(c.time_sec);
          println(cue, "  TRACK {:02d} AUDIO", track_no);
          println(cue, "    INDEX 01 {}", idx);
        }
      }
    }
  }

  println(cout, "Exported {} frames ({} Hz, {} ch) to {}",
          mix.audio.frames(), mix.audio.sample_rate, mix.audio.channels(),
          out_path.generic_string());
  println(cout, "Wrote CUE: {}", cue_path.generic_string());
}

// Apply intro/outro constraints to a group of tracks: pick a random intro
// candidate and move it to the front; pick a random outro candidate and move
// it to the end. Operates only on the given group.
void apply_intro_outro_constraints(const track_database& database,
                                  const Matcher& intro_matcher,
                                  const Matcher& outro_matcher,
                                  std::vector<path>& group,
                                  std::mt19937& rng)
{
  if (group.empty()) return;

  // Intro candidates within this group
  std::vector<size_t> intro_indices;
  intro_indices.reserve(group.size());
  for (size_t i = 0; i < group.size(); ++i) {
    if (auto* ti = database.find(group[i])) {
      if (intro_matcher(*ti)) {
        intro_indices.push_back(i);
      }
    }
  }

  if (!intro_indices.empty()) {
    std::uniform_int_distribution<size_t> dist(0, intro_indices.size() - 1);
    size_t pick = intro_indices[dist(rng)];
    if (pick != 0) {
      path tmp = group[pick];
      group.erase(group.begin() + static_cast<std::ptrdiff_t>(pick));
      group.insert(group.begin(), std::move(tmp));
    }
  }

  // Outro candidates (recompute after possible intro move)
  std::vector<size_t> outro_indices;
  outro_indices.reserve(group.size());
  for (size_t i = 0; i < group.size(); ++i) {
    if (auto* ti = database.find(group[i])) {
      if (outro_matcher(*ti)) {
        outro_indices.push_back(i);
      }
    }
  }

  if (!outro_indices.empty()) {
    std::uniform_int_distribution<size_t> dist(0, outro_indices.size() - 1);
    size_t pick = outro_indices[dist(rng)];
    size_t last = group.size() - 1;
    if (pick != last) {
      path tmp = group[pick];
      group.erase(group.begin() + static_cast<std::ptrdiff_t>(pick));
      group.push_back(std::move(tmp));
    }
  }
}

}

int main(int argc, char** argv)
{
  if (argc < 2) {
    cerr << "Usage: clmix <trackdb.txt> [options]\n"
            "Options:\n"
            "  --random <expr>   Build mix from random tracks matching tag/BPM expr (can be given multiple times)\n"
            "  --bpm <value>     Force mix BPM\n"
            "  --export <file>   Render mix to 24-bit WAV and exit\n"
            "  --intro <expr>    Matcher expression for intro tracks (default: tag 'intro')\n"
            "  --outro <expr>    Matcher expression for outro tracks (default: tag 'outro')\n";
    return EXIT_FAILURE;
  }

  const path trackdb_path = argv[1];

  track_database database = load_database(trackdb_path);
  Matcher intro_matcher = Matcher::tag("intro");
  Matcher outro_matcher = Matcher::tag("outro");

  vector<path> mix_tracks;

  // Command-line options (after trackdb_path)
  vector<Matcher>       opt_random_exprs;
  optional<double> forced_mix_bpm;
  optional<path>   opt_export_path;

  // Prepare getopt_long
  int opt;
  int option_index = 0;
  static struct option long_options[] = {
    {"random", required_argument, nullptr, 'r'},
    {"bpm",    required_argument, nullptr, 'b'},
    {"export", required_argument, nullptr, 'e'},
    {"intro",  required_argument, nullptr, 'i'},
    {"outro",  required_argument, nullptr, 'o'},
    {nullptr,  0,                 nullptr,  0 }
  };

  // Start parsing after the trackdb argument
  optind = 2;
  while ((opt = getopt_long(argc, argv, "", long_options, &option_index)) != -1) {
    switch (opt) {
      case 'r': {
        try {
          opt_random_exprs.push_back(Matcher::parse(optarg));
        } catch (const std::exception& e) {
          println(cerr, "Invalid --random expression '{}': {}",
                  optarg, e.what());
          return EXIT_FAILURE;
        }
        break;
      }
      case 'b': {
        auto v = parse_number<double>(optarg);
        if (!v || *v <= 0.0) {
          println(cerr, "Invalid --bpm value: {}",
                  v ? "must be > 0" : v.error());
          return EXIT_FAILURE;
        }
        forced_mix_bpm = *v;
        break;
      }
      case 'e':
        opt_export_path = path(optarg);
        break;
      case 'i': {
        try {
          intro_matcher = Matcher::parse(optarg);
        } catch (const std::exception& e) {
          println(cerr, "Invalid --intro expression '{}': {}",
                  optarg, e.what());
          return EXIT_FAILURE;
        }
        break;
      }
      case 'o': {
        try {
          outro_matcher = Matcher::parse(optarg);
        } catch (const std::exception& e) {
          println(cerr, "Invalid --outro expression '{}': {}",
                  optarg, e.what());
          return EXIT_FAILURE;
        }
        break;
      }
      default:
        return EXIT_FAILURE;
    }
  }

  // If any --random was given, build mix_tracks from DB using one or more Matchers
  if (!opt_random_exprs.empty()) {
    if (database.items.empty()) {
      println(cerr, "Track DB is empty.");
      return 1;
    }

    std::mt19937 rng(std::random_device{}());
    mix_tracks.clear();

    for (const auto& matcher : opt_random_exprs) {
      auto group = match(database, matcher);

      if (group.empty()) {
        println(cerr, "No tracks with cues matching one of the --random expressions.");
        return EXIT_FAILURE;
      }

      std::shuffle(group.begin(), group.end(), rng);
      apply_intro_outro_constraints(database, intro_matcher, outro_matcher, group, rng);
      ranges::move(group, std::back_inserter(mix_tracks));
    }
  }

  // Non-interactive export mode
  if (opt_export_path) {
    if (mix_tracks.empty()) {
      println(cerr, "No tracks in mix.");
      return EXIT_FAILURE;
    }
    try {
      export_current_mix(database, mix_tracks, *opt_export_path, forced_mix_bpm);
    } catch (std::exception &e) {
      println(cerr, "{}", e.what());
      return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
  }

  // Interactive mode from here on: needs audio device and REPL.

  try {
    auto device = miniplayer(44100, 2,
      [](multichannel<float> output, uint32_t device_rate) {
        if (g_player.playing.load()) play(g_player, output, device_rate);
      }
    );

    device.start();

    g_device_rate = device.sample_rate();
    g_device_channels = device.channels();

    // If we have any mix tracks (from --random or previous DB state) and/or a forced BPM,
    // prebuild the mix into the player for interactive use.
    if (!mix_tracks.empty()) {
      try {
        rebuild_mix_into_player(database, mix_tracks, forced_mix_bpm);
      } catch (const std::exception& e) {
        println(cerr, "Failed to build initial mix: {}", e.what());
        return EXIT_FAILURE;
      }
    } else if (forced_mix_bpm) {
      println(cerr, "Warning: --bpm specified but no tracks in mix.");
    }

    // Set up readline completion for track-info filenames (with quoting)
    g_db_for_completion = &database;
    rl_attempted_completion_function = clmix_completion;
    // Reasonable shell-like word breaks; Readline will respect quotes when completing.
    rl_basic_word_break_characters = const_cast<char*>(" \t\n\"'`@$><=;|&{(");

    REPL repl;

    repl.register_command("help",
      "List commands",
      [&](command_args) {
        repl.print_help();
      }
    );
    repl.register_command("exit",
      "Exit program",
      [&](command_args) {
        repl.stop();
      }
    );
    repl.register_command("quit",
      "Alias for exit",
      [&](command_args) {
        repl.stop();
      }
    );
    repl.register_command("track-info",
      "track-info <file> - open per-track shell",
      [&](command_args args) {
        if (args.size() != 1) {
          println(cerr, "Usage: track-info <file>");
          return;
        }
        run_track_info_shell(database, args[0], trackdb_path);
      }
    );

    register_volume_command(repl, "Mix");

    // Mix commands
    repl.register_command("add",
      "add <file> - add track to mix (opens track-info if not in DB)",
      [&](command_args a) {
        if (a.size() != 1) {
          println(cerr, "Usage: add <file>");
          return;
        }
        path f = a[0];
        if (!database.find(f)) {
          run_track_info_shell(database, f, trackdb_path);
        }
        if (!database.find(f)) {
          println(cerr, "Track still not in DB. Aborting.");
          return;
        }
        mix_tracks.push_back(f);
        try {
          rebuild_mix_into_player(database, mix_tracks, forced_mix_bpm);
          cout << "Added. Mix size: " << mix_tracks.size()
                    << ", BPM: " << mix_bpm(database, mix_tracks, forced_mix_bpm)
                    << ", BPB: " << g_mix_bpb << "\n";
        } catch (const std::exception& e) {
          println(cerr, "Failed to build mix: {}", e.what());
        }
      }
    );

    repl.register_command("move",
      "move <from> <to> - move track at index <from> to position <to> in mix (1-based)",
      [&](command_args a) {
        if (a.size() != 2) {
          println(cerr, "Usage: move <from> <to>");
          return;
        }
        if (mix_tracks.empty()) {
          println(cerr, "No tracks in mix.");
          return;
        }

        auto fromIdx = parse_number<int>(a[0]);
        auto toIdx   = parse_number<int>(a[1]);
        if (!fromIdx) {
          println(cerr, "Invalid <from> index: {}", fromIdx.error());
          return;
        }
        if (!toIdx) {
          println(cerr, "Invalid <to> index: {}", toIdx.error());
          return;
        }

        int n = static_cast<int>(mix_tracks.size());
        int from = *fromIdx;
        int to   = *toIdx;

        if (from < 1 || from > n || to < 1 || to > n) {
          println(cerr, "Indices must be between 1 and {}.", n);
          return;
        }

        // Convert to 0-based
        size_t fromPos = static_cast<size_t>(from - 1);
        size_t toPos   = static_cast<size_t>(to   - 1);

        if (fromPos == toPos) {
          println(cout, "No change (from == to).");
          return;
        }

        // Move element: erase+insert
        path tmp = mix_tracks[fromPos];
        mix_tracks.erase(mix_tracks.begin() + static_cast<std::ptrdiff_t>(fromPos));

        // After erase, if we removed an earlier element, the target index shifts left by 1
        if (fromPos < toPos) {
          --toPos;
        }

        mix_tracks.insert(mix_tracks.begin() + static_cast<std::ptrdiff_t>(toPos), std::move(tmp));

        try {
          rebuild_mix_into_player(database, mix_tracks, forced_mix_bpm);
          println(cout, "Moved track {} -> {}. Mix size: {}, BPM: {}, BPB: {}",
                  from, to, mix_tracks.size(),
                  mix_bpm(database, mix_tracks, forced_mix_bpm),
                  g_mix_bpb);
        } catch (const std::exception& e) {
          println(cerr, "Failed to rebuild mix: {}", e.what());
        }
      }
    );

    repl.register_command("bpm",
      "bpm [value] - show/set mix BPM (recomputes mix)",
      [&](command_args a) {
        if (mix_tracks.empty()) {
          println(cerr, "No tracks in mix.");
          return;
        }
        if (a.empty()) {
          const double bpm = mix_bpm(database, mix_tracks, forced_mix_bpm);
          println(cout, "Mix BPM: {:.2f}{}", bpm, forced_mix_bpm ? " (forced)" : "");
          return;
        }
        if (auto v = parse_number<double>(a[0]); v) {
          if (*v <= 0.0) {
            println(cerr, "Invalid BPM: must be > 0");
            return;
          }
          forced_mix_bpm = *v;
          try {
            rebuild_mix_into_player(database, mix_tracks, forced_mix_bpm);
            println(cout, "Mix BPM set to {:.2f} and recomputed.",
                    mix_bpm(database, mix_tracks, forced_mix_bpm));
          } catch (const std::exception& e) {
            println(cerr, "Failed to rebuild mix: {}", e.what());
          }
        } else {
          println(cerr, "Invalid BPM: {}", v.error());
        }
      }
    );

    repl.register_command("play",
      "Start playback (mix)",
      [&](command_args) {
        if (!g_player.track) {
          if (mix_tracks.empty()) {
            println(cerr, "No tracks in mix.");
            return;
          }
          try {
            rebuild_mix_into_player(database, mix_tracks, forced_mix_bpm);
          } catch (const std::exception& e) {
            println(cerr, "Build mix failed: {}", e.what());
            return;
          }
        }
        g_player.seekPending.store(false);
        g_player.playing.store(true);
      }
    );

    repl.register_command("stop",
      "Stop playback",
      [&](command_args) {
        g_player.playing.store(false);
      }
    );

    repl.register_command("seek",
      "seek <bar> - jump to mix bar (1-based)",
      [&](command_args a) {
        if (a.size() != 1 || !g_player.track) {
          println(cerr, "Usage: seek <bar>");
          return;
        }
        if (auto bar1 = parse_number<int>(a[0]); bar1) {
          int bar0 = max(0, *bar1 - 1);
          const double bpm = mix_bpm(database, mix_tracks, forced_mix_bpm);
          double framesPerBeat = (double)g_player.track->sample_rate * 60.0 / bpm;
          double shift = g_player.upbeatBeats.load() * framesPerBeat
                         + g_player.timeOffsetSec.load() * (double)g_player.track->sample_rate;
          double target = shift + (double)bar0 * (double)g_mix_bpb * framesPerBeat;
          size_t total_frames = g_player.track->frames();
          if (target >= (double)total_frames) target = (double)total_frames - 1.0;
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
          println(cerr, "Invalid bar number: {}", bar1.error());
        }
      }
    );

    repl.register_command("cues",
      "List all cue points in current mix",
      [&](command_args) {
        if (g_mix_cues.empty()) {
          println(cout, "(no cues)");
          return;
        }
        for (const auto& c : g_mix_cues) {
          println(cout, "mix bar {}  |  track: {}  |  track bar {}",
                  c.bar, c.track.filename().stem().generic_string(), c.local_bar);
        }
      }
    );

    repl.register_command("tags",
      "List all tags present in track DB",
      [&](command_args) {
        // Count how many tracks have each tag
        std::map<std::string, size_t> counts;
        for (const auto& [path, ti] : database.items) {
          (void)path;
          for (const auto& tag : ti.tags) {
            ++counts[tag];
          }
        }

        if (counts.empty()) {
          println(cout, "(no tags)");
          return;
        }

        // Move to vector and sort by count (descending), then by tag name
        vector<std::pair<std::string, size_t>> v;
        v.reserve(counts.size());
        for (auto& [tag, cnt] : counts) {
          v.emplace_back(tag, cnt);
        }

        ranges::sort(v, [](auto const& a, auto const& b) {
          if (a.second != b.second) return a.second > b.second; // more tracks first
          return a.first < b.first;                             // tie-break by name
        });

        bool first = true;
        for (auto const& [tag, cnt] : v) {
          if (!first) cout << ", ";
          cout << tag << " (" << cnt << ")";
          first = false;
        }
        cout << "\n";
      }
    );

    repl.register_command("list",
      "list [tag_expr] - list tracks in DB matching tag/bpm expression "
      "(e.g. \">=140bpm & <150bpm | techno\")",
      [&](command_args args) {
        if (database.items.empty()) {
          println(cerr, "Track DB is empty.");
          return;
        }

        // Build matcher: empty args => match everything
        Matcher matcher;
        if (!args.empty()) {
          std::string expr;
          for (size_t i = 0; i < args.size(); ++i) {
            if (i) expr.push_back(' ');
            expr += args[i];
          }
          try {
            matcher = Matcher::parse(expr);
          } catch (const std::exception& e) {
            println(cerr, "Invalid tag expression: {}", e.what());
            return;
          }
        }

        auto matched = database.items | views::values | views::filter(matcher);
        for (const auto& [index, info]: views::enumerate(matched)) {
          cout << std::setw(3) << (index + 1) << ". "
               << info.filename.generic_string()
               << "  |  BPM: " << std::fixed << std::setprecision(2) << info.bpm
               << "  |  Tags: ";
          if (info.tags.empty()) {
            cout << "(none)";
          } else {
            bool first = true;
            for (const auto& tag : info.tags) {
              if (!first) cout << ", ";
              cout << tag;
              first = false;
            }
          }
          cout << "\n";
        }

        if (ranges::empty(matched)) {
          println(cout, "(no tracks matching expression)");
        }
      }
    );

    repl.register_command("random",
      "random [expr1 [expr2 ...]] - build mix from DB; "
      "no expr => all tracks; multiple exprs => append random block per expr",
      [&](command_args args) {
        if (database.items.empty()) {
          println(cerr, "Track DB is empty.");
          return;
        }

        std::mt19937 rng(std::random_device{}());
        mix_tracks.clear();

        auto append_block_for_matcher = [&](const Matcher& matcher,
                                            std::string_view desc) -> bool {
          auto group = match(database, matcher);
          if (group.empty()) {
            if (desc.empty()) {
              println(cerr, "No tracks with cues in DB.");
            } else {
              println(cerr, "No tracks with cues matching expression '{}'.",
                      desc);
            }
            return false;
          }
          std::shuffle(group.begin(), group.end(), rng);
          apply_intro_outro_constraints(database, intro_matcher, outro_matcher, group, rng);
          ranges::move(group, std::back_inserter(mix_tracks));
          return true;
        };

        if (args.empty()) {
          // No matcher: use all tracks with cues
          Matcher match_all; // default-constructed => always true
          if (!append_block_for_matcher(match_all, {})) return;
        } else {
          for (const auto& expr : args) {
            Matcher matcher;
            try {
              matcher = Matcher::parse(expr);
            } catch (const std::exception& e) {
              println(cerr, "Invalid tag expression '{}': {}", expr, e.what());
              mix_tracks.clear();
              return;
            }
            if (!append_block_for_matcher(matcher, expr)) {
              mix_tracks.clear();
              return;
            }
          }
        }

        println(cout, "Track order:");
        for (auto [index, file]: views::enumerate(mix_tracks))
          println(cout, "  {}. {}", index + 1, file.filename().stem().generic_string());

        rebuild_mix_into_player(database, mix_tracks, forced_mix_bpm);
        println(cout, "Random mix created with {} tracks. BPM: {}, BPB: {}",
                mix_tracks.size(),
                mix_bpm(database, mix_tracks, forced_mix_bpm),
                g_mix_bpb);
      }
    );

    repl.register_command("export",
      "export <file.wav> - render mix to 24-bit WAV",
      [&](command_args a) {
        if (a.size() != 1) {
          println(cerr, "Usage: export <file.wav>");
          return;
        }
        if (mix_tracks.empty()) {
          println(cerr, "No tracks in mix.");
          return;
        }

        // Stop playback to avoid concurrent access while rendering/exporting
        g_player.playing.store(false);
        // Release any existing mix from the player to avoid holding two copies in RAM
        g_player.track.reset();

        export_current_mix(database, mix_tracks, a[0], forced_mix_bpm);
      }
    );

    repl.run("clmix> ");
  } catch (const std::exception& e) {
    println(cerr, "Audio init failed: {}", e.what());
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
