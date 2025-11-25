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
#include <execution>
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
#include <string_view>
#include <sstream>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>
#include <cstring> // for std::memcpy

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

#include <nlohmann/json.hpp>

#include <ebur128.h>


namespace {

using nlohmann::json;

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
  std::size_t frames_   = 0;
  std::size_t channels_ = 0;

public:
  uint32_t sample_rate = 0;

  Interleaved() = default;

  Interleaved(uint32_t sr, std::size_t ch, std::size_t frames)
  : storage(frames * ch), frames_(frames), channels_(ch), sample_rate(sr)
  { assert(ch > 0); }

  // move-only
  Interleaved(const Interleaved&) = delete;
  Interleaved& operator=(const Interleaved&) = delete;

  Interleaved(Interleaved&&) noexcept = default;
  Interleaved& operator=(Interleaved&&) noexcept = default;

  [[nodiscard]] std::size_t frames()   const noexcept { return frames_; }
  [[nodiscard]] std::size_t channels() const noexcept { return channels_; }
  [[nodiscard]] std::size_t samples()  const noexcept { return storage.size(); }
  [[nodiscard]] T*       data()       noexcept { return storage.data(); }
  [[nodiscard]] const T* data() const noexcept { return storage.data(); }

  template<class Elem>
  class FrameViewBase {
  protected:
    Elem* row_;
    std::size_t ch_;

    FrameViewBase(Elem* row, std::size_t ch) : row_(row), ch_(ch) {}

  public:
    [[nodiscard]] T peak() const noexcept {
      T p = T(0);
      for (std::size_t c = 0; c < ch_; ++c) {
        T v = row_[c];
        if constexpr (std::is_floating_point_v<T>) {
          if (!std::isfinite(v)) continue;
        }
        v = std::abs(v);
        if (v > p) p = v;
      }
      return p;
    }
  };

  class FrameView : public FrameViewBase<T> {
  public:
    FrameView(T* row, std::size_t ch)
      : FrameViewBase<T>(row, ch) {}

    template<typename U> requires std::is_arithmetic_v<U>
    FrameView& operator*=(U gain) noexcept {
      const T g = static_cast<T>(gain);
      for (std::size_t c = 0; c < this->ch_; ++c)
        this->row_[c] *= g;
      return *this;
    }
  };

  class ConstFrameView : public FrameViewBase<const T> {
  public:
    ConstFrameView(const T* row, std::size_t ch)
    : FrameViewBase<const T>(row, ch) {}
  };

  // 2D element access via multi-arg operator[]
  T& operator[](std::size_t frame, std::size_t ch) noexcept {
    assert(frame < frames_ && ch < channels_);
    return storage[frame * channels_ + ch];
  }
  const T& operator[](std::size_t frame, std::size_t ch) const noexcept {
    assert(frame < frames_ && ch < channels_);
    return storage[frame * channels_ + ch];
  }

  // 1D frame view
  FrameView operator[](std::size_t frame) noexcept {
    assert(frame < frames_);
    return FrameView(storage.data() + frame * channels_, channels_);
  }
  ConstFrameView operator[](std::size_t frame) const noexcept {
    assert(frame < frames_);
    return ConstFrameView(storage.data() + frame * channels_, channels_);
  }

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
    storage.resize(new_frames * channels_);
    frames_ = new_frames;
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

struct ebur128_deleter {
  void operator()(ebur128_state* p) const noexcept {
    if (p) {
      ebur128_destroy(&p);
    }
  }
};

using ebur128_ptr = std::unique_ptr<ebur128_state, ebur128_deleter>;

[[nodiscard]] std::expected<double, std::string>
measure_lufs(const Interleaved<float>& t)
{
  const unsigned int  channels   = t.channels();
  const unsigned long samplerate = t.sample_rate;
  const std::size_t   frames     = t.frames();

  if (channels == 0 || samplerate == 0 || frames == 0) {
    return std::unexpected("measure_lufs: empty or invalid track");
  }

  ebur128_ptr st{ebur128_init(channels, samplerate, EBUR128_MODE_I)};
  if (!st) {
    return std::unexpected("measure_lufs: ebur128_init failed");
  }

  if (int err = ebur128_add_frames_float(st.get(), t.data(), frames);
      err != EBUR128_SUCCESS) {
    return std::unexpected("measure_lufs: ebur128_add_frames_float failed");
  }

  double lufs = 0.0;
  if (int err = ebur128_loudness_global(st.get(), &lufs);
      err != EBUR128_SUCCESS) {
    return std::unexpected("measure_lufs: ebur128_loudness_global failed");
  }

  return lufs;
}

[[nodiscard]] Interleaved<float> change_tempo(
  const Interleaved<float>& in,
  double from_bpm, double to_bpm,
  uint32_t to_rate,
  int converter_type
) {
  const std::size_t channels     = in.channels();
  const std::size_t in_frames_sz = in.frames();

  // Our invariants: we control all call sites.
  assert(channels > 0);
  assert(in_frames_sz > 0);
  assert(from_bpm > 0.0);
  assert(to_bpm   > 0.0);
  assert(in.sample_rate > 0);
  assert(to_rate        > 0);

  // libsamplerate constraints: still throw on overflow.
  if (!std::in_range<long>(in_frames_sz))
    throw std::overflow_error("Input too large for libsamplerate (frame count exceeds 'long').");

  const long in_frames = static_cast<long>(in_frames_sz);

  // Resampling ratio so that when played at to_rate, tempo becomes to_bpm.
  // Derivation: tempo_out = tempo_in * (to_rate / (ratio * from_rate))
  // -> ratio = (to_rate/from_rate) * (from_bpm/to_bpm)
  const double ratio =
      (static_cast<double>(to_rate) / static_cast<double>(in.sample_rate)) *
      (from_bpm / to_bpm);

  // With valid inputs, ratio must be finite and > 0.
  assert(ratio > 0.0);
  assert(std::isfinite(ratio));

  // Estimate output frames (add 1 for safety).
  const double est_out_frames_d = std::ceil(static_cast<double>(in_frames) * ratio) + 1.0;
  const auto   est_out_frames_sz = static_cast<std::size_t>(est_out_frames_d);

  if (!std::in_range<long>(est_out_frames_sz))
    throw std::overflow_error("Output too large for libsamplerate (frame count exceeds 'long').");

  const long out_frames_est = static_cast<long>(est_out_frames_d);

  // libsamplerate channel constraint: still throw.
  if (!std::in_range<int>(channels))
    throw std::invalid_argument("Channel count too large for libsamplerate.");
  const auto ch = static_cast<int>(channels);

  Interleaved<float> out(to_rate, channels, static_cast<std::size_t>(out_frames_est));

  SRC_DATA data{};
  data.data_in       = in.data();
  data.data_out      = out.data();
  data.input_frames  = in_frames;
  data.output_frames = out_frames_est;
  data.end_of_input  = 1;
  data.src_ratio     = ratio;

  if (const int err = src_simple(&data, converter_type, ch); err != 0)
    throw std::runtime_error(src_strerror(err));

  out.resize(static_cast<std::size_t>(data.output_frames_gen));

  return out;
}

// Fade curve for envelopes / crossfades.
enum class FadeCurve {
  Linear,
  Sine,  // sine-shaped equal-power style fade
};

// Map a normalized 0..1 parameter to a gain using the chosen curve.
// For FadeCurve::Sine we use a sine-shaped equal-power style curve.
[[nodiscard]] inline float apply_fade_curve(FadeCurve curve, double x) noexcept {
  x = std::clamp(x, 0.0, 1.0);
  switch (curve) {
    case FadeCurve::Linear:
      return static_cast<float>(x);

    case FadeCurve::Sine: {
      // Equal-power style fade: sin(pi/2 * x)
      return static_cast<float>(std::sin(0.5 * std::numbers::pi_v<double> * x));
    }
  }
  return static_cast<float>(x); // fallback
}

// Piecewise fade: fade-in from start->firstCue, unity between [firstCue,lastCue],
// fade-out from lastCue->end. Uses a chosen curve (typically Sine = equal-power style).
[[nodiscard]] inline float fade_for_frame(
  size_t frameIndex,
  size_t totalFrames,
  double firstCue,
  double lastCue,
  FadeCurve curve = FadeCurve::Sine
) noexcept
{
  if (totalFrames == 0) return 0.0f;
  if (lastCue < firstCue) std::swap(lastCue, firstCue);

  const double f = static_cast<double>(frameIndex);

  // Fade-in region: [0, firstCue]
  if (f <= firstCue) {
    if (firstCue <= 0.0) return 1.0f; // degenerate: no fade-in
    const double p = f / firstCue;    // 0..1
    return apply_fade_curve(curve, p);
  }

  // Fade-out region: [lastCue, totalFrames)
  if (f >= lastCue) {
    const double denom = static_cast<double>(totalFrames) - lastCue;
    if (denom <= 1e-12) return 0.0f;  // degenerate: no tail
    const double p = (f - lastCue) / denom; // 0..1
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
    const double q = std::max(0.0, posSrcFrames) / framesPerBeatSrc;
    const double qFloor = std::floor(q);
    const auto bi = static_cast<uint64_t>(qFloor);
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

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(TrackInfo,
  filename, beats_per_bar, bpm, upbeat_beats, time_offset_sec, cue_bars, tags
)

class Matcher {
  struct Node;
  using Ptr = std::shared_ptr<const Node>;

  struct Node {
    struct Symbol { std::string name; };
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

  static bool eval_node(const Node& n, const TrackInfo& ti) {
    return std::visit([&](const auto& node) -> bool {
      using T = std::decay_t<decltype(node)>;
      if constexpr (std::is_same_v<T, Node::Symbol>) {
        return ti.tags.contains(node.name);
      } else if constexpr (std::is_same_v<T, Node::Not>) {
        return !eval_node(*node.child, ti);
      } else if constexpr (std::is_same_v<T, Node::And>) {
        if (!eval_node(*node.lhs, ti)) return false; // short-circuit
        return eval_node(*node.rhs, ti);
      } else if constexpr (std::is_same_v<T, Node::Or>) {
        if (eval_node(*node.lhs, ti)) return true;   // short-circuit
        return eval_node(*node.rhs, ti);
      } else if constexpr (std::is_same_v<T, Node::Compare>) {
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
  static Matcher tag(std::string name) {
    return Matcher(std::make_shared<Node>(Node::Symbol{std::move(name)}));
  }

  friend Matcher operator~(Matcher const &matcher)
  { return Matcher(std::make_shared<Node>(Node::Not{matcher.root_})); }

  friend Matcher operator&(Matcher const &lhs, Matcher const &rhs)
  { return Matcher(std::make_shared<Node>(Node::And{lhs.root_, rhs.root_})); }

  friend Matcher operator|(Matcher const &lhs, Matcher const &rhs)
  { return Matcher(std::make_shared<Node>(Node::Or{lhs.root_, rhs.root_})); }

  // Evaluate against a set of tags
  bool operator()(const TrackInfo& ti) const {
    if (!root_) return true; // empty expression is vacuously true
    return eval_node(*root_, ti);
  }

  // Parse from a string with operators: ~ (NOT), & (AND), | (OR), and parentheses.
  // Precedence: ~ > & > |
  // Also supports BPM comparisons like ">=140bpm & <150bpm".
  static Matcher parse(std::string_view input) {
    struct Parser {
      std::string_view s;
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

      [[noreturn]] void error(std::string_view msg) const {
        std::string err("Matcher parse error: ");
        err.append(msg);
        err.append(" at position ");
        err.append(std::to_string(i));
        err.append(" near '");
        const size_t start = i, end = std::min(i + 10, s.size());
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

        std::string num_str(s.substr(start_num, i - start_num));
        double v{};
        auto res = std::from_chars(num_str.data(), num_str.data() + num_str.size(), v);
        if (res.ec != std::errc{}) {
          i = start_num;
          return false;
        }

        auto match_suffix = [&](std::string_view suf) {
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
        std::string name(s.substr(start, i - start));

        // Restrict tags starting with digits: must be all digits
        if (!name.empty() && std::isdigit(static_cast<unsigned char>(name[0]))) {
          bool all_digits = std::ranges::all_of(name,
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

  bool load(const std::filesystem::path& dbfile)
  {
    items.clear();

    std::ifstream in(dbfile);
    if (!in.is_open()) {
      // Missing file => treat as empty DB, like load()
      return false;
    }

    json j;
    try {
      in >> j;
    } catch (const std::exception& e) {
      std::println(std::cerr, "Failed to parse JSON trackdb '{}': {}",
                   dbfile.generic_string(), e.what());
      return false;
    }

    if (!j.is_object()) {
      std::println(std::cerr, "Invalid JSON trackdb '{}': root is not an object",
                   dbfile.generic_string());
      return false;
    }

    // Version is optional but recommended; currently we only support 1.
    int version = j.value("version", 1);
    if (version != 1) {
      std::println(std::cerr, "Unsupported trackdb JSON version {} in '{}'",
                   version, dbfile.generic_string());
      return false;
    }

    if (!j.contains("tracks") || !j["tracks"].is_array()) {
      std::println(std::cerr, "Invalid JSON trackdb '{}': missing 'tracks' array",
                   dbfile.generic_string());
      return false;
    }

    try {
      for (const auto& jti : j["tracks"]) upsert(jti.get<TrackInfo>());
    } catch (const std::exception& e) {
      std::println(std::cerr, "Error decoding TrackInfo from JSON '{}': {}",
                   dbfile.generic_string(), e.what());
      items.clear();
      return false;
    }

    return true;
  }

  bool save(const std::filesystem::path& dbfile) const
  {
    auto tracks = json::array();
    for (const auto& [key, ti] : items) {
      tracks.push_back(json(ti));
    }

    json root = {
      {"version", 1},
      {"tracks", std::move(tracks)}
    };

    std::ofstream out(dbfile, std::ios::trunc);
    if (!out.is_open()) {
      std::println(std::cerr, "Failed to open JSON trackdb for writing: {}",
                   dbfile.generic_string());
      return false;
    }

    try {
      out << root.dump(2); // pretty-print with 2-space indent
    } catch (const std::exception& e) {
      std::println(std::cerr, "Failed to write JSON trackdb '{}': {}",
                   dbfile.generic_string(), e.what());
      return false;
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
  double frame;                 // absolute cue frame in mix timeline
  long   bar;                   // 1-based global bar number in the mix
  std::filesystem::path track;  // which track this cue comes from
  int    local_bar;             // bar number within that track (1-based)
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
  const auto samplerate = static_cast<uint_t>(track.sample_rate);

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
          sum += track[fr, c];
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
  if (sr == 0 || frames == 0) return;

  assert(max_attack_db_per_s > 0.0f);
  assert(max_release_db_per_s > 0.0f);

  // 1) Required attenuation (dB) to meet ceiling at each frame (computed on demand)
  auto required_att_dB = [&](size_t f) -> float {
    float pk = buf[f].peak();
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
    buf[f] *= std::clamp(dbamp(-att[f]), 0.0f, 1.0f);
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

  struct Track {
    std::filesystem::path filename;
    TrackInfo ti;
  };
  // Collect TrackInfo and ensure cues exist
  std::vector<Track> tracks;
  tracks.reserve(files.size());
  for (auto const& file : files) {
    auto* info = g_db.find(file);
    if (!info || info->cue_bars.empty()) {
      throw std::runtime_error("Track missing in DB or has no cues: " + file.generic_string());
    }
    tracks.push_back(Track{file, *info});
  }

  // Mix BPM default: mean of track bpms (unless forced)
  auto bpm = force_bpm.value_or([&]{
    const auto bpms = tracks | std::views::transform([](Track const &track) { return track.ti.bpm; });
    return std::ranges::fold_left(bpms, 0.0, std::plus<double>{}) / static_cast<double>(tracks.size());
  }());
  g_mix_bpm = bpm;
  g_mix_bpb = tracks.front().ti.beats_per_bar;

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
    std::expected<double, std::string> lufs;
    double gain_db = 0.0;
  };
  std::vector<Item> items(files.size());

  // Parallel per-track processing with std::async (exceptions propagate via future::get)
  std::transform(std::execution::par, tracks.begin(), tracks.end(), items.begin(),
    [&](Track const& track) {
      Interleaved<float> t = load_track(track.filename);

      // Pre-resample peak headroom
      const float targetHeadroom = dbamp(kHeadroomDB);
      const float peak_in = t.peak();
      if (peak_in > 0.f && peak_in > targetHeadroom) {
        const float gain = targetHeadroom / peak_in;
        t *= gain;
      }

      auto res = change_tempo(t, track.ti.bpm, bpm, outRate, converter_type);

      auto lufs = measure_lufs(res);
      if (!lufs) {
        std::println(std::cerr, "LUFS measurement failed for {}: {}",
                     track.filename.generic_string(), lufs.error());
      }

      size_t frames = res.frames();

      int firstBar = track.ti.cue_bars.front();
      int lastBar  = track.ti.cue_bars.back();
      double shiftOut = track.ti.upbeat_beats * fpb + track.ti.time_offset_sec * (double)outRate;
      double firstCue = shiftOut + (double)(firstBar - 1) * (double)track.ti.beats_per_bar * fpb;
      double lastCue  = shiftOut + (double)(lastBar  - 1) * (double)track.ti.beats_per_bar * fpb;

      // Clamp
      if (frames == 0) { firstCue = lastCue = 0.0; }
      else {
        firstCue = std::clamp(firstCue, 0.0, (double)(frames - 1));
        lastCue  = std::clamp(lastCue,  0.0, (double)(frames - 1));
      }

      return Item{ track.filename, track.ti, std::move(res), firstCue, lastCue, 0.0, std::move(lufs), 0.0 };
    }
  );

  // Compute target LUFS as mean of all track LUFS; throw if any failed
  double sum_lufs = 0.0;
  for (auto const& it : items) {
    if (!it.lufs) {
      throw std::runtime_error(
        "LUFS measurement failed for " + it.file.generic_string() +
        ": " + it.lufs.error()
      );
    }
    sum_lufs += *it.lufs;
  }
  const double target_lufs = items.empty()
    ? 0.0
    : sum_lufs / static_cast<double>(items.size());

  // Compute per-track gain_db with clamping
  for (auto& it : items) {
    it.gain_db = std::clamp(target_lufs - *it.lufs, -12.0, 6.0);
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
    const auto offsetFrames = static_cast<std::size_t>(std::ceil(it.offset));
    totalFrames = std::max(totalFrames, offsetFrames + it.res.frames());
  }

  auto out = std::make_shared<Interleaved<float>>(outRate, (size_t)outCh, totalFrames);
  std::fill_n(out->data(), out->samples(), 0.0f);

  // Mix down to out channels
  const auto outChS = (size_t)outCh;
  for (auto& it : items) {
    const size_t inChS = it.res.channels();
    const float gain_lin = dbamp(static_cast<float>(it.gain_db));
    std::println("gain_db: {}", it.gain_db);
    for (size_t f = 0; f < it.res.frames(); ++f) {
      double absF = it.offset + (double)f;
      if (absF < 0.0) continue;
      auto outF = static_cast<size_t>(absF);
      if (outF >= totalFrames) break;
      const auto a = gain_lin * fade_for_frame(
        f,
        it.res.frames(),
        it.firstCue,
        it.lastCue,
        FadeCurve::Sine  // sine-shaped equal-power style fade
      );
      if (a <= 0.0f) continue;

      for (size_t ch = 0; ch < outChS; ++ch) {
        const size_t sC = ch % inChS;
        (*out)[outF, ch] += a * it.res[f, sC];
      }
    }
    it.res = Interleaved<float>();
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

      g_mix_cues.push_back(MixCue{
        mixFrame,
        barIdx,
        it.file,
        bar
      });
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
        auto beatNow  = (uint64_t)std::floor(adjNow / framesPerBeatSrc);
        auto beatNext = (uint64_t)std::floor(adjNext / framesPerBeatSrc);
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
      size_t i1 = std::min(i0 + 1, totalSrcFrames - 1);

      float click = player.metro.process(pos - shiftSrc, framesPerBeatSrc, devRate);

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
        std::println(std::cerr,  "Unknown command: {}", args[0]);
        continue;
      }
      try {
        it->second.fn(std::span<const std::string>{args}.subspan(1));
      } catch (const std::exception& e) {
        std::println(std::cerr, "Error: {}", e.what());
      } catch (...) {
        std::println(std::cerr, "Unknown error.");
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
        std::println(std::cerr, "Invalid dB value: {}", v.error());
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
    std::println(std::cerr, "Error: {}", e.what());
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
      std::println(std::cerr, "BPM detection failed: {}", e.what());
    }
  }

  std::println(std::cout, "Opened {}", f.generic_string());
  if (guessedBpm > 0) std::println(std::cout, "Guessed BPM: {:.2f}", guessedBpm);
  std::println(std::cout, "BPM: {:.2f}", ti.bpm);
  std::println(std::cout, "Beats/bar: {}", ti.beats_per_bar);
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
      if (*v <= 0.0) {
        std::println(std::cerr, "Invalid BPM: must be > 0");
        return;
      }
      g_player.metro.bpm.store(*v);
      ti.bpm = *v;
      dirty = true;
      print_estimated_bars();
    } else {
      std::println(std::cerr, "Invalid BPM: {}", v.error());
    }
  });

  sub.register_command("bpb", "bpb [value] - get/set beats per bar", [&](std::span<const std::string> a){
    if (a.empty()) {
      std::println(std::cout, "Beats/bar: {}", g_player.metro.bpb.load());
      return;
    }
    if (auto v = parse_number<unsigned>(a[0]); v) {
      if (*v == 0u) {
        std::println(std::cerr, "Invalid beats-per-bar: must be > 0");
        return;
      }
      g_player.metro.bpb.store(*v);
      ti.beats_per_bar = *v;
      dirty = true;
      print_estimated_bars();
    } else {
      std::println(std::cerr, "Invalid beats-per-bar: {}", v.error());
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
      std::println(std::cerr, "Invalid upbeat value: {}", v.error());
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
      std::println(std::cerr, "Invalid time offset: {}", v.error());
    }
  });

  sub.register_command("cue", "cue <bar> - add a cue at given bar (1-based)", [&](std::span<const std::string> a){
    if (a.size() != 1) {
      std::println(std::cerr, "Usage: cue <bar>");
      return;
    }
    if (auto bar = parse_number<int>(a[0]); bar) {
      if (*bar <= 0) {
        std::println(std::cerr, "Invalid bar: must be > 0");
        return;
      }
      if (std::find(ti.cue_bars.begin(), ti.cue_bars.end(), *bar) == ti.cue_bars.end()) {
        ti.cue_bars.push_back(*bar);
        std::sort(ti.cue_bars.begin(), ti.cue_bars.end());
        dirty = true;
      }
      if (ti.cue_bars.empty()) {
        std::println(std::cout, "(no cues)");
      } else {
        std::cout << "Cues: ";
        for (size_t i = 0; i < ti.cue_bars.size(); ++i) {
          if (i) std::cout << ',';
          std::cout << ti.cue_bars[i];
        }
        std::cout << "\n";
      }
    } else {
      std::println(std::cerr, "Invalid bar: {}", bar.error());
    }
  });

  sub.register_command("uncue", "uncue <bar> - remove a cue", [&](std::span<const std::string> a){
    if (a.size() != 1) {
      std::println(std::cerr, "Usage: uncue <bar>");
      return;
    }
    if (auto bar = parse_number<int>(a[0]); bar) {
      if (std::erase(ti.cue_bars, *bar) > 0) {
        dirty = true;
      }
    } else {
      std::println(std::cerr, "Invalid bar: {}", bar.error());
    }
  });

  sub.register_command("cues", "List cue bars", [&](std::span<const std::string>){
    if (ti.cue_bars.empty()) {
      std::println(std::cout, "(no cues)");
      return;
    }
    for (size_t i = 0; i < ti.cue_bars.size(); ++i) {
      if (i) std::cout << ',';
      std::cout << ti.cue_bars[i];
    }
    std::cout << "\n";
  });

  sub.register_command("tags", "List tags for this track", [&](std::span<const std::string>){
    if (ti.tags.empty()) {
      std::println(std::cout, "(no tags)");
      return;
    }
    bool first = true;
    for (const auto& tag : ti.tags) {
      if (!first) std::cout << ", ";
      std::cout << tag;
      first = false;
    }
    std::cout << "\n";
  });

  sub.register_command("tag", "tag <name> - add a tag to this track", [&](std::span<const std::string> a){
    if (a.size() != 1) {
      std::println(std::cerr, "Usage: tag <name>");
      return;
    }
    std::string name = a[0];
    auto first = std::find_if_not(name.begin(), name.end(),
                                  [](unsigned char c){ return std::isspace(c); });
    auto last  = std::find_if_not(name.rbegin(), name.rend(),
                                  [](unsigned char c){ return std::isspace(c); }).base();
    if (first >= last) {
      std::println(std::cerr, "Empty tag name.");
      return;
    }
    std::string trimmed(first, last);
    if (trimmed.empty()) {
      std::println(std::cerr, "Empty tag name.");
      return;
    }
    ti.tags.insert(trimmed);
    dirty = true;
    std::cout << "Tags: ";
    bool firstOut = true;
    for (const auto& tag : ti.tags) {
      if (!firstOut) std::cout << ", ";
      std::cout << tag;
      firstOut = false;
    }
    std::cout << "\n";
  });

  register_volume_command(sub, "Track");

  sub.register_command("save", "Persist BPM/Beats-per-bar to trackdb", [&](std::span<const std::string>){
    g_db.upsert(ti);
    if (g_db.save(trackdb_path)) {
      std::println(std::cout, "Saved to {}", trackdb_path.generic_string());
      dirty = false;
    } else {
      std::println(std::cerr, "Failed to save DB to {}",
                   trackdb_path.generic_string());
    }
  });

  sub.register_command("play", "Start playback with metronome overlay", [&](std::span<const std::string>){
    if (!g_player.track) {
      std::println(std::cerr, "No track loaded.");
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
      std::println(std::cerr, "Usage: seek <bar>");
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
      std::println(std::cerr, "Invalid bar number: {}", bar1.error());
    }
  });

  sub.register_command("exit", "Leave track shell", [&](std::span<const std::string>){
    sub.stop();
  });

  sub.run("track-info> ");
  g_player.playing.store(false);
}

// Readline completion for track-info filenames (supports spaces via quoting)
char** clmix_completion(const char* text, int start, int end) {
  (void)end;

  const char* line = rl_line_buffer;

  // Extract the first word (command) from the line up to 'start'
  std::string_view sv(line, static_cast<size_t>(start));
  auto it = std::find_if_not(sv.begin(), sv.end(),
                             [](unsigned char c){ return std::isspace(c); });
  if (it == sv.end()) {
    return nullptr;
  }
  auto cmd_start = it;
  auto cmd_end = std::find_if(cmd_start, sv.end(),
                              [](unsigned char c){ return std::isspace(c); });
  std::string cmd(cmd_start, cmd_end);

  // Only provide custom completion for the first argument of "track-info"
  if (cmd != "track-info") {
    return nullptr;
  }

  // Generator that iterates over g_db.items and returns matching filenames.
  auto generator = [](const char* text, int state) -> char* {
    static std::vector<std::string> matches;
    static size_t index;

    if (state == 0) {
      matches.clear();
      index = 0;

      std::string prefix(text);
      for (const auto& [path, ti] : g_db.items) {
        (void)ti;
        const std::string name = path.generic_string();
        if (name.starts_with(prefix)) {
          matches.push_back(name);
        }
      }
    }

    if (index >= matches.size()) {
      return nullptr;
    }

    const std::string& s = matches[index++];
    char* out = static_cast<char*>(std::malloc(s.size() + 1));
    if (!out) return nullptr;
    std::memcpy(out, s.c_str(), s.size() + 1);
    return out;
  };

  return rl_completion_matches(text, generator);
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

  g_db.load(trackdb_path);

  ma_device_config config = ma_device_config_init(ma_device_type_playback);
  config.playback.format   = ma_format_f32;
  config.playback.channels = 2;
  config.sampleRate        = 44100;
  config.noPreSilencedOutputBuffer = false;
  config.dataCallback      = callback;
  config.pUserData         = &g_player;
  ma_device device;
  ma_result res = ma_device_init(nullptr, &config, &device);
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

  // Set up readline completion for track-info filenames (with quoting)
  rl_attempted_completion_function = clmix_completion;
  // Reasonable shell-like word breaks; Readline will respect quotes when completing.
  rl_basic_word_break_characters = const_cast<char*>(" \t\n\"'`@$><=;|&{(");

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
      std::println(std::cerr, "Usage: track-info <file>");
      return;
    }
    run_track_info_shell(args[0], trackdb_path);
  });
  
  register_volume_command(repl, "Mix");
  
  // Mix commands
  repl.register_command("add", "add <file> - add track to mix (opens track-info if not in DB)", [&](std::span<const std::string> a){
    if (a.size() != 1) {
      std::println(std::cerr, "Usage: add <file>");
      return;
    }
    std::filesystem::path f = a[0];
    if (!g_db.find(f)) {
      run_track_info_shell(f, trackdb_path);
    }
    if (!g_db.find(f)) {
      std::println(std::cerr, "Track still not in DB. Aborting.");
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
      std::println(std::cerr, "Failed to build mix: {}", e.what());
    }
  });

  repl.register_command("bpm", "bpm [value] - show/set mix BPM (recomputes mix)", [&](std::span<const std::string> a){
    if (g_mix_tracks.empty()) {
      std::println(std::cerr, "No tracks in mix.");
      return;
    }
    if (a.empty()) {
      std::println(std::cout, "Mix BPM: {:.2f}", g_mix_bpm);
      return;
    }
    if (auto v = parse_number<double>(a[0]); v) {
      if (*v <= 0.0) {
        std::println(std::cerr, "Invalid BPM: must be > 0");
        return;
      }
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
      std::println(std::cerr, "Invalid BPM: {}", v.error());
    }
  });

  repl.register_command("play", "Start playback (mix)", [&](std::span<const std::string>){
    if (!g_player.track) {
      if (g_mix_tracks.empty()) {
        std::println(std::cerr, "No tracks in mix.");
        return;
      }
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
        std::println(std::cerr, "Build mix failed: {}", e.what());
        return;
      }
    }
    g_player.seekPending.store(false);
    g_player.playing.store(true);
  });

  repl.register_command("stop", "Stop playback", [&](std::span<const std::string>){
    g_player.playing.store(false);
  });

  repl.register_command("seek", "seek <bar> - jump to mix bar (1-based)", [&](std::span<const std::string> a){
    if (a.size() != 1 || !g_player.track) {
      std::println(std::cerr, "Usage: seek <bar>");
      return;
    }
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
      std::println(std::cerr, "Invalid bar number: {}", bar1.error());
    }
  });

  repl.register_command("cues", "List all cue points in current mix", [&](std::span<const std::string>){
    if (g_mix_cues.empty()) {
      std::println(std::cout, "(no cues)");
      return;
    }
    for (const auto& c : g_mix_cues) {
      auto name = c.track.filename().generic_string();
      std::println(std::cout,
                   "mix bar {}  |  track: {}  |  track bar {}",
                   c.bar, name, c.local_bar);
    }
  });

  repl.register_command("tags", "List all tags present in track DB", [&](std::span<const std::string>){
    std::set<std::string> all;
    for (const auto& [path, ti] : g_db.items) {
      all.insert(ti.tags.begin(), ti.tags.end());
    }
    if (all.empty()) {
      std::println(std::cout, "(no tags)");
      return;
    }
    bool first = true;
    for (const auto& tag : all) {
      if (!first) std::cout << ", ";
      std::cout << tag;
      first = false;
    }
    std::cout << "\n";
  });

  repl.register_command("list",
    "list [tag_expr] - list tracks in DB matching tag/bpm expression "
    "(e.g. \">=140bpm & <150bpm | techno\")",
    [&](std::span<const std::string> args){
      if (g_db.items.empty()) {
        std::println(std::cerr, "Track DB is empty.");
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
          std::println(std::cerr, "Invalid tag expression: {}", e.what());
          return;
        }
      }

      std::size_t count = 0;
      for (const auto& [path, ti] : g_db.items) {
        if (!ti.cue_bars.empty() && matcher(ti)) {
          ++count;
          std::cout << std::setw(3) << count << ". "
                    << path.generic_string()
                    << "  |  BPM: " << std::fixed << std::setprecision(2) << ti.bpm
                    << "  |  Tags: ";
          if (ti.tags.empty()) {
            std::cout << "(none)";
          } else {
            bool first = true;
            for (const auto& tag : ti.tags) {
              if (!first) std::cout << ", ";
              std::cout << tag;
              first = false;
            }
          }
          std::cout << "\n";
        }
      }

      if (count == 0) {
        if (args.empty()) {
          std::println(std::cout, "(no tracks with cues in DB)");
        } else {
          std::println(std::cout, "(no tracks matching expression)");
        }
      }
    });

  repl.register_command("random", "random [tag_expr] - build mix from all trackdb entries in random order; optional tag_expr filters by tags or bpm (e.g. \">=140bpm & <150bpm\")", [&](std::span<const std::string> args){
    if (g_db.items.empty()) {
      std::println(std::cerr, "Track DB is empty.");
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
        std::println(std::cerr, "Invalid tag expression: {}", e.what());
        return;
      }
    }

    std::vector<std::filesystem::path> all;
    all.reserve(g_db.items.size());
    for (const auto& kv : g_db.items) {
      const TrackInfo& ti = kv.second;
      if (!ti.cue_bars.empty() && matcher(ti)) {
        all.push_back(ti.filename);
      }
    }
    if (all.empty()) {
      if (args.empty()) {
        std::println(std::cerr, "No tracks with cues in DB.");
      } else {
        std::println(std::cerr, "No tracks with cues matching tag expression.");
      }
      return;
    }
    std::mt19937 rng(std::random_device{}());
    std::shuffle(all.begin(), all.end(), rng);

    g_mix_tracks = std::move(all);
    std::println(std::cout, "Track order:");
    for (size_t i = 0; i < g_mix_tracks.size(); ++i) {
      std::println(std::cout, "  {}. {}",
                   i + 1, g_mix_tracks[i].generic_string());
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
      std::println(std::cerr, "Failed to build random mix: {}", e.what());
    }
  });

  repl.register_command("export", "export <file.wav> - render mix to 24-bit WAV", [&](std::span<const std::string> a){
    if (a.size() != 1) {
      std::println(std::cerr, "Usage: export <file.wav>");
      return;
    }
    if (g_mix_tracks.empty()) {
      std::println(std::cerr, "No tracks in mix.");
      return;
    }

    const std::filesystem::path outPath = a[0];
    try {
      // Stop playback to avoid concurrent access while rendering/exporting
      g_player.playing.store(false);
      // Release any existing mix from the player to avoid holding two copies in RAM
      g_player.track.reset();

      // Rebuild a fresh mix with current BPM and tracks
      auto mixTrack = build_mix_track(g_mix_tracks, g_mix_bpm, SRC_SINC_BEST_QUALITY);

      if (!std::in_range<sf_count_t>(mixTrack->frames())) {
        std::println(std::cerr, "Export failed: frame count too large for libsndfile.");
        return;
      }
      const auto frames = static_cast<sf_count_t>(mixTrack->frames());

      // Open 24-bit WAV for writing
      SndfileHandle sf(outPath.string(),
                       SFM_WRITE,
                       SF_FORMAT_WAV | SF_FORMAT_PCM_24,
                       static_cast<int>(mixTrack->channels()),
                       static_cast<int>(mixTrack->sample_rate));
      if (sf.error()) {
        std::println(std::cerr, "Failed to open output file: {}", outPath.generic_string());
        return;
      }

      // Write frames; libsndfile converts float -> PCM_24 and clips if needed
      const sf_count_t written = sf.writef(mixTrack->data(), frames);
      if (written != frames) {
        std::println(std::cerr, "Short write: wrote {} of {} frames",
                     written, frames);
      } else {
        std::cout << "Exported " << frames
                  << " frames (" << mixTrack->sample_rate << " Hz, "
                  << mixTrack->channels() << " ch) to " << outPath << "\n";
      }
    } catch (const std::exception& e) {
      std::println(std::cerr, "Export failed: {}", e.what());
    }
  });

  repl.run("clmix> ");

  ma_device_uninit(&device);

  return 0;
}
