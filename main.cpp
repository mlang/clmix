// Simple command-line tool to mix electronic music
//
// At least C++23 is required to compile this program.
// No need for defensive programming.
// We aim for beautiful code.


#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numbers>
#include <print>
#include <stdexcept>
#include <string>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "vendor/mdspan.hpp"

// We rely on mdspan's C++23 multi-arg operator[] for indexing (e.g., out[i, ch]).
template<typename T>
using multichannel = Kokkos::mdspan<T, Kokkos::dextents<std::size_t, 2>>;

#include <readline/history.h>
#include <readline/readline.h>

#include <sndfile.hh>

extern "C" {
#include <aubio/aubio.h>
}

#define MINIAUDIO_IMPLEMENTATION
#include <miniaudio.h>

struct Track {
  int sample_rate;
  int channels;
  std::vector<float> sound;
};

// Encapsulated metronome state and processing
struct Metronome {
  std::atomic<float> bpm{120.f};
  std::atomic<int> bpb{4};

  // runtime state
  uint64_t lastBeatIndex = 0;
  int clickSamplesLeft = 0;
  int clickLen = 0; // in device samples
  float clickPhase = 0.f;
  float clickAmp = 0.f;

  // click parameters
  float clickFreqHz = 1000.f;
  float downbeatAmp = 0.35f;
  float beatAmp = 0.18f;

  void reset_runtime() {
    lastBeatIndex = 0;
    clickSamplesLeft = 0;
    clickLen = 0;
    clickPhase = 0.f;
    clickAmp = 0.f;
  }

  void prepare_after_seek(double posSrcFrames, double framesPerBeatSrc) {
    reset_runtime();
    lastBeatIndex = (uint64_t)std::floor(std::max(0.0, posSrcFrames) / framesPerBeatSrc);
  }

  float process(double posSrcFrames, double framesPerBeatSrc, uint32_t devRate) {
    if (clickLen == 0) clickLen = std::max(1, (int)(devRate / 100)); // ~10ms
    uint64_t beatIndex = (uint64_t)std::floor(std::max(0.0, posSrcFrames) / framesPerBeatSrc);
    if (beatIndex != lastBeatIndex) {
      lastBeatIndex = beatIndex;
      clickSamplesLeft = clickLen;
      clickPhase = 0.f;
      int curBpb = std::max(1, bpb.load());
      bool downbeat = (beatIndex % (uint64_t)curBpb) == 0;
      clickAmp = downbeat ? downbeatAmp : beatAmp;
    }
    float click = 0.f;
    if (clickSamplesLeft > 0) {
      float env = (float)clickSamplesLeft / (float)clickLen; // linear decay
      clickPhase += 2.0f * std::numbers::pi_v<float> * clickFreqHz / (float)devRate;
      click = clickAmp * std::sinf(clickPhase) * env;
      --clickSamplesLeft;
    }
    return click;
  }
};

// Track metadata persisted in the DB
struct TrackInfo {
  std::filesystem::path filename;
  int beats_per_bar = 4;
  float bpm = 120.f; // required > 0
  std::vector<int> cue_bars; // 1-based bar numbers
};

 // Simple text DB:
 // Each line: "filename with quotes" <space> <beats_per_bar> <space> <bpm> <space> <cues_csv_or_->
 // Lines starting with '#' or blank lines are ignored.
struct TrackDB {
  std::unordered_map<std::string, TrackInfo> items;

  static std::string normkey(const std::filesystem::path& p) {
    // Use a stable textual key; generic_string() uses '/' separators.
    return p.lexically_normal().generic_string();
  }

  TrackInfo* find(const std::filesystem::path& file) {
    auto it = items.find(normkey(file));
    return (it == items.end()) ? nullptr : &it->second;
  }

  void upsert(const TrackInfo& info) {
    items[normkey(info.filename)] = info;
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
      int beats = 4;
      std::string bpm_tok;
      if (!(iss >> std::ws >> std::quoted(fname) >> beats)) {
        // Malformed line; skip
        continue;
      }
      float bpm = 120.f;
      if (iss >> bpm_tok) {
        try {
          float v = std::stof(bpm_tok);
          if (v > 0.f) bpm = v;
        } catch (...) {
          bpm = 120.f;
        }
      }
      std::vector<int> cues;
      std::string cues_tok;
      if (iss >> cues_tok) {
        if (cues_tok != "-") {
          std::stringstream ss(cues_tok);
          std::string tok;
          while (std::getline(ss, tok, ',')) {
            try {
              int bar = std::stoi(tok);
              if (bar > 0) cues.push_back(bar);
            } catch (...) {
              // ignore invalid tokens
            }
          }
          std::sort(cues.begin(), cues.end());
          cues.erase(std::unique(cues.begin(), cues.end()), cues.end());
        }
      }
      TrackInfo ti;
      ti.filename = fname;
      ti.beats_per_bar = beats;
      ti.bpm = bpm;
      ti.cue_bars = std::move(cues);
      upsert(ti);
    }
    return true;
  }

  bool save(const std::filesystem::path& dbfile) const {
    std::ofstream out(dbfile, std::ios::trunc);
    if (!out.is_open()) {
      return false;
    }
    out << "# clmix track db: \"filename\" beats_per_bar bpm cues_csv_or_-\n";
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
      out << std::quoted(ti.filename.generic_string()) << ' '
          << ti.beats_per_bar << ' '
          << std::to_string(ti.bpm) << ' '
          << cues_tok << '\n';
    }
    return true;
  }
};

struct PlayerState {
  std::atomic<bool> playing{false};
  std::shared_ptr<Track> track; // set before play; not swapped while playing
  std::atomic<float> trackGainDB{0.f}; // Track gain in dB (0 = unity; negative attenuates)

  // Seek control (source frames)
  std::atomic<bool> seekPending{false};
  std::atomic<double> seekTargetFrames{0.0};

  // Playback runtime (audio thread)
  double srcPos = 0.0;              // in source frames (fractional)

  Metronome metro;
};

static PlayerState g_player;
static TrackDB g_db;

Track load_track(std::filesystem::path file)
{
  SndfileHandle sf(file.string());
  if (sf.error()) {
    throw std::runtime_error("Failed to open audio file: " + file.string());
  }

  Track track;
  track.sample_rate = sf.samplerate();
  track.channels = sf.channels();

  const sf_count_t frames = sf.frames();
  track.sound.resize(static_cast<std::size_t>(frames) * static_cast<std::size_t>(track.channels));

  const sf_count_t read_frames = sf.readf(track.sound.data(), frames);
  if (read_frames < 0) {
    throw std::runtime_error("Failed to read audio data from file: " + file.string());
  }
  if (read_frames != frames) {
    track.sound.resize(static_cast<std::size_t>(read_frames) * static_cast<std::size_t>(track.channels));
  }

  return track;
}

float detect_bpm(const Track& track)
{
  if (track.sample_rate <= 0 || track.channels <= 0 || track.sound.empty()) {
    throw std::invalid_argument("detect_bpm: invalid or empty track");
  }

  const uint_t win_s = 1024;
  const uint_t hop_s = 512;
  const uint_t samplerate = static_cast<uint_t>(track.sample_rate);

  aubio_tempo_t* tempo = new_aubio_tempo((char*)"default", win_s, hop_s, samplerate);
  if (!tempo) {
    throw std::runtime_error("aubio: failed to create tempo object");
  }

  fvec_t* inbuf = new_fvec(hop_s);
  fvec_t* out = new_fvec(1);
  if (!inbuf || !out) {
    if (inbuf) del_fvec(inbuf);
    if (out) del_fvec(out);
    del_aubio_tempo(tempo);
    throw std::runtime_error("aubio: failed to allocate buffers");
  }

  const std::size_t channels = static_cast<std::size_t>(track.channels);
  const std::size_t total_frames = track.sound.size() / channels;

  for (std::size_t frame = 0; frame < total_frames; frame += hop_s) {
    for (uint_t j = 0; j < hop_s; ++j) {
      const std::size_t fr = frame + j;
      float v = 0.f;
      if (fr < total_frames) {
        const std::size_t base = fr * channels;
        float sum = 0.f;
        for (std::size_t c = 0; c < channels; ++c) {
          sum += track.sound[base + c];
        }
        v = sum / static_cast<float>(channels);
      }
      inbuf->data[j] = v;
    }
    aubio_tempo_do(tempo, inbuf, out);
  }

  const float bpm = aubio_tempo_get_bpm(tempo);

  del_fvec(inbuf);
  del_fvec(out);
  del_aubio_tempo(tempo);

  return bpm;
}

template<typename T>
static inline T dbamp(T db)
{
  return std::pow(T(10.0), db * T(0.05));
}

static inline float lininterp(float a, float b, float t) {
  return std::lerp(a, b, t);
}

static void play(
  PlayerState &player, multichannel<float> output, uint32_t devRate
)
{
  if (player.track) {
    auto &track = *player.track;
    const float bpm = std::max(1.f, player.metro.bpm.load());
    const float gainLin = dbamp(player.trackGainDB.load());
    const size_t srcCh = static_cast<size_t>(track.channels);
    const size_t totalSrcFrames = track.sound.size() / srcCh;
    if (totalSrcFrames == 0) return;


    const double framesPerBeatSrc = (double)track.sample_rate * 60.0 / (double)bpm;
    const double incrSrcPerOut = (double)track.sample_rate / (double)devRate;

    double pos = player.srcPos;

    for (uint32_t i = 0; i < output.extent(0); ++i) {
      if (player.seekPending.load(std::memory_order_acquire)) {
	pos = player.seekTargetFrames.load(std::memory_order_relaxed);
	player.seekPending.store(false, std::memory_order_release);
	player.metro.prepare_after_seek(pos, framesPerBeatSrc);
      }

      if (pos >= (double)(totalSrcFrames - 1)) {
	player.playing.store(false);
	break;
      }

      // Linear interpolation per channel
      size_t i0 = (size_t)pos;
      double frac = pos - (double)i0;
      size_t i1 = std::min(i0 + 1, totalSrcFrames - 1);
      size_t base0 = i0 * srcCh;
      size_t base1 = i1 * srcCh;

      float click = player.metro.process(pos, framesPerBeatSrc, devRate);

      // Write each output channel from the corresponding source channel (wrap if more outs)
      for (uint32_t ch = 0; ch < output.extent(1); ++ch) {
        size_t srcC = (size_t)ch % srcCh;
        float s0 = track.sound[base0 + srcC];
        float s1 = track.sound[base1 + srcC];
        float smp = lininterp(s0, s1, static_cast<float>(frac));
        float mix = (smp * gainLin) + click;
        output[i, ch] = mix;
      }

      pos += incrSrcPerOut;
    }

    player.srcPos = pos;
  }
}

static void callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount)
{
  PlayerState &player = *static_cast<PlayerState*>(pDevice->pUserData);
  multichannel<float> output(static_cast<float*>(pOutput),
    frameCount, pDevice->playback.channels
  );

  if (player.playing.load()) play(player, output, pDevice->sampleRate);
}

// Shell-style tokenizer supporting quotes and backslashes.
// - Whitespace splits args when not inside quotes.
// - Single quotes: literals (no escapes inside).
// - Double quotes: supports backslash escaping of \" and \\ (simple treatment).
static std::vector<std::string> parse_command_line(const std::string& s) {
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
using Command = std::function<void(const std::vector<std::string>&)>;

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
      std::vector<std::string> pargs(args.begin() + 1, args.end());
      try {
        it->second.fn(pargs);
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
    std::unordered_map<std::string, CommandEntry> commands_;
    bool running_ = true;
};

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
  config.sampleRate        = 48000;
  config.noPreSilencedOutputBuffer = false;
  config.dataCallback      = callback;
  config.pUserData         = &g_player;
  ma_device device;
  if (ma_device_init(NULL, &config, &device) != MA_SUCCESS) {
    return 1;
  }
  ma_device_start(&device);

  g_db.load(trackdb_path);

  REPL repl;

  repl.register_command("help", "List commands", [&](const std::vector<std::string>&){
    repl.print_help();
  });
  repl.register_command("exit", "Exit program", [&](const std::vector<std::string>&){
    repl.stop();
  });
  repl.register_command("quit", "Alias for exit", [&](const std::vector<std::string>&){
    repl.stop();
  });
  repl.register_command("echo", "Echo arguments; supports quoted args", [](const std::vector<std::string>& args){
    for (size_t i = 0; i < args.size(); ++i) {
      if (i) std::cout << ' ';
      std::cout << args[i];
    }
    std::cout << "\n";
  });
  repl.register_command("track-info", "track-info <file> - open per-track shell", [&](const std::vector<std::string>& args){
    if (args.size() != 1) {
      std::cerr << "Usage: track-info <file>\n";
      return;
    }
    std::filesystem::path f = args[0];
    Track t;
    try {
      t = load_track(f);
    } catch (const std::exception& e) {
      std::cerr << "Error: " << e.what() << "\n";
      return;
    }
    auto tr = std::make_shared<Track>(std::move(t));

    float guessedBpm = 0.f;

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

    // Initialize player state for this track (not playing yet)
    g_player.track = tr;
    g_player.srcPos = 0.0;
    g_player.seekTargetFrames.store(0.0);
    g_player.seekPending.store(false);
    g_player.trackGainDB.store(0.f);
    g_player.metro.reset_runtime();
    g_player.metro.bpm.store(ti.bpm);
    g_player.metro.bpb.store(std::max(1, ti.beats_per_bar));

    auto print_estimated_bars = [&](){
      float bpmNow = std::max(1.f, g_player.metro.bpm.load());
      int bpbNow = std::max(1, g_player.metro.bpb.load());
      size_t totalFrames = tr->sound.size() / (size_t)tr->channels;
      double framesPerBeat = (double)tr->sample_rate * 60.0 / (double)bpmNow;
      double beats = (double)totalFrames / framesPerBeat;
      double bars = beats / (double)bpbNow;
      std::println(std::cout, "Estimated bars: {:.2f}", bars);
    };
    print_estimated_bars();

    // Per-track subshell
    REPL sub;
    bool dirty = false;

    sub.register_command("help", "List commands", [&](const std::vector<std::string>&){ sub.print_help(); });

    sub.register_command("bpm", "bpm [value] - get/set BPM", [&](const std::vector<std::string>& a){
      if (a.empty()) {
        std::println(std::cout, "BPM: {:.2f}", g_player.metro.bpm.load());
        return;
      }
      try {
        float v = std::stof(a[0]);
        if (v <= 0) throw std::runtime_error("BPM must be > 0");
        g_player.metro.bpm.store(v);
        ti.bpm = v;
        dirty = true;
        print_estimated_bars();
      } catch (...) {
        std::cerr << "Invalid BPM.\n";
      }
    });

    sub.register_command("bpb", "bpb [value] - get/set beats per bar", [&](const std::vector<std::string>& a){
      if (a.empty()) {
        std::cout << "Beats/bar: " << g_player.metro.bpb.load() << "\n";
        return;
      }
      try {
        int v = std::stoi(a[0]);
        if (v <= 0) throw std::runtime_error("bpb must be > 0");
        g_player.metro.bpb.store(v);
        ti.beats_per_bar = v;
        dirty = true;
        print_estimated_bars();
      } catch (...) {
        std::cerr << "Invalid beats-per-bar.\n";
      }
    });

    sub.register_command("cue", "cue <bar> - add a cue at given bar (1-based)", [&](const std::vector<std::string>& a){
      if (a.size() != 1) {
        std::cerr << "Usage: cue <bar>\n";
        return;
      }
      try {
        int bar = std::stoi(a[0]);
        if (bar <= 0) throw std::runtime_error("bar must be > 0");
        if (std::find(ti.cue_bars.begin(), ti.cue_bars.end(), bar) == ti.cue_bars.end()) {
          ti.cue_bars.push_back(bar);
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
      } catch (...) {
        std::cerr << "Invalid bar.\n";
      }
    });

    sub.register_command("uncue", "uncue <bar> - remove a cue", [&](const std::vector<std::string>& a){
      if (a.size() != 1) {
        std::cerr << "Usage: uncue <bar>\n";
        return;
      }
      try {
        int bar = std::stoi(a[0]);
        auto it = std::remove(ti.cue_bars.begin(), ti.cue_bars.end(), bar);
        if (it != ti.cue_bars.end()) {
          ti.cue_bars.erase(it, ti.cue_bars.end());
          dirty = true;
        }
      } catch (...) {
        std::cerr << "Invalid bar.\n";
      }
    });

    sub.register_command("cues", "List cue bars", [&](const std::vector<std::string>&){
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

    sub.register_command("vol", "vol [dB] - get/set track volume in dB (0=unity, negative attenuates)", [&](const std::vector<std::string>& a){
      if (a.empty()) {
        float db = g_player.trackGainDB.load();
        float lin = std::pow(10.f, db / 20.f);
        std::println(std::cout, "Track volume: {:.2f} dB (x{:.3f})", db, lin);
        return;
      }
      try {
        std::string s = a[0];
        if (s.size() >= 2) {
          std::string tail = s.substr(s.size() - 2);
          std::transform(tail.begin(), tail.end(), tail.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
          if (tail == "db") s.resize(s.size() - 2);
        }
        float db = std::stof(s);
        if (db < -60.f) db = -60.f;
        if (db > 12.f) db = 12.f;
        g_player.trackGainDB.store(db);
        float lin = std::pow(10.f, db / 20.f);
        std::println(std::cout, "Track volume set to {:.2f} dB (x{:.3f})", db, lin);
      } catch (...) {
        std::cerr << "Invalid dB value.\n";
      }
    });

    sub.register_command("save", "Persist BPM/Beats-per-bar to trackdb", [&](const std::vector<std::string>&){
      g_db.upsert(ti);
      if (g_db.save(trackdb_path)) {
        std::cout << "Saved to " << trackdb_path << "\n";
        dirty = false;
      } else {
        std::cerr << "Failed to save DB to " << trackdb_path << "\n";
      }
    });

    sub.register_command("play", "Start playback with metronome overlay", [&](const std::vector<std::string>&){
      if (!g_player.track) {
        std::cerr << "No track loaded.\n";
        return;
      }
      // ensure position valid
      g_player.seekPending.store(false);
      g_player.playing.store(true);
    });

    sub.register_command("stop", "Stop playback", [&](const std::vector<std::string>&){
      g_player.playing.store(false);
    });

    sub.register_command("seek", "seek <bar> - jump to given bar (1-based)", [&](const std::vector<std::string>& a){
      if (a.size() != 1) {
        std::cerr << "Usage: seek <bar>\n";
        return;
      }
      try {
        int bar1 = std::stoi(a[0]); // 1-based
        int bar0 = std::max(0, bar1 - 1);
        float bpmNow = std::max(1.f, g_player.metro.bpm.load());
        int bpbNow = std::max(1, g_player.metro.bpb.load());
        double framesPerBeat = (double)tr->sample_rate * 60.0 / (double)bpmNow;
        double target = (double)bar0 * (double)bpbNow * framesPerBeat;
        size_t totalFrames = tr->sound.size() / (size_t)tr->channels;
        if (target >= (double)totalFrames) target = (double)totalFrames - 1.0;
        if (target < 0.0) target = 0.0;
        g_player.seekTargetFrames.store(target, std::memory_order_relaxed);
        g_player.seekPending.store(true, std::memory_order_release);
      } catch (...) {
        std::cerr << "Invalid bar number.\n";
      }
    });

    sub.register_command("exit", "Leave track shell", [&](const std::vector<std::string>&){
      sub.stop();
    });

    sub.run("track-info> ");
    g_player.playing.store(false);
  });

  repl.run("clmix> ");

  ma_device_uninit(&device);
  return 0;
}
