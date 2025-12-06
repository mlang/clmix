# clmix

`clmix` is a command-line tool for preparing and mixing electronic music.  
It focuses on terminal workflows instead of GUIs and assumes tempo-stable tracks
(no tempo changes / live warping).

Main ideas:

- Work entirely from the shell (readline-based REPL).
- Maintain a small track database (JSON) with BPM, upbeat, cue bars, and tags.
- Build mixes by aligning cue bars between tracks at a fixed BPM.
- Render a final mix to WAV or preview it with a metronome click.

---

## Status and scope

- Designed for electronic music with constant tempo.
- No support for tempo curves, variable BPM, or live beatmatching.
- Uses offline analysis (aubio, ebur128) and simple rules to align tracks.
- Intended for personal use and experimentation, not as a full DJ suite.

---

## Dependencies

You need:

- C++23 compiler (GCC, Clang, or similar)
- CMake ≥ 3.25
- [nlohmann/json](https://github.com/nlohmann/json)
- [Boost](https://www.boost.org/) (header-only `boost` + `tbb` backend)
- [oneTBB](https://github.com/oneapi-src/oneTBB)
- [libsndfile](https://libsndfile.github.io/libsndfile/)
- [libsamplerate](https://github.com/libsndfile/libsamplerate)
- [libebur128](https://github.com/jiixyj/libebur128)
- [aubio](https://aubio.org/)
- [readline](https://tiswww.case.edu/php/chet/readline/rltop.html)
- [miniaudio](https://miniaud.io/) header (`miniaudio.h`) in your include path

On many Linux systems, most of these are available as packages.

---

## Building

Clone and build:

```bash
git clone https://github.com/mlang/clmix.git
cd clmix
cmake -B build -S .
cmake --build build
```

This produces `build/clmix`.

By default, `-march=native` is enabled. To disable:

```bash
cmake -B build -S . -DCLMIX_ENABLE_MARCH_NATIVE=OFF
cmake --build build
```

---

## Track database

`clmix` uses a JSON track database file you specify on startup:

```bash
./clmix trackdb.json
```

If `trackdb.json` does not exist, it starts empty and will be created when you
save.

The DB stores, per track:

- `filename` (path to audio file)
- `beats_per_bar` (e.g. 4)
- `bpm`
- `upbeat_beats` (offset in beats before bar 1; can be negative)
- `time_offset_sec` (time offset in seconds; can be negative)
- `cue_bars` (list of bar numbers with cues, 1-based)
- `tags` (set of strings)

Example entry:

```json
{
  "version": 1,
  "tracks": [
    {
      "filename": "music/track1.flac",
      "beats_per_bar": 4,
      "bpm": 138.0,
      "upbeat_beats": 0.0,
      "time_offset_sec": 0.0,
      "cue_bars": [1, 33, 65],
      "tags": ["trance", "uplifting"]
    }
  ]
}
```

You normally do not edit this by hand; use the `track-info` shell.

---

## Basic usage overview

Start `clmix` with a DB:

```bash
./clmix trackdb.json
```

You get a REPL prompt:

```
clmix>
```



 Type `help` to see commands.



 Typical workflow:



 1. Add tracks and annotate them via `track-info`.

 2. Build a mix (e.g. `add`, `random`, `bpm`).

 3. Preview with `play` / `stop` and `seek`.

 4. Export to WAV with `export`.



---



## Per-track editing: `track-info`



Use `track-info` to inspect and edit metadata for a single track:

```
clmix> track-info path/to/track.flac
```



This opens a sub-shell:

```
track-info>
```



If the track is not in the DB yet:



- It is loaded and analyzed.
- BPM is guessed via aubio (if possible).
- Default values are shown.

You can then adjust:


### BPM and beats per bar

```
track-info> bpm          # show BPM

track-info> bpm 138.5    # set BPM

track-info> bpb          # show beats per bar

track-info> bpb 4        # set beats per bar
```



These define the musical grid used for cues and the metronome.



### Upbeat and time offset



These align the musical grid to the audio:



- `upbeat` (in beats): how many beats before bar 1 the audio starts.
- `offset` (in seconds): additional time shift.

```
track-info> upbeat       # show upbeat in beats

track-info> upbeat -0.5  # set upbeat

track-info> offset       # show time offset in seconds

track-info> offset 0.12  # set offset
```

Together, BPM, beats-per-bar, upbeat, and offset define where bar 1, beat 1
falls in the audio.

### Cue bars

Cue bars are bar numbers (1-based) where you want important musical events
(e.g. intro start, breakdown, drop, outro). They are used later to align
tracks in a mix.

```
track-info> cue 1        # add cue at bar 1

track-info> cue 33       # add cue at bar 33

track-info> cues         # list all cue bars

track-info> uncue 33     # remove cue at bar 33
```

At least two cue bars (e.g. first and last) are useful for automatic grid
fitting.

### Tags


Tags are arbitrary strings used for filtering and random selection.

```
track-info> tags         # list tags

track-info> tag techno

track-info> tag "peak time"
```

Tags are stored in the DB and can be used in expressions like `techno & >=138bpm`.

### Autogrid

`autogrid` refines BPM and time offset using detected transients between the
first and last cue bar.

```
track-info> autogrid

track-info> autogrid 40 onsets   # window_ms=40, use onset detector
```


- Uses aubio to detect beats or onsets.
- Matches them to the theoretical grid implied by your cues and BPM.
- Fits a straight line (beat index → time) and, if the fit is good enough,
  adjusts BPM and time offset slightly (±1% BPM limit).



This is useful when your initial BPM or offset is close but not exact.

### Playback and metronome in `track-info`

You can preview the track with a metronome click:

```
track-info> play

track-info> stop
```

The metronome:

- Uses the current `bpm` and `bpb` (beats per bar).
- Clicks on every beat; downbeats (bar starts) have a higher pitch.
- Follows seeks and offset/upbeat changes.

Seek by bar:

```
track-info> seek 1       # go to bar 1

track-info> seek 33      # go to bar 33
```

If playback is running, seeks are quantized to the next bar boundary.



### Volume

Per-track playback volume (in dB):

```
track-info> vol          # show volume

track-info> vol -6       # set to -6 dB

track-info> vol -6db     # same
```

### Saving

When you are done editing:

```
track-info> save
```

This writes the updated `track_info` back to `trackdb.json`.

Exit the per-track shell:

```
track-info> exit
```

---

## Building and playing a mix

Back in the main `clmix>` shell, you can build and play mixes.

### Adding tracks manually

```
clmix> add path/to/track1.flac

clmix> add path/to/track2.flac
```

If a track is not yet in the DB, `add` will open `track-info` for it first.

After adding, `clmix` rebuilds the mix:



- Each track is time-stretched to the mix BPM (no pitch shift) using libsamplerate.
- Tracks are aligned so that the last cue of track A lines up with the first cue of track B.
- Fades are applied:
  - Fade-in from start to first cue.
  - Unity gain between first and last cue.
  - Fade-out from last cue to end.
- Loudness is normalized around the mean LUFS of all tracks.
- A simple two-pass limiter is applied.

### Mix BPM

```
clmix> bpm          # show current mix BPM

clmix> bpm 140      # set mix BPM and rebuild mix
```

If you do not set it explicitly, the mix BPM defaults to the mean BPM of the
tracks in the mix.

### Mix playback and seek

```
clmix> play

clmix> stop
```

The same metronome concept applies here, but now over the rendered mix.

Seek by mix bar:

```
clmix> seek 1

clmix> seek 64
```

Bars are counted in the mix timeline using the mix BPM and `g_mix_bpb`
(usually 4).

### Mix volume

```
clmix> vol

clmix> vol -3
```

This controls the playback gain of the mix (not the rendered file).

---

## Mix cues

When building a mix, `clmix` also computes global cue points:



- For each track’s cue bar, it computes the corresponding frame in the mix.
- It then converts that to a global bar index in the mix.

- If multiple tracks share the same global bar, the last one wins.

You can list them:

```
clmix> cues
```


Output example:

```
mix bar 1  |  track: music/track1.flac  |  track bar 1
mix bar 33 |  track: music/track2.flac  |  track bar 1
...
```


This helps you understand where each track’s structure lands in the final mix.

---

## Tags and selection


You can inspect tags across the DB:

```
clmix> tags
```

This prints all tags and how many tracks have each.



### Listing tracks with expressions

```
clmix> list

clmix> list "techno & >=138bpm & <142bpm"
```

The expression language supports:



- Tag names (e.g. `techno`, `peak`)
- Boolean operators:
  - `&` (AND)
  - `|` (OR)
  - `~` (NOT)
- Parentheses: `( ... )`
- BPM comparisons:
  - `<140bpm`, `<=128bpm`, `>135bpm`, `>=150bpm`, `==140bpm`



Examples:



- `techno & >=138bpm & <142bpm`
- `~slow & (psy | trance)`
- `>=140bpm | hard`



### Random mixes

You can build a random mix from the DB:

```
clmix> random                 # all tracks with cues

clmix> random "techno"        # random block of techno tracks

clmix> random "techno" "trance & >=138bpm"
```

For each expression:



- All tracks with cues matching the expression are collected.
- The group is shuffled and appended to the mix track list.

After `random`, the mix is rebuilt and you can `play`, `bpm`, `export`, etc.

---

## Exporting a mix



You can export the current mix to a 24-bit WAV:

```
clmix> export out.wav
```

This:



- Rebuilds the mix offline at current device rate and channel count.
- Uses highest-quality resampling (`SRC_SINC_BEST_QUALITY`).
- Writes a 24-bit WAV via libsndfile.



You can also run in non-interactive export mode:

```bash
./clmix trackdb.json --random "techno & >=138bpm" --bpm 140 --export out.wav
```

Options:


- `--random <expr>`: same expression language as `random` command.
  Can be given multiple times.
- `--bpm <value>`: force mix BPM.
- `--export <file>`: render and exit (no REPL).

---


## Notes

- Audio playback uses miniaudio; device sample rate and channel count are taken
  from the selected device.
- Internally, audio is stored as interleaved `float` frames.
- Loudness is measured with libebur128 (LUFS) and normalized per track.
- The limiter is simple but sufficient to avoid clipping in typical use.

This project is intended as a small, scriptable tool for people comfortable
with the terminal who want to prepare and render DJ-style mixes from
tempo-stable electronic tracks.
