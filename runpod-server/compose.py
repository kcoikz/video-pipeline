#!/usr/bin/env python3
"""compose.py — v31 Step 6: cinematic ffmpeg composer.

Replaces the inline ffmpeg concat-demuxer rendering in _run_render_background
with a complex filter graph that adds motion (zoompan), smooth transitions
(xfade), unified color grading (lut3d), film grain texture overlay, and a
vignette. Result: still-image slideshow → cinematic film.

Per-clip processing:
  1. scale to 1920x1080 (preserves aspect, crops overflow)
  2. zoompan applies a slow 5% zoom over the clip's duration. Direction
     alternates by index (in-center, out-center, in-left, out-right) so
     adjacent clips don't all pan the same way.

Between clips:
  3. xfade crossfade (default 1.0s). Replaces hard cuts with film-style
     dissolves.

Whole-video pass:
  4. lut3d color grading from /workspace/assets/film_noir.cube (if present).
     Pushes everything to the same teal/orange or B&W noir look.
  5. Overlay film grain from /workspace/assets/grain.mp4 (looped, screen-
     blended at ~18% opacity) so digital perfection looks like celluloid.
  6. vignette darkens corners (PI/4 angle, dirty look).

Audio:
  7. afade in 2s, afade out 2s. No abrupt start/stop.

If LUT or grain files don't exist, those filters are skipped silently. The
zoompan, xfade, vignette, and audio fades always apply.

Usage:
    compose.py --job-id JOB --video-type short_en --scene-count 90 \\
               --scene-dur 8 --audio /path/audio.wav --output /path/out.mp4
"""

import argparse
import subprocess
import sys
from pathlib import Path


def get_audio_duration(audio_path: Path) -> float:
    """Return actual audio duration in seconds via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(
            f"ffprobe failed to read duration from {audio_path}: {result.stderr}"
        )
    return float(result.stdout.strip())

# Asset paths — files are optional. If missing, the corresponding filter is
# skipped. Drop a .cube and a grain.mp4 here to enable the full cinematic look.
ASSETS_DIR = Path("/workspace/assets")
LUT_PATH   = ASSETS_DIR / "film_noir.cube"
GRAIN_PATH = ASSETS_DIR / "grain.mp4"

# Render settings
TARGET_W = 1920
TARGET_H = 1080
FPS      = 30
FADE_DUR = 1.0       # crossfade duration in seconds
AUDIO_FADE = 2.0     # in/out audio fade duration


def _per_clip_filter(in_idx: int, duration: float, zoom_dir: str) -> str:
    """Build the per-clip filter chain: scale → zoompan → format → label."""
    zoom_frames = max(1, round(duration * FPS))
    step = 0.05 / zoom_frames  # 1.0 → 1.05 over duration

    # zoompan z expression and x/y anchoring vary by direction
    if zoom_dir == "in_center":
        z_expr = f"if(eq(on,0),1,zoom+{step})"
        xy     = "x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
    elif zoom_dir == "out_center":
        z_expr = f"if(eq(on,0),1.05,if(gt(zoom,1.001),zoom-{step},1.001))"
        xy     = "x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
    elif zoom_dir == "in_left":
        z_expr = f"if(eq(on,0),1,zoom+{step})"
        xy     = "x='0':y='ih/2-(ih/zoom/2)'"
    elif zoom_dir == "out_right":
        z_expr = f"if(eq(on,0),1.05,if(gt(zoom,1.001),zoom-{step},1.001))"
        xy     = "x='iw-iw/zoom':y='ih/2-(ih/zoom/2)'"
    else:
        z_expr = "1.0"
        xy     = "x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"

    return (
        f"[{in_idx}:v]"
        f"scale={TARGET_W}:{TARGET_H}:flags=lanczos:force_original_aspect_ratio=increase,"
        f"crop={TARGET_W}:{TARGET_H},"
        f"zoompan=z='{z_expr}':d={zoom_frames}:s={TARGET_W}x{TARGET_H}:{xy}:fps={FPS},"
        f"format=yuv420p,setpts=PTS-STARTPTS[clip{in_idx}]"
    )


def compose(
    job_id: str,
    video_type: str,
    scene_count: int,
    scene_dur: float,           # was int — now float for exact audio sync
    audio_path: Path,
    output_path: Path,
    jobs_dir: Path,
    scene_offset: int = 0,      # first global image index for this render (sleep chapters)
    failed_chunks: list[int] | None = None,  # global indices to substitute
) -> str:
    """Build and run the cinematic ffmpeg pipeline. Returns output path."""
    folder = "short" if "short" in video_type else "sleep"
    job_path = jobs_dir / job_id / folder
    failed_set: set[int] = set(failed_chunks or [])

    # ── Collect image paths ───────────────────────────────────────────
    # First pass: find all valid images in the requested range.
    valid: dict[int, Path] = {}
    for i in range(scene_offset, scene_offset + scene_count):
        for ext in ("png", "jpg"):
            img = job_path / f"scene_{i:03d}.{ext}"
            if img.exists():
                valid[i] = img
                break

    # Second pass: build clip list substituting failed / missing frames.
    clips: list[Path] = []
    all_valid_sorted = sorted(valid.keys())
    for i in range(scene_offset, scene_offset + scene_count):
        if i not in failed_set and i in valid:
            clips.append(valid[i])
            continue
        # Need a substitute — prefer nearest PRIOR valid non-failed frame.
        sub: Path | None = None
        for j in range(i - 1, scene_offset - 1, -1):
            if j in valid and j not in failed_set:
                sub = valid[j]
                break
        if sub is None:
            for j in range(i + 1, scene_offset + scene_count):
                if j in valid and j not in failed_set:
                    sub = valid[j]
                    break
        if sub is None:
            raise FileNotFoundError(
                f"No valid frames available to substitute for index {i} "
                f"(scene_offset={scene_offset}, scene_count={scene_count})"
            )
        clips.append(sub)

    if not clips:
        raise ValueError("scene_count is 0 — nothing to render")
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    # ── Duration correction ───────────────────────────────────────────
    # Get the true audio duration via ffprobe, then back-calculate the
    # per-clip duration so that:
    #   total_video_dur = corrected_scene_dur * N - FADE_DUR * (N - 1)
    #                   = audio_dur  (exactly — no frozen frames, no cutoff)
    #
    # Formula derivation:
    #   total_dur = c * N - FADE_DUR * (N - 1) = audio_dur
    #   => c = (audio_dur + FADE_DUR * (N - 1)) / N
    audio_dur = get_audio_duration(audio_path)
    N = len(clips)
    if N > 1:
        corrected_scene_dur = (audio_dur + FADE_DUR * (N - 1)) / N
    else:
        corrected_scene_dur = audio_dur  # single clip — no xfade

    if N > 1 and corrected_scene_dur <= FADE_DUR:
        raise ValueError(
            f"corrected_scene_dur={corrected_scene_dur:.3f}s ≤ FADE_DUR={FADE_DUR}s "
            f"(audio={audio_dur:.1f}s, N={N}). Too many scenes for audio length."
        )

    has_lut   = LUT_PATH.exists()
    has_grain = GRAIN_PATH.exists()

    # ── Build inputs ──────────────────────────────────────────────────
    cmd: list[str] = ["ffmpeg", "-y"]

    # One -loop input per scene image, duration = corrected_scene_dur
    for img in clips:
        cmd.extend(["-loop", "1", "-t", f"{corrected_scene_dur:.6f}", "-i", str(img)])

    audio_idx = len(clips)
    cmd.extend(["-i", str(audio_path)])

    grain_idx: int | None = None
    if has_grain:
        cmd.extend(["-stream_loop", "-1", "-i", str(GRAIN_PATH)])
        grain_idx = audio_idx + 1

    # ── Build filter_complex ──────────────────────────────────────────
    parts: list[str] = []
    directions = ["in_center", "out_center", "in_left", "out_right"]

    # Per-clip zoompan — uses corrected_scene_dur so frame count is exact
    for i in range(N):
        parts.append(_per_clip_filter(i, corrected_scene_dur, directions[i % 4]))

    # xfade chain — each xfade overlaps previous clip's end with next's start
    if N == 1:
        prev = "clip0"
    else:
        prev = "clip0"
        for i in range(1, N):
            # Each clip contributes (corrected_scene_dur - FADE_DUR) of unique time.
            # xfade at position i starts at corrected_scene_dur*i - FADE_DUR*i.
            offset = corrected_scene_dur * i - FADE_DUR * i
            label = f"x{i}"
            parts.append(
                f"[{prev}][clip{i}]"
                f"xfade=transition=fade:duration={FADE_DUR}:offset={offset:.6f}"
                f"[{label}]"
            )
            prev = label

    # LUT color grading
    if has_lut:
        parts.append(f"[{prev}]lut3d=file={LUT_PATH}[lut]")
        prev = "lut"

    # Film grain overlay
    if grain_idx is not None:
        parts.append(
            f"[{grain_idx}:v]scale={TARGET_W}:{TARGET_H}:flags=lanczos,"
            f"format=yuv420p[g]"
        )
        parts.append(
            f"[{prev}][g]blend=all_mode=screen:all_opacity=0.18:shortest=1,"
            f"format=yuv420p[grained]"
        )
        prev = "grained"

    # Vignette (always on — cheap and dramatic)
    parts.append(f"[{prev}]vignette=PI/4[vfinal]")

    # Audio fades — anchored to real audio duration (from ffprobe), NOT to
    # the estimated total_dur. This ensures the fade-out happens at the
    # actual end of the audio track, not early due to xfade overlap math.
    audio_fade_out_start = max(0.0, audio_dur - AUDIO_FADE)
    parts.append(
        f"[{audio_idx}:a]"
        f"afade=t=in:st=0:d={AUDIO_FADE},"
        f"afade=t=out:st={audio_fade_out_start:.3f}:d={AUDIO_FADE}"
        f"[afinal]"
    )

    filter_complex = ";".join(parts)

    cmd.extend([
        "-filter_complex", filter_complex,
        "-map", "[vfinal]",
        "-map", "[afinal]",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(output_path),
    ])

    # ── Run ───────────────────────────────────────────────────────────
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # ffmpeg stderr can be huge — keep last 2000 chars for diagnosis
        raise RuntimeError(f"ffmpeg failed (rc={result.returncode}): {result.stderr[-2000:]}")

    return str(output_path)


def main() -> int:
    p = argparse.ArgumentParser(description="v32 cinematic composer")
    p.add_argument("--job-id",        required=True)
    p.add_argument("--video-type",    required=True)
    p.add_argument("--scene-count",   type=int,   required=True)
    p.add_argument("--scene-dur",     type=float, required=True)   # float since v32
    p.add_argument("--audio",         required=True)
    p.add_argument("--output",        required=True)
    p.add_argument("--jobs-dir",      default="/tmp/jobs")
    p.add_argument("--scene-offset",  type=int,   default=0,
                   help="First global image index for this render (sleep chapters)")
    p.add_argument("--failed-chunks", type=str,   default="",
                   help="Comma-separated global image indices to substitute")
    args = p.parse_args()

    failed: list[int] = []
    if args.failed_chunks.strip():
        try:
            failed = [int(x) for x in args.failed_chunks.split(",") if x.strip()]
        except ValueError as e:
            print(f"ERROR: --failed-chunks parse error: {e}", file=sys.stderr)
            return 1

    try:
        out = compose(
            args.job_id,
            args.video_type,
            args.scene_count,
            args.scene_dur,
            Path(args.audio),
            Path(args.output),
            Path(args.jobs_dir),
            scene_offset=args.scene_offset,
            failed_chunks=failed,
        )
        print(f"OK: {out}")
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
