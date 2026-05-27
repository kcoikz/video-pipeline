import asyncio
import copy
import json
import subprocess
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

app = FastAPI()

JOBS_DIR = Path("/tmp/jobs")
JOBS_DIR.mkdir(exist_ok=True)

CHATTERBOX_URL = "http://127.0.0.1:8004"
COMFY_URL = "http://127.0.0.1:8188"

# FLUX.1 Schnell workflow — 4 steps, 1024x576 (16:9)
FLUX_WORKFLOW = {
    "4":  {"class_type": "UNETLoader",      "inputs": {"unet_name": "flux1-schnell.safetensors", "weight_dtype": "fp8_e4m3fn"}},
    "10": {"class_type": "VAELoader",        "inputs": {"vae_name": "ae.safetensors"}},
    "11": {"class_type": "DualCLIPLoader",   "inputs": {"clip_name1": "clip_l.safetensors", "clip_name2": "t5xxl_fp8_e4m3fn.safetensors", "type": "flux", "device": "default"}},
    "5":  {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 576, "batch_size": 1}},
    "6":  {"class_type": "CLIPTextEncode",   "inputs": {"text": "", "clip": ["11", 0]}},
    "7":  {"class_type": "CLIPTextEncode",   "inputs": {"text": "", "clip": ["11", 0]}},
    "3":  {"class_type": "KSampler",         "inputs": {"model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0], "seed": 0, "steps": 4, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
    "8":  {"class_type": "VAEDecode",        "inputs": {"samples": ["3", 0], "vae": ["10", 0]}},
    "9":  {"class_type": "SaveImage",        "inputs": {"images": ["8", 0], "filename_prefix": "gen"}},
}


# ── Health ────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


# ── TTS — proxy to Chatterbox ─────────────────────────────────────

@app.post("/tts")
async def tts(request: Request):
    body = await request.json()
    body.setdefault("voice_mode", "predefined")
    body.setdefault("predefined_voice_id", "Emily.wav")
    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(f"{CHATTERBOX_URL}/tts", json=body)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)

    # Parse WAV header to get duration: data_size / (sample_rate * channels * bits/8)
    wav = resp.content
    duration_sec = 0.0
    if len(wav) > 44 and wav[:4] == b"RIFF":
        import struct
        sample_rate = struct.unpack_from("<I", wav, 24)[0]
        channels    = struct.unpack_from("<H", wav, 22)[0]
        bits        = struct.unpack_from("<H", wav, 34)[0]
        data_size   = struct.unpack_from("<I", wav, 40)[0]
        if sample_rate and channels and bits:
            duration_sec = round(data_size / (sample_rate * channels * (bits // 8)), 3)

    return StreamingResponse(
        iter([wav]),
        media_type="audio/wav",
        headers={
            "Content-Disposition": "attachment; filename=audio.wav",
            "X-Audio-Duration": str(duration_sec),
        },
    )


# ── Image generation — FLUX Schnell via ComfyUI ───────────────────

@app.post("/generate-image")
async def generate_image(request: Request):
    body = await request.json()

    workflow = copy.deepcopy(FLUX_WORKFLOW)
    workflow["6"]["inputs"]["text"]         = body.get("prompt", "")
    workflow["5"]["inputs"]["width"]        = body.get("width", 1024)
    workflow["5"]["inputs"]["height"]       = body.get("height", 576)
    workflow["3"]["inputs"]["steps"]        = body.get("steps", 4)
    workflow["3"]["inputs"]["seed"]         = body.get("seed", int(time.time()) % 2**32)

    async with httpx.AsyncClient(timeout=180) as client:
        # Submit to ComfyUI queue
        submit = await client.post(f"{COMFY_URL}/prompt", json={"prompt": workflow})
        if submit.status_code != 200:
            raise HTTPException(status_code=500, detail=f"ComfyUI submit failed: {submit.text}")
        prompt_id = submit.json()["prompt_id"]

        # Poll history until done (max 2 min)
        for _ in range(120):
            await asyncio.sleep(1)
            history = (await client.get(f"{COMFY_URL}/history/{prompt_id}")).json()
            if prompt_id not in history:
                continue
            outputs = history[prompt_id].get("outputs", {})
            for node_out in outputs.values():
                images = node_out.get("images", [])
                if not images:
                    continue
                fname = images[0]["filename"]
                img = await client.get(f"{COMFY_URL}/view", params={"filename": fname, "type": "output"})
                return StreamingResponse(
                    iter([img.content]),
                    media_type="image/png",
                    headers={"Content-Disposition": f"attachment; filename={fname}"},
                )
            raise HTTPException(status_code=500, detail="ComfyUI returned no images")

    raise HTTPException(status_code=504, detail="ComfyUI timed out after 120s")


# ── TTS + save (no binary in n8n) ─────────────────────────────────

@app.post("/tts-upload/{job_id}/{folder}")
async def tts_upload(job_id: str, folder: str, request: Request):
    import struct
    body = await request.json()
    body.setdefault("voice_mode", "predefined")
    body.setdefault("predefined_voice_id", "Emily.wav")
    async with httpx.AsyncClient(timeout=600) as client:
        resp = await client.post(f"{CHATTERBOX_URL}/tts", json=body)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
    wav = resp.content
    dest = JOBS_DIR / job_id / folder
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "en_audio.wav").write_bytes(wav)
    duration_sec = 0.0
    if len(wav) > 44 and wav[:4] == b"RIFF":
        sample_rate = struct.unpack_from("<I", wav, 24)[0]
        channels    = struct.unpack_from("<H", wav, 22)[0]
        bits        = struct.unpack_from("<H", wav, 34)[0]
        data_size   = struct.unpack_from("<I", wav, 40)[0]
        if sample_rate and channels and bits:
            duration_sec = round(data_size / (sample_rate * channels * (bits // 8)), 3)
    scene_count = max(1, int(duration_sec / 20))
    return {"success": True, "duration_sec": duration_sec, "scene_count": scene_count}


# ── Generate image + save (no binary in n8n) ──────────────────────

@app.post("/generate-and-save/{job_id}/{folder}/{filename}")
async def generate_and_save(job_id: str, folder: str, filename: str, request: Request):
    body = await request.json()
    wf = copy.deepcopy(FLUX_WORKFLOW)
    wf["6"]["inputs"]["text"]   = body.get("prompt", "")
    wf["5"]["inputs"]["width"]  = body.get("width", 1024)
    wf["5"]["inputs"]["height"] = body.get("height", 576)
    wf["3"]["inputs"]["steps"]  = body.get("steps", 4)
    wf["3"]["inputs"]["seed"]   = body.get("seed", int(time.time()) % 2**32)
    async with httpx.AsyncClient(timeout=180) as client:
        submit = await client.post(f"{COMFY_URL}/prompt", json={"prompt": wf})
        if submit.status_code != 200:
            raise HTTPException(status_code=500, detail=f"ComfyUI submit failed: {submit.text}")
        prompt_id = submit.json()["prompt_id"]
        for _ in range(120):
            await asyncio.sleep(1)
            history = (await client.get(f"{COMFY_URL}/history/{prompt_id}")).json()
            if prompt_id not in history:
                continue
            for node_out in history[prompt_id].get("outputs", {}).values():
                images = node_out.get("images", [])
                if not images:
                    continue
                fname = images[0]["filename"]
                img = await client.get(f"{COMFY_URL}/view", params={"filename": fname, "type": "output"})
                dest = JOBS_DIR / job_id / folder
                dest.mkdir(parents=True, exist_ok=True)
                (dest / filename).write_bytes(img.content)
                return {"success": True}
            raise HTTPException(status_code=500, detail="ComfyUI returned no images")
    raise HTTPException(status_code=504, detail="ComfyUI timed out")


# ── File upload ───────────────────────────────────────────────────

@app.post("/upload/{job_id}/{folder}/{filename}")
async def upload_file(job_id: str, folder: str, filename: str, request: Request):
    dest = JOBS_DIR / job_id / folder
    dest.mkdir(parents=True, exist_ok=True)
    (dest / filename).write_bytes(await request.body())
    return {"success": True}


# ── Render — FFmpeg slideshow ─────────────────────────────────────

@app.post("/render/{job_id}")
async def render_video(job_id: str, request: Request):
    body         = await request.json()
    video_type   = body.get("type", "short_en")       # "short_en" | "sleep_en"
    scene_count  = int(body.get("scene_count", 9))
    scene_dur    = int(body.get("scene_duration", 20))

    folder     = "short" if "short" in video_type else "sleep"
    job_path   = JOBS_DIR / job_id / folder
    audio_path = job_path / "en_audio.wav"
    output_mp4 = JOBS_DIR / job_id / f"{video_type}.mp4"
    concat_txt = JOBS_DIR / job_id / f"{video_type}_concat.txt"

    if not audio_path.exists():
        raise HTTPException(status_code=400, detail=f"Audio not found: {audio_path}")

    # Build FFmpeg concat file
    lines = []
    for i in range(scene_count):
        img = job_path / f"scene_{str(i).zfill(3)}.jpg"
        if not img.exists():
            # Try .png fallback (ComfyUI saves PNG by default)
            img = job_path / f"scene_{str(i).zfill(3)}.png"
        if not img.exists():
            raise HTTPException(status_code=400, detail=f"Missing scene image {i}")
        lines.append(f"file '{img}'\nduration {scene_dur}")
    # Concat demuxer requires last file repeated without duration
    lines.append(f"file '{img}'")
    concat_txt.write_text("\n".join(lines))

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", str(concat_txt),
        "-i", str(audio_path),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        str(output_mp4),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=f"FFmpeg failed: {result.stderr[-600:]}")

    return {
        "render_id": f"{job_id}_{video_type}",
        "status": "done",
        "download_url": f"/download/{job_id}/{video_type}",
    }


# ── Video download ────────────────────────────────────────────────

@app.get("/download/{job_id}/{video_type}")
async def download_video(job_id: str, video_type: str):
    path = JOBS_DIR / job_id / f"{video_type}.mp4"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(str(path), media_type="video/mp4", filename=f"{job_id}_{video_type}.mp4")
