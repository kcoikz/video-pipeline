import asyncio
import copy
import json
import os
import re
import signal
import struct
import subprocess
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

app = FastAPI()

JOBS_DIR = Path("/tmp/jobs")
JOBS_DIR.mkdir(exist_ok=True)

# OAuth config — persisted on volume so it survives pod restarts.
# Populated via POST /admin/save-oauth-config the first time.
OAUTH_CONFIG_PATH = Path("/workspace/.oauth_config.json")
if not OAUTH_CONFIG_PATH.parent.exists():
    OAUTH_CONFIG_PATH = Path("/tmp/.oauth_config.json")

CHATTERBOX_URL = "http://127.0.0.1:8004"
COMFY_URL = "http://127.0.0.1:8188"

# Track PIDs of managed services so we can restart them
_service_pids: dict[str, int] = {}

# Active job tracker — watchdog calls /status to check if pipeline is busy
_active_jobs: dict = {}

# Background TTS tasks: "{job_id}/{folder}" -> {"status": "processing"/"done"/"error", ...}
_tts_tasks: dict[str, dict] = {}

# Background render tasks: "{job_id}/{video_type}" -> {"status": ..., ...}
_render_tasks: dict[str, dict] = {}

# Background Drive upload tasks: job_id -> {"status": ..., ...}
_drive_tasks: dict[str, dict] = {}

# Background image batch tasks: "{job_id}/{folder}" -> {"status": ..., "total": N, "done": N, "errors": [...]}
_image_batch_tasks: dict[str, dict] = {}

# v31: Background key-art generation tasks: job_id -> {"status": ..., "path": ..., "error": ...}
_key_art_tasks: dict[str, dict] = {}

# Max concurrent ComfyUI image generations (prevents pod overload / Cloudflare 504s)
IMAGE_BATCH_CONCURRENCY = 4

def _job_start(job_id: str):
    _active_jobs[job_id] = time.time()

def _job_end(job_id: str):
    _active_jobs.pop(job_id, None)

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


# ── Health & Status ───────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/status")
async def status():
    jobs = [{"job_id": k, "running_sec": round(time.time() - v)} for k, v in _active_jobs.items()]
    return {"pipeline_active": len(jobs) > 0, "active_jobs": jobs}


@app.get("/services-status")
async def services_status():
    results = {}
    for name, url in [("chatterbox", CHATTERBOX_URL), ("comfyui", COMFY_URL)]:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(f"{url}/health")
                results[name] = {"reachable": True, "status_code": r.status_code}
        except Exception as e:
            results[name] = {"reachable": False, "error": str(e)}
    return results


@app.post("/restart-tts")
async def restart_tts():
    # Kill existing Chatterbox process on port 8004
    kill_result = subprocess.run(
        ["bash", "-c", "fuser -k 8004/tcp; sleep 1; echo killed"],
        capture_output=True, text=True
    )
    # Read HF_HOME from env so models load from workspace cache
    env = {**os.environ, "HF_HOME": "/workspace/.hf_cache"}
    proc = subprocess.Popen(
        ["python", "server.py"],
        cwd="/root/Chatterbox-TTS-Server",
        env=env,
        stdout=open("/workspace/logs/chatterbox.log", "a"),
        stderr=subprocess.STDOUT,
    )
    _service_pids["chatterbox"] = proc.pid
    return {"started": True, "pid": proc.pid, "kill_output": kill_result.stdout.strip()}


@app.post("/restart-server")
async def restart_server():
    """Triggers uvicorn restart so start.sh's loop pulls fresh code from git.
    Call this after pushing new code — no Docker rebuild or pod recreation needed."""
    async def _exit_soon():
        await asyncio.sleep(1)
        os._exit(0)
    asyncio.create_task(_exit_soon())
    return {"status": "restarting", "note": "start.sh will git pull and restart uvicorn"}


@app.get("/logs/{service}")
async def get_logs(service: str, lines: int = 100):
    log_path = Path(f"/workspace/logs/{service}.log")
    if not log_path.exists():
        raise HTTPException(status_code=404, detail=f"Log not found: {log_path}")
    result = subprocess.run(["tail", f"-{lines}", str(log_path)], capture_output=True, text=True)
    return {"log": result.stdout, "path": str(log_path)}


# ── TTS — proxy to Chatterbox (legacy, returns binary) ───────────

@app.post("/tts")
async def tts(request: Request):
    body = await request.json()
    body.setdefault("voice_mode", "predefined")
    body.setdefault("predefined_voice_id", "Emily.wav")
    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(f"{CHATTERBOX_URL}/tts", json=body)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
    wav = resp.content
    duration_sec = _wav_duration(wav)
    return StreamingResponse(
        iter([wav]),
        media_type="audio/wav",
        headers={
            "Content-Disposition": "attachment; filename=audio.wav",
            "X-Audio-Duration": str(duration_sec),
        },
    )


# ── TTS + save (pipeline uses this — async: start then poll) ──────

_SLEEP_CHAPTER_RE = re.compile(r'^sleep_ch([1-5])$')


def _is_sleep_chapter(folder: str) -> bool:
    """True for v32 per-chapter sleep folders: sleep_ch1 .. sleep_ch5."""
    return bool(_SLEEP_CHAPTER_RE.match(folder))


async def _run_tts_background(job_id: str, folder: str, body: dict, task_key: str):
    """Background coroutine: calls Chatterbox, saves WAV, updates _tts_tasks.

    v32: sleep chapters (folder='sleep_ch1'..'sleep_ch5') are saved to
    {JOBS_DIR}/{job_id}/audio/sleep_ch{N}.wav instead of the old
    {job_id}/{folder}/en_audio.wav, so each chapter has its own audio file
    for per-chapter render alignment.
    """
    try:
        body.setdefault("voice_mode", "predefined")
        body.setdefault("predefined_voice_id", "Emily.wav")
        async with httpx.AsyncClient(timeout=1800) as client:
            resp = await client.post(f"{CHATTERBOX_URL}/tts", json=body)
            if resp.status_code != 200:
                err_msg = resp.text[:500]
                print(f"[TTS_ERR] job={job_id} folder={folder} chatterbox_status={resp.status_code} err={err_msg!r}")
                _tts_tasks[task_key] = {"status": "error", "error": err_msg}
                return
        wav = resp.content

        # v32: sleep chapters → {job_id}/audio/sleep_ch{N}.wav
        if _is_sleep_chapter(folder):
            dest_dir = JOBS_DIR / job_id / "audio"
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_file = dest_dir / f"{folder}.wav"
        else:
            dest_dir = JOBS_DIR / job_id / folder
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_file = dest_dir / "en_audio.wav"

        dest_file.write_bytes(wav)
        duration_sec = _wav_duration(wav)
        scene_count = max(1, int(duration_sec / 20))
        _tts_tasks[task_key] = {
            "status": "done",
            "success": True,
            "duration_sec": duration_sec,
            "scene_count": scene_count,
        }
    except Exception as e:
        print(f"[TTS_ERR] job={job_id} folder={folder} exc={e}")
        _tts_tasks[task_key] = {"status": "error", "error": str(e)}
    finally:
        _job_end(job_id)


@app.post("/tts-upload/{job_id}/{folder}")
async def tts_upload(job_id: str, folder: str, request: Request):
    """Legacy sync endpoint — kept for compatibility; use /tts-start + /tts-poll instead."""
    body = await request.json()
    task_key = f"{job_id}/{folder}"
    # Reuse async machinery
    _tts_tasks[task_key] = {"status": "processing"}
    _job_start(job_id)
    asyncio.create_task(_run_tts_background(job_id, folder, body, task_key))
    # Wait up to 90 s (safe under 120s Cloudflare limit) then 503 if not done
    for _ in range(18):
        await asyncio.sleep(5)
        t = _tts_tasks.get(task_key, {})
        if t.get("status") == "done":
            return {k: v for k, v in t.items() if k != "status"}
        if t.get("status") == "error":
            raise HTTPException(status_code=500, detail=t.get("error", "TTS failed"))
    raise HTTPException(status_code=503, detail="TTS still processing — use /tts-poll")


@app.post("/tts-start/{job_id}/{folder}")
async def tts_start(job_id: str, folder: str, request: Request):
    """Start TTS in background. Returns immediately. Poll /tts-poll/{job_id}/{folder}.

    v32: folder='sleep' is no longer valid — use sleep_ch1..sleep_ch5 instead.
    """
    # v32: old monolithic sleep endpoint removed to avoid confusion
    if folder == "sleep":
        raise HTTPException(
            status_code=404,
            detail="Old /tts-start/{job_id}/sleep endpoint removed in v32. "
                   "Use /tts-start/{job_id}/sleep_ch{N} (N=1..5) for per-chapter TTS.",
        )
    body = await request.json()
    task_key = f"{job_id}/{folder}"
    if _tts_tasks.get(task_key, {}).get("status") == "processing":
        return {"status": "already_running", "task_key": task_key}
    _tts_tasks[task_key] = {"status": "processing"}
    _job_start(job_id)
    asyncio.create_task(_run_tts_background(job_id, folder, body, task_key))
    return {"status": "started", "task_key": task_key}


@app.get("/tts-poll/{job_id}/{folder}")
async def tts_poll(job_id: str, folder: str):
    """Poll TTS background task.
    Returns 200 when done, 503 when still processing (triggers n8n retryOnFail),
    500 on error."""
    task_key = f"{job_id}/{folder}"
    t = _tts_tasks.get(task_key)
    if t is None:
        raise HTTPException(status_code=404, detail=f"No TTS task: {task_key}")
    if t["status"] == "done":
        return {"status": "done", **{k: v for k, v in t.items() if k != "status"}}
    if t["status"] == "error":
        raise HTTPException(status_code=500, detail=t.get("error", "TTS failed"))
    # Still processing — 503 so n8n HTTP Request retryOnFail keeps polling
    raise HTTPException(status_code=503, detail="TTS still processing")


# ── Image generation — FLUX Schnell via ComfyUI (legacy) ─────────

@app.post("/generate-image")
async def generate_image(request: Request):
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
                return StreamingResponse(
                    iter([img.content]),
                    media_type="image/png",
                    headers={"Content-Disposition": f"attachment; filename={fname}"},
                )
            raise HTTPException(status_code=500, detail="ComfyUI returned no images")
    raise HTTPException(status_code=504, detail="ComfyUI timed out after 120s")


# ── Generate image + save (pipeline uses this — returns JSON) ─────

@app.post("/generate-and-save/{job_id}/{folder}/{filename}")
async def generate_and_save(job_id: str, folder: str, filename: str, request: Request):
    _job_start(job_id)
    try:
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
    finally:
        _job_end(job_id)


# ── Batch image generation (async: start + poll, bounded concurrency) ─

# v31 STEP 4: img2img style anchoring. When key_art.png exists for the job, we
# build a workflow that VAE-encodes the key-art and feeds it as the latent
# starting point to KSampler with denoise<1.0. The image content is overwritten
# by the text prompt but the VISUAL STYLE (palette, lighting, texture, era) is
# preserved across all frames — solves "different artistic styles between frames".
#
# Schnell is distilled so denoise tuning is tricky: too high (>0.9) → no style
# preservation; too low (<0.7) → content stays too close to key_art. 0.85 is
# the empirical sweet spot at 6 steps.
IMG2IMG_STEPS    = 6
IMG2IMG_DENOISE  = 0.85
COMFY_INPUT_DIR  = Path("/root/ComfyUI/input")


def _build_img2img_workflow(prompt: str, width: int, height: int,
                              steps: int, seed: int, keyart_ref: str) -> dict:
    """FLUX img2img workflow conditioned on a key-art reference image.
    keyart_ref is the FILENAME (not path) inside ComfyUI's input directory."""
    return {
        "4":  {"class_type": "UNETLoader",      "inputs": {"unet_name": "flux1-schnell.safetensors", "weight_dtype": "fp8_e4m3fn"}},
        "10": {"class_type": "VAELoader",        "inputs": {"vae_name": "ae.safetensors"}},
        "11": {"class_type": "DualCLIPLoader",   "inputs": {"clip_name1": "clip_l.safetensors", "clip_name2": "t5xxl_fp8_e4m3fn.safetensors", "type": "flux", "device": "default"}},
        "20": {"class_type": "LoadImage",        "inputs": {"image": keyart_ref}},
        "22": {"class_type": "ImageScale",       "inputs": {"image": ["20", 0], "upscale_method": "lanczos", "width": width, "height": height, "crop": "center"}},
        "21": {"class_type": "VAEEncode",        "inputs": {"pixels": ["22", 0], "vae": ["10", 0]}},
        "6":  {"class_type": "CLIPTextEncode",   "inputs": {"text": prompt, "clip": ["11", 0]}},
        "7":  {"class_type": "CLIPTextEncode",   "inputs": {"text": "", "clip": ["11", 0]}},
        "3":  {"class_type": "KSampler",         "inputs": {"model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["21", 0], "seed": seed, "steps": steps, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": IMG2IMG_DENOISE}},
        "8":  {"class_type": "VAEDecode",        "inputs": {"samples": ["3", 0], "vae": ["10", 0]}},
        "9":  {"class_type": "SaveImage",        "inputs": {"images": ["8", 0], "filename_prefix": "gen"}},
    }


async def _gen_one_image(client: httpx.AsyncClient, job_id: str, folder: str,
                          idx: int, filename: str, prompt: str,
                          width: int, height: int, steps: int, task_key: str,
                          keyart_ref: str | None = None) -> bool:
    """Generate ONE image via ComfyUI and save to disk.

    Returns True on success, False on any failure.
    Records failures into the batch task's errors list (never raises).
    Includes structured log lines for grep-able debugging.

    v31 Step 4: if keyart_ref is set (filename inside ComfyUI's input dir),
    builds an img2img workflow conditioned on that reference image. Otherwise
    falls back to plain text-to-image with FLUX_WORKFLOW.
    v32: now returns bool so caller can track done vs failed counts separately.
    """
    def _record_error(msg: str) -> bool:
        full_msg = msg[:400]
        _image_batch_tasks[task_key]["errors"].append({"filename": filename, "error": full_msg})
        print(f"[IMG_ERR] job={job_id} folder={folder} chunk_id={idx} file={filename} err={full_msg!r}")
        return False

    try:
        seed = int(time.time() * 1000) % 2**32
        if keyart_ref:
            wf = _build_img2img_workflow(prompt, width, height, IMG2IMG_STEPS, seed, keyart_ref)
        else:
            wf = copy.deepcopy(FLUX_WORKFLOW)
            wf["6"]["inputs"]["text"]   = prompt
            wf["5"]["inputs"]["width"]  = width
            wf["5"]["inputs"]["height"] = height
            wf["3"]["inputs"]["steps"]  = steps
            wf["3"]["inputs"]["seed"]   = seed
        submit = await client.post(f"{COMFY_URL}/prompt", json={"prompt": wf})
        if submit.status_code != 200:
            return _record_error(f"comfy submit {submit.status_code}: {submit.text[:200]}")
        prompt_id = submit.json().get("prompt_id")
        if not prompt_id:
            return _record_error(f"no prompt_id in: {submit.text[:200]}")
        # Poll ComfyUI history up to 3 min per image
        for _ in range(180):
            await asyncio.sleep(1)
            try:
                history = (await client.get(f"{COMFY_URL}/history/{prompt_id}")).json()
            except Exception:
                continue
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
                return True
        return _record_error("comfy timeout after 180s")
    except Exception as e:
        return _record_error(str(e)[:300])


async def _run_image_batch_background(job_id: str, folder: str, prompts: list,
                                       width: int, height: int, steps: int,
                                       task_key: str) -> None:
    """Generate all prompts with bounded concurrency. Updates _image_batch_tasks.

    v31 Step 4: if /workspace/key_art/{job_id}.png exists, copies it ONCE into
    ComfyUI's input directory and switches every generation to img2img mode
    using it as the style anchor."""
    semaphore = asyncio.Semaphore(IMAGE_BATCH_CONCURRENCY)

    # v31 Step 4: prepare key-art reference (once per batch).
    keyart_ref: str | None = None
    keyart_src = _key_art_volume_path(job_id)
    if keyart_src.exists() and keyart_src.stat().st_size > 0:
        try:
            COMFY_INPUT_DIR.mkdir(parents=True, exist_ok=True)
            keyart_ref = f"keyart_{job_id}.png"
            ref_path = COMFY_INPUT_DIR / keyart_ref
            if not ref_path.exists():
                ref_path.write_bytes(keyart_src.read_bytes())
        except Exception as e:
            # If copy fails, fall back to text-to-image — log but don't fail batch
            _image_batch_tasks[task_key].setdefault("warnings", []).append(
                {"stage": "keyart_copy", "error": str(e)[:200]}
            )
            keyart_ref = None

    async def gen_one(idx: int, item: dict):
        async with semaphore:
            async with httpx.AsyncClient(timeout=600) as client:
                ok = await _gen_one_image(
                    client, job_id, folder,
                    idx, item["filename"], item["prompt"],
                    width, height, steps, task_key,
                    keyart_ref=keyart_ref,
                )
        # v32: only count successful generations; track failed indices separately
        if ok:
            _image_batch_tasks[task_key]["done"] += 1
        else:
            _image_batch_tasks[task_key]["failed_indices"].append(idx)

    try:
        await asyncio.gather(*[gen_one(i, item) for i, item in enumerate(prompts)])
        _image_batch_tasks[task_key]["status"] = "done"
        _image_batch_tasks[task_key]["mode"] = "img2img" if keyart_ref else "txt2img"
    except Exception as e:
        _image_batch_tasks[task_key]["status"] = "error"
        _image_batch_tasks[task_key]["error"] = str(e)
    finally:
        _job_end(job_id)


@app.post("/generate-batch-start/{job_id}/{folder}")
async def generate_batch_start(job_id: str, folder: str, request: Request):
    """Start batch image generation in background. Returns immediately.

    Body: {
      "prompts": [{"filename": "scene_000.png", "prompt": "..."}, ...],
      "width": 1024, "height": 576, "steps": 4
    }
    Then poll /generate-batch-poll/{job_id}/{folder} until status=done.
    """
    body = await request.json()
    prompts = body.get("prompts", [])
    if not prompts or not isinstance(prompts, list):
        raise HTTPException(status_code=400, detail="prompts must be non-empty list")
    for p in prompts:
        if not isinstance(p, dict) or "filename" not in p or "prompt" not in p:
            raise HTTPException(status_code=400, detail="each prompt needs {filename, prompt}")

    width  = body.get("width", 1024)
    height = body.get("height", 576)
    steps  = body.get("steps", 4)

    task_key = f"{job_id}/{folder}"
    existing = _image_batch_tasks.get(task_key)
    if existing and existing.get("status") == "processing":
        return {
            "status": "already_running",
            "task_key": task_key,
            "total": existing["total"],
            "done": existing["done"],
        }

    _image_batch_tasks[task_key] = {
        "status": "processing",
        "total": len(prompts),
        "done": 0,
        "errors": [],
        "failed_indices": [],   # v32: 0-based indices of failed chunks
    }
    _job_start(job_id)
    asyncio.create_task(
        _run_image_batch_background(job_id, folder, prompts, width, height, steps, task_key)
    )
    return {"status": "started", "task_key": task_key, "total": len(prompts)}


@app.get("/generate-batch-poll/{job_id}/{folder}")
async def generate_batch_poll(job_id: str, folder: str):
    """Poll image batch task.
    200 when done, 503 while processing (triggers n8n loop), 500 on error."""
    task_key = f"{job_id}/{folder}"
    t = _image_batch_tasks.get(task_key)
    if t is None:
        raise HTTPException(status_code=404, detail=f"No image batch task: {task_key}")
    if t["status"] == "done":
        return {
            "status": "done",
            "total": t["total"],
            "done": t["done"],
            "failed": sorted(t.get("failed_indices", [])),  # v32: always present, even if empty
            "errors": t["errors"],
        }
    if t["status"] == "error":
        raise HTTPException(status_code=500, detail=t.get("error", "Image batch failed"))
    raise HTTPException(
        status_code=503,
        detail=f"Image batch processing: {t['done']}/{t['total']}",
    )


# ── Key-art generation (v31): single hero image, higher quality ─────
# Schnell is a distilled model so higher step counts don't help much, but a
# larger resolution gives noticeably more detail. We render the key-art at
# 1536x864 (50% wider/taller than scene frames) and persist it to the volume
# so subsequent stages (Flux Redux conditioning, ffmpeg branding, archival)
# can load it. Idempotent: if key_art.png already exists for this job_id,
# we return immediately.

KEY_ART_WIDTH  = 1536
KEY_ART_HEIGHT = 864
KEY_ART_STEPS  = 8   # Schnell sweet spot — diminishing returns above 8

def _key_art_path(job_id: str) -> Path:
    return JOBS_DIR / job_id / "key_art.png"

def _key_art_volume_path(job_id: str) -> Path:
    return Path("/workspace/key_art") / f"{job_id}.png"


async def _run_key_art_background(job_id: str, prompt: str, negative: str) -> None:
    """Generate ONE high-quality hero image via ComfyUI. Saves to both ephemeral
    JOBS_DIR (for download endpoints) and /workspace/key_art (for persistence
    across pod restarts and downstream stages)."""
    try:
        wf = copy.deepcopy(FLUX_WORKFLOW)
        wf["6"]["inputs"]["text"]   = prompt
        wf["7"]["inputs"]["text"]   = negative      # FLUX negative slot
        wf["5"]["inputs"]["width"]  = KEY_ART_WIDTH
        wf["5"]["inputs"]["height"] = KEY_ART_HEIGHT
        wf["3"]["inputs"]["steps"]  = KEY_ART_STEPS
        wf["3"]["inputs"]["seed"]   = int(time.time() * 1000) % 2**32

        async with httpx.AsyncClient(timeout=600) as client:
            submit = await client.post(f"{COMFY_URL}/prompt", json={"prompt": wf})
            if submit.status_code != 200:
                _key_art_tasks[job_id] = {"status": "error", "error": f"comfy submit {submit.status_code}: {submit.text[:300]}"}
                return
            prompt_id = submit.json().get("prompt_id")
            if not prompt_id:
                _key_art_tasks[job_id] = {"status": "error", "error": f"no prompt_id: {submit.text[:200]}"}
                return

            for _ in range(300):  # up to 5 min for key-art (it's higher quality)
                await asyncio.sleep(1)
                try:
                    history = (await client.get(f"{COMFY_URL}/history/{prompt_id}")).json()
                except Exception:
                    continue
                if prompt_id not in history:
                    continue
                for node_out in history[prompt_id].get("outputs", {}).values():
                    images = node_out.get("images", [])
                    if not images:
                        continue
                    fname = images[0]["filename"]
                    img = await client.get(f"{COMFY_URL}/view", params={"filename": fname, "type": "output"})

                    # Save to ephemeral location
                    ephemeral = _key_art_path(job_id)
                    ephemeral.parent.mkdir(parents=True, exist_ok=True)
                    ephemeral.write_bytes(img.content)

                    # Persist to volume
                    persistent = _key_art_volume_path(job_id)
                    persistent.parent.mkdir(parents=True, exist_ok=True)
                    persistent.write_bytes(img.content)

                    _key_art_tasks[job_id] = {
                        "status": "done",
                        "path": str(ephemeral),
                        "volume_path": str(persistent),
                        "download_url": f"/key-art-image/{job_id}",
                    }
                    return
            _key_art_tasks[job_id] = {"status": "error", "error": "comfy timeout after 300s"}
    except Exception as e:
        _key_art_tasks[job_id] = {"status": "error", "error": str(e)[:300]}
    finally:
        _job_end(job_id)


@app.post("/key-art-start/{job_id}")
async def key_art_start(job_id: str, request: Request):
    """Start key-art generation in background. Returns immediately.
    Body: {"prompt": "...", "negative_style": "..."}
    Poll /key-art-poll/{job_id}.
    Idempotent: if key_art.png already exists, returns done immediately."""
    body = await request.json()
    prompt   = body.get("prompt", "").strip()
    _neg_raw = body.get("negative_style", "")
    if isinstance(_neg_raw, list):
        negative = ", ".join(str(x) for x in _neg_raw)
    else:
        negative = str(_neg_raw).strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")

    # Idempotency: check persistent volume first (survives pod restarts)
    persistent = _key_art_volume_path(job_id)
    if persistent.exists() and persistent.stat().st_size > 0:
        _key_art_tasks[job_id] = {
            "status": "done",
            "path": str(persistent),
            "volume_path": str(persistent),
            "download_url": f"/key-art-image/{job_id}",
            "cached": True,
        }
        return {"status": "already_done", "cached": True}

    if _key_art_tasks.get(job_id, {}).get("status") == "processing":
        return {"status": "already_running"}

    _key_art_tasks[job_id] = {"status": "processing"}
    _job_start(job_id)
    asyncio.create_task(_run_key_art_background(job_id, prompt, negative))
    return {"status": "started"}


@app.get("/key-art-poll/{job_id}")
async def key_art_poll(job_id: str):
    """Poll key-art generation. 200 done, 503 processing, 500 error."""
    t = _key_art_tasks.get(job_id)
    if t is None:
        raise HTTPException(status_code=404, detail=f"No key-art task: {job_id}")
    if t["status"] == "done":
        return {"status": "done", **{k: v for k, v in t.items() if k != "status"}}
    if t["status"] == "error":
        raise HTTPException(status_code=500, detail=t.get("error", "Key-art failed"))
    raise HTTPException(status_code=503, detail="Key-art still rendering")


@app.get("/key-art-image/{job_id}")
async def key_art_image(job_id: str):
    """Serve the key-art PNG for download (so Drive/Flux Redux can fetch it)."""
    path = _key_art_volume_path(job_id)
    if not path.exists():
        path = _key_art_path(job_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No key-art for {job_id}")
    return FileResponse(str(path), media_type="image/png", filename=f"{job_id}_key_art.png")


# ── File upload ───────────────────────────────────────────────────

@app.post("/upload/{job_id}/{folder}/{filename}")
async def upload_file(job_id: str, folder: str, filename: str, request: Request):
    dest = JOBS_DIR / job_id / folder
    dest.mkdir(parents=True, exist_ok=True)
    (dest / filename).write_bytes(await request.body())
    return {"success": True}


# ── Render — FFmpeg slideshow (async: start + poll) ───────────────

def _locate_compose_script() -> Path:
    """Find compose.py — next to server.py, or fall back to workspace path."""
    p = Path(__file__).resolve().parent / "compose.py"
    if not p.exists():
        p = Path("/workspace/code/runpod-server/compose.py")
    return p


async def _run_render_background(
    job_id: str,
    video_type: str,
    scene_count: int,
    scene_dur: float,           # v32: float for exact audio sync
    task_key: str,
    failed_chunks: list[int] | None = None,
):
    """Run the cinematic composer (compose.py) in a thread (short video or legacy sleep).

    v31 Step 6: delegates to compose.py (zoompan + xfade + lut3d + grain + vignette + afade).
    v32: scene_dur is float; failed_chunks substituted in compose.py.
    """
    import shutil
    folder     = "short" if "short" in video_type else "sleep"
    job_path   = JOBS_DIR / job_id / folder
    audio_path = job_path / "en_audio.wav"
    output_mp4 = JOBS_DIR / job_id / f"{video_type}.mp4"
    compose_script = _locate_compose_script()

    try:
        if not audio_path.exists():
            _render_tasks[task_key] = {"status": "error", "error": f"Audio not found: {audio_path}"}
            return
        if not compose_script.exists():
            _render_tasks[task_key] = {"status": "error", "error": f"compose.py not found at {compose_script}"}
            return

        cmd = [
            "python3", str(compose_script),
            "--job-id",      job_id,
            "--video-type",  video_type,
            "--scene-count", str(scene_count),
            "--scene-dur",   str(scene_dur),
            "--audio",       str(audio_path),
            "--output",      str(output_mp4),
            "--jobs-dir",    str(JOBS_DIR),
        ]
        if failed_chunks:
            cmd.extend(["--failed-chunks", ",".join(str(x) for x in failed_chunks)])

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: subprocess.run(cmd, capture_output=True, text=True)
        )
        if result.returncode != 0:
            err_tail = (result.stderr or result.stdout or "")[-1200:]
            _render_tasks[task_key] = {"status": "error", "error": f"compose.py failed: {err_tail}"}
            return

        volume_out = Path("/workspace/output") / job_id
        volume_out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(output_mp4), str(volume_out / f"{video_type}.mp4"))

        _render_tasks[task_key] = {
            "status": "done",
            "render_id": f"{job_id}_{video_type}",
            "download_url": f"/download/{job_id}/{video_type}",
            "volume_path": f"/workspace/output/{job_id}/{video_type}.mp4",
        }
    except Exception as e:
        _render_tasks[task_key] = {"status": "error", "error": str(e)}
    finally:
        _job_end(job_id)


async def _run_render_sleep_background(
    job_id: str,
    audio_files: list[str],     # e.g. ["sleep_ch1.wav", ..., "sleep_ch5.wav"]
    chapters: list[dict],       # [{duration_sec, scene_count, scene_offset}, ...]
    failed_chunks: list[int],   # global image indices to substitute
    task_key: str,
):
    """Sleep render: concat all chapter WAVs → single compose.py call.

    Rule 1 (audio is master): compose.py uses ffprobe to measure the real
    audio duration and back-calculates per_image = audio_dur / N so the
    video is always exactly as long as the audio. No frozen frames, no
    silent tails.

    Rule 2 (no chapter gaps): all chapter WAVs are concatenated into one
    continuous track with ffmpeg concat demuxer (stream-copy, lossless).
    compose.py sees one unbroken audio file → one unbroken video with
    smooth xfade transitions throughout, no inter-chapter pauses.
    """
    import shutil

    audio_dir = JOBS_DIR / job_id / "audio"
    compose_script = _locate_compose_script()

    if not compose_script.exists():
        _render_tasks[task_key] = {"status": "error", "error": f"compose.py not found at {compose_script}"}
        _job_end(job_id)
        return

    try:
        loop = asyncio.get_event_loop()

        # ── Step 1: verify all chapter audio files exist ──────────────
        chapter_wavs: list[Path] = []
        for i, fname in enumerate(audio_files, start=1):
            wav = audio_dir / fname
            if not wav.exists():
                _render_tasks[task_key] = {
                    "status": "error",
                    "error": f"Chapter {i} audio not found: {wav}",
                }
                return
            chapter_wavs.append(wav)

        # ── Step 2: concatenate chapter WAVs into one continuous track ─
        combined_audio = audio_dir / "sleep_combined.wav"
        concat_list    = audio_dir / "sleep_concat_list.txt"
        concat_list.write_text(
            "\n".join(f"file '{w.resolve()}'" for w in chapter_wavs)
        )
        concat_audio_cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", str(concat_list),
            "-c", "copy",
            str(combined_audio),
        ]
        print(f"[RENDER] job={job_id} sleep: concatenating {len(chapter_wavs)} chapter WAVs")
        result = await loop.run_in_executor(
            None, lambda: subprocess.run(concat_audio_cmd, capture_output=True, text=True)
        )
        if result.returncode != 0:
            err = (result.stderr or result.stdout or "")[-800:]
            _render_tasks[task_key] = {"status": "error", "error": f"audio concat failed: {err}"}
            return

        # ── Step 3: single compose.py call with all images + combined audio
        total_scene_count = sum(int(c["scene_count"]) for c in chapters)
        output_mp4        = JOBS_DIR / job_id / "sleep_en.mp4"

        cmd = [
            "python3", str(compose_script),
            "--job-id",      job_id,
            "--video-type",  "sleep_en",
            "--scene-count", str(total_scene_count),
            "--scene-dur",   "8.0",        # hint only — overridden by ffprobe inside compose.py
            "--scene-offset", "0",
            "--audio",       str(combined_audio),
            "--output",      str(output_mp4),
            "--jobs-dir",    str(JOBS_DIR),
        ]
        if failed_chunks:
            cmd.extend(["--failed-chunks", ",".join(str(x) for x in failed_chunks)])

        print(f"[RENDER] job={job_id} sleep: {total_scene_count} scenes, single combined audio")
        result = await loop.run_in_executor(
            None, lambda c=cmd: subprocess.run(c, capture_output=True, text=True)
        )
        if result.returncode != 0:
            err = (result.stderr or result.stdout or "")[-1200:]
            _render_tasks[task_key] = {"status": "error", "error": f"compose.py failed: {err}"}
            return

        volume_out = Path("/workspace/output") / job_id
        volume_out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(output_mp4), str(volume_out / "sleep_en.mp4"))

        _render_tasks[task_key] = {
            "status": "done",
            "render_id": f"{job_id}_sleep_en",
            "download_url": f"/download/{job_id}/sleep_en",
            "volume_path": f"/workspace/output/{job_id}/sleep_en.mp4",
        }
    except Exception as e:
        _render_tasks[task_key] = {"status": "error", "error": str(e)}
    finally:
        _job_end(job_id)


@app.post("/render-start/{job_id}")
async def render_start(job_id: str, request: Request):
    """Start render in background. Poll /render-poll/{job_id}/{video_type}.

    v32 body variants:

    Short (or any non-chapter video):
      {
        "type": "short_en",
        "scene_count": 100,
        "audio_duration": 893.4,    # optional, informational
        "scene_duration": 8.934,    # float — audio_duration / scene_count
        "failed_chunks": []
      }

    Sleep (per-chapter):
      {
        "type": "sleep_en",
        "audio_files": ["sleep_ch1.wav", ..., "sleep_ch5.wav"],
        "chapters": [
          {"duration_sec": 720.5, "scene_count": 90, "scene_offset": 0},
          ...
        ],
        "failed_chunks": []
      }

    Backward compat: old body {"type", "scene_count", "scene_duration"} still works.
    """
    body       = await request.json()
    video_type = body.get("type", "short_en")
    task_key   = f"{job_id}/{video_type}"

    if _render_tasks.get(task_key, {}).get("status") == "processing":
        return {"status": "already_running", "task_key": task_key}

    _render_tasks[task_key] = {"status": "processing"}
    _job_start(job_id)

    failed_chunks: list[int] = [int(x) for x in body.get("failed_chunks", [])]

    # v32 per-chapter sleep path
    if "sleep" in video_type and "chapters" in body:
        audio_files: list[str] = body.get("audio_files", [])
        chapters:    list[dict] = body.get("chapters", [])
        if not audio_files or not chapters:
            _render_tasks[task_key] = {"status": "error", "error": "sleep render requires audio_files and chapters"}
            _job_end(job_id)
            return {"status": "error", "detail": "sleep render requires audio_files and chapters"}
        asyncio.create_task(
            _run_render_sleep_background(job_id, audio_files, chapters, failed_chunks, task_key)
        )
        return {"status": "started", "task_key": task_key, "mode": "per_chapter_sleep"}

    # Short (or legacy single-audio sleep)
    scene_count = int(body.get("scene_count", 9))
    # scene_duration is float since v32; fall back to legacy scene_dur key too
    scene_dur   = float(body.get("scene_duration", body.get("scene_dur", 8.0)))
    asyncio.create_task(
        _run_render_background(job_id, video_type, scene_count, scene_dur, task_key, failed_chunks)
    )
    return {"status": "started", "task_key": task_key}


@app.get("/render-poll/{job_id}/{video_type}")
async def render_poll(job_id: str, video_type: str):
    """Poll render status.
    Returns 200 when done, 503 when still processing (triggers n8n retryOnFail),
    500 on error."""
    task_key = f"{job_id}/{video_type}"
    t = _render_tasks.get(task_key)
    if t is None:
        raise HTTPException(status_code=404, detail=f"No render task: {task_key}")
    if t["status"] == "done":
        return {"status": "done", **{k: v for k, v in t.items() if k != "status"}}
    if t["status"] == "error":
        raise HTTPException(status_code=500, detail=t.get("error", "Render failed"))
    # Still processing — 503 so n8n HTTP Request retryOnFail keeps polling
    raise HTTPException(status_code=503, detail="Render still processing")


@app.post("/render/{job_id}")
async def render_video(job_id: str, request: Request):
    """Legacy sync endpoint — kept for compat. Use /render-start + /render-poll instead."""
    body       = await request.json()
    video_type = body.get("type", "short_en")
    task_key   = f"{job_id}/{video_type}"
    _render_tasks[task_key] = {"status": "processing"}
    _job_start(job_id)
    asyncio.create_task(_run_render_background(
        job_id, video_type,
        int(body.get("scene_count", 9)),
        int(body.get("scene_duration", 20)),
        task_key
    ))
    for _ in range(18):  # wait up to 90s
        await asyncio.sleep(5)
        t = _render_tasks.get(task_key, {})
        if t.get("status") == "done":
            return {k: v for k, v in t.items() if k != "status"}
        if t.get("status") == "error":
            raise HTTPException(status_code=500, detail=t.get("error", "Render failed"))
    raise HTTPException(status_code=503, detail="Render still processing — use /render-poll")


# ── Video download ────────────────────────────────────────────────

@app.get("/download/{job_id}/{video_type}")
async def download_video(job_id: str, video_type: str):
    path = JOBS_DIR / job_id / f"{video_type}.mp4"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(str(path), media_type="video/mp4", filename=f"{job_id}_{video_type}.mp4")


# ── Upload videos to Google Drive (async: start + poll, OAuth user-token) ───

def _load_oauth_config() -> dict:
    """Read OAuth config from volume file or env vars (env wins)."""
    cfg = {}
    if OAUTH_CONFIG_PATH.exists():
        try:
            cfg = json.loads(OAUTH_CONFIG_PATH.read_text())
        except Exception:
            cfg = {}
    # Env vars override file
    for k_env, k_cfg in [
        ("GOOGLE_CLIENT_ID",        "client_id"),
        ("GOOGLE_CLIENT_SECRET",    "client_secret"),
        ("GOOGLE_REFRESH_TOKEN",    "refresh_token"),
        ("DRIVE_PARENT_FOLDER_ID",  "parent_folder_id"),
    ]:
        v = os.environ.get(k_env)
        if v:
            cfg[k_cfg] = v
    return cfg


async def _get_drive_access_token() -> str:
    """Exchange refresh_token for a fresh access_token. Called on every upload."""
    cfg = _load_oauth_config()
    missing = [k for k in ("client_id", "client_secret", "refresh_token") if not cfg.get(k)]
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"OAuth not configured (missing: {missing}). POST /admin/save-oauth-config first.",
        )
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "grant_type":    "refresh_token",
                "refresh_token": cfg["refresh_token"],
                "client_id":     cfg["client_id"],
                "client_secret": cfg["client_secret"],
            },
        )
    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"OAuth token refresh failed: {resp.text[:300]}")
    return resp.json()["access_token"]


@app.post("/admin/save-oauth-config")
async def save_oauth_config(request: Request):
    """One-time setup: save OAuth client_id / secret / refresh_token / parent_folder_id
    to the volume so the server uses them for Drive uploads."""
    body = await request.json()
    required = ["client_id", "client_secret", "refresh_token", "parent_folder_id"]
    missing = [k for k in required if k not in body]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing: {missing}")
    OAUTH_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    OAUTH_CONFIG_PATH.write_text(json.dumps({k: body[k] for k in required}, indent=2))
    OAUTH_CONFIG_PATH.chmod(0o600)
    return {"status": "saved", "path": str(OAUTH_CONFIG_PATH)}


@app.get("/admin/oauth-config-status")
async def oauth_config_status():
    """Quick check that OAuth is wired up (does NOT leak the secrets)."""
    cfg = _load_oauth_config()
    return {
        "configured":      all(cfg.get(k) for k in ("client_id", "client_secret", "refresh_token")),
        "has_parent_folder": bool(cfg.get("parent_folder_id")),
        "config_path":     str(OAUTH_CONFIG_PATH),
        "file_exists":     OAUTH_CONFIG_PATH.exists(),
    }


async def _run_drive_background(job_id: str, case_name: str, parent_folder_id: str | None):
    try:
        access_token = await _get_drive_access_token()
        if not parent_folder_id:
            parent_folder_id = _load_oauth_config().get("parent_folder_id")
        if not parent_folder_id:
            raise HTTPException(status_code=500, detail="No parent_folder_id provided or configured")
        result = await _do_upload_to_drive(job_id, access_token, parent_folder_id, case_name)
        # Bug fix: if BOTH videos failed, mark task as error (not done)
        short_err = "error" in (result.get("short_en") or {})
        sleep_err = "error" in (result.get("sleep_en") or {})
        if short_err and sleep_err:
            _drive_tasks[job_id] = {
                "status": "error",
                "error":  f"Both uploads failed. short_en: {result['short_en'].get('error', '?')[:200]} | sleep_en: {result['sleep_en'].get('error', '?')[:200]}",
            }
        else:
            _drive_tasks[job_id] = {"status": "done", **result}
    except HTTPException as he:
        _drive_tasks[job_id] = {"status": "error", "error": str(he.detail)}
    except Exception as e:
        _drive_tasks[job_id] = {"status": "error", "error": str(e)}
    finally:
        _job_end(job_id)


@app.post("/drive-start/{job_id}")
async def drive_start(job_id: str, request: Request):
    """Start Drive upload in background. Poll /drive-poll/{job_id}.
    Body: {"case_name": "...", "folder_id": "..."(optional, falls back to env/config)}
    """
    body = await request.json()
    if _drive_tasks.get(job_id, {}).get("status") == "processing":
        return {"status": "already_running"}
    _drive_tasks[job_id] = {"status": "processing"}
    _job_start(job_id)
    asyncio.create_task(_run_drive_background(
        job_id,
        body.get("case_name", job_id),
        body.get("folder_id"),
    ))
    return {"status": "started"}


@app.get("/drive-poll/{job_id}")
async def drive_poll(job_id: str):
    """Poll Drive upload.
    Returns 200 when done, 503 when still processing (triggers n8n retryOnFail),
    500 on error."""
    t = _drive_tasks.get(job_id)
    if t is None:
        raise HTTPException(status_code=404, detail=f"No drive task: {job_id}")
    if t["status"] == "done":
        return {"status": "done", **{k: v for k, v in t.items() if k != "status"}}
    if t["status"] == "error":
        raise HTTPException(status_code=500, detail=t.get("error", "Upload failed"))
    # Still processing — 503 so n8n HTTP Request retryOnFail keeps polling
    raise HTTPException(status_code=503, detail="Drive upload still processing")


async def _do_upload_to_drive(job_id: str, access_token: str, parent_folder_id: str, case_name: str):
    # Create subfolder for this job
    async with httpx.AsyncClient(timeout=60) as client:
        folder_resp = await client.post(
            "https://www.googleapis.com/drive/v3/files",
            headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
            json={"name": case_name, "mimeType": "application/vnd.google-apps.folder", "parents": [parent_folder_id]},
        )
        if folder_resp.status_code not in (200, 201):
            raise HTTPException(status_code=500, detail=f"Folder creation failed: {folder_resp.text}")
        subfolder_id = folder_resp.json()["id"]

    results = {
        "folder_id": subfolder_id,
        "folder_url": f"https://drive.google.com/drive/folders/{subfolder_id}",
    }

    CHUNK = 10 * 1024 * 1024  # 10 MB chunks

    for video_type in ["short_en", "sleep_en"]:
        file_path = JOBS_DIR / job_id / f"{video_type}.mp4"
        if not file_path.exists():
            results[video_type] = {"error": "not found"}
            continue

        file_size = file_path.stat().st_size

        async with httpx.AsyncClient(timeout=3600) as client:
            # Initiate resumable upload
            init_resp = await client.post(
                "https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json",
                    "X-Upload-Content-Type": "video/mp4",
                    "X-Upload-Content-Length": str(file_size),
                },
                json={"name": f"{video_type}.mp4", "parents": [subfolder_id]},
            )
            if init_resp.status_code != 200:
                results[video_type] = {"error": f"init failed {init_resp.status_code}: {init_resp.text[:200]}"}
                continue

            upload_url = init_resp.headers["Location"]

            file_id = None
            with open(file_path, "rb") as f:
                offset = 0
                while offset < file_size:
                    chunk = f.read(CHUNK)
                    if not chunk:
                        break
                    end = offset + len(chunk) - 1
                    upload_resp = await client.put(
                        upload_url,
                        content=chunk,
                        headers={
                            "Content-Length": str(len(chunk)),
                            "Content-Range": f"bytes {offset}-{end}/{file_size}",
                        },
                        timeout=300,
                    )
                    if upload_resp.status_code in (200, 201):
                        file_id = upload_resp.json()["id"]
                        break
                    elif upload_resp.status_code == 308:
                        offset = end + 1
                    else:
                        results[video_type] = {"error": f"chunk failed {upload_resp.status_code}"}
                        break

            if file_id:
                results[video_type] = {
                    "file_id": file_id,
                    "view_url": f"https://drive.google.com/file/d/{file_id}/view",
                }

    return results


# ── Helpers ───────────────────────────────────────────────────────

def _wav_duration(wav: bytes) -> float:
    if len(wav) > 44 and wav[:4] == b"RIFF":
        sample_rate = struct.unpack_from("<I", wav, 24)[0]
        channels    = struct.unpack_from("<H", wav, 22)[0]
        bits        = struct.unpack_from("<H", wav, 34)[0]
        data_size   = struct.unpack_from("<I", wav, 40)[0]
        if sample_rate and channels and bits:
            return round(data_size / (sample_rate * channels * (bits // 8)), 3)
    return 0.0
