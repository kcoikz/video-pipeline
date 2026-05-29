import asyncio
import copy
import os
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

async def _run_tts_background(job_id: str, folder: str, body: dict, task_key: str):
    """Background coroutine: calls Chatterbox, saves WAV, updates _tts_tasks."""
    try:
        body.setdefault("voice_mode", "predefined")
        body.setdefault("predefined_voice_id", "Emily.wav")
        async with httpx.AsyncClient(timeout=1800) as client:
            resp = await client.post(f"{CHATTERBOX_URL}/tts", json=body)
            if resp.status_code != 200:
                _tts_tasks[task_key] = {"status": "error", "error": resp.text}
                return
        wav = resp.content
        dest = JOBS_DIR / job_id / folder
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "en_audio.wav").write_bytes(wav)
        duration_sec = _wav_duration(wav)
        scene_count = max(1, int(duration_sec / 20))
        _tts_tasks[task_key] = {
            "status": "done",
            "success": True,
            "duration_sec": duration_sec,
            "scene_count": scene_count,
        }
    except Exception as e:
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
    """Start TTS in background. Returns immediately. Poll /tts-poll/{job_id}/{folder}."""
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
    """Poll TTS background task. Returns status field so n8n IF-loop can check."""
    task_key = f"{job_id}/{folder}"
    t = _tts_tasks.get(task_key)
    if t is None:
        raise HTTPException(status_code=404, detail=f"No TTS task: {task_key}")
    if t["status"] == "done":
        return {"status": "done", **{k: v for k, v in t.items() if k != "status"}}
    if t["status"] == "error":
        return {"status": "error", "error": t.get("error", "TTS failed")}
    # Still processing — return 200 with status so n8n IF-loop handles it
    return {"status": "processing"}


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


# ── File upload ───────────────────────────────────────────────────

@app.post("/upload/{job_id}/{folder}/{filename}")
async def upload_file(job_id: str, folder: str, filename: str, request: Request):
    dest = JOBS_DIR / job_id / folder
    dest.mkdir(parents=True, exist_ok=True)
    (dest / filename).write_bytes(await request.body())
    return {"success": True}


# ── Render — FFmpeg slideshow (async: start + poll) ───────────────

async def _run_render_background(job_id: str, video_type: str, scene_count: int, scene_dur: int, task_key: str):
    """Run FFmpeg in a thread so we don't block the event loop."""
    import shutil
    folder    = "short" if "short" in video_type else "sleep"
    job_path  = JOBS_DIR / job_id / folder
    audio_path = job_path / "en_audio.wav"
    output_mp4 = JOBS_DIR / job_id / f"{video_type}.mp4"
    concat_txt = JOBS_DIR / job_id / f"{video_type}_concat.txt"

    try:
        if not audio_path.exists():
            _render_tasks[task_key] = {"status": "error", "error": f"Audio not found: {audio_path}"}
            return

        lines = []
        last_img = None
        for i in range(scene_count):
            img = job_path / f"scene_{str(i).zfill(3)}.jpg"
            if not img.exists():
                img = job_path / f"scene_{str(i).zfill(3)}.png"
            if not img.exists():
                _render_tasks[task_key] = {"status": "error", "error": f"Missing scene image {i}"}
                return
            lines.append(f"file '{img}'\nduration {scene_dur}")
            last_img = img
        lines.append(f"file '{last_img}'")
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
        # Run blocking FFmpeg in thread pool so event loop stays free
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: subprocess.run(cmd, capture_output=True, text=True)
        )
        if result.returncode != 0:
            _render_tasks[task_key] = {"status": "error", "error": f"FFmpeg failed: {result.stderr[-600:]}"}
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


@app.post("/render-start/{job_id}")
async def render_start(job_id: str, request: Request):
    """Start render in background. Poll /render-poll/{job_id}/{video_type}."""
    body        = await request.json()
    video_type  = body.get("type", "short_en")
    scene_count = int(body.get("scene_count", 9))
    scene_dur   = int(body.get("scene_duration", 20))
    task_key    = f"{job_id}/{video_type}"

    if _render_tasks.get(task_key, {}).get("status") == "processing":
        return {"status": "already_running", "task_key": task_key}

    _render_tasks[task_key] = {"status": "processing"}
    _job_start(job_id)
    asyncio.create_task(_run_render_background(job_id, video_type, scene_count, scene_dur, task_key))
    return {"status": "started", "task_key": task_key}


@app.get("/render-poll/{job_id}/{video_type}")
async def render_poll(job_id: str, video_type: str):
    """Poll render status. Returns status field so n8n IF-loop can check."""
    task_key = f"{job_id}/{video_type}"
    t = _render_tasks.get(task_key)
    if t is None:
        raise HTTPException(status_code=404, detail=f"No render task: {task_key}")
    if t["status"] == "done":
        return {"status": "done", **{k: v for k, v in t.items() if k != "status"}}
    if t["status"] == "error":
        return {"status": "error", "error": t.get("error", "Render failed")}
    return {"status": "processing"}


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


# ── Upload videos to Google Drive (async: start + poll) ───────────

async def _run_drive_background(job_id: str, access_token: str, parent_folder_id: str, case_name: str):
    try:
        result = await _do_upload_to_drive(job_id, access_token, parent_folder_id, case_name)
        _drive_tasks[job_id] = {"status": "done", **result}
    except Exception as e:
        _drive_tasks[job_id] = {"status": "error", "error": str(e)}
    finally:
        _job_end(job_id)


@app.post("/drive-start/{job_id}")
async def drive_start(job_id: str, request: Request):
    """Start Drive upload in background. Poll /drive-poll/{job_id}."""
    body = await request.json()
    if _drive_tasks.get(job_id, {}).get("status") == "processing":
        return {"status": "already_running"}
    _drive_tasks[job_id] = {"status": "processing"}
    _job_start(job_id)
    asyncio.create_task(_run_drive_background(
        job_id, body["access_token"], body["folder_id"], body.get("case_name", job_id)
    ))
    return {"status": "started"}


@app.get("/drive-poll/{job_id}")
async def drive_poll(job_id: str):
    """Poll Drive upload. Returns status field so n8n IF-loop can check."""
    t = _drive_tasks.get(job_id)
    if t is None:
        raise HTTPException(status_code=404, detail=f"No drive task: {job_id}")
    if t["status"] == "done":
        return {"status": "done", **{k: v for k, v in t.items() if k != "status"}}
    if t["status"] == "error":
        return {"status": "error", "error": t.get("error", "Upload failed")}
    return {"status": "processing"}


@app.post("/upload-to-drive/{job_id}")
async def upload_to_drive(job_id: str, request: Request):
    """Legacy sync endpoint — use /drive-start + /drive-poll instead."""
    body = await request.json()
    _drive_tasks[job_id] = {"status": "processing"}
    _job_start(job_id)
    asyncio.create_task(_run_drive_background(
        job_id, body["access_token"], body["folder_id"], body.get("case_name", job_id)
    ))
    for _ in range(18):  # wait up to 90s
        await asyncio.sleep(5)
        t = _drive_tasks.get(job_id, {})
        if t.get("status") == "done":
            return {k: v for k, v in t.items() if k != "status"}
        if t.get("status") == "error":
            raise HTTPException(status_code=500, detail=t.get("error", "Upload failed"))
    raise HTTPException(status_code=503, detail="Upload still processing — use /drive-poll")


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
