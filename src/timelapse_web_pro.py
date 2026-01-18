#!/usr/bin/env python3
"""
Time-Lapse Web Pro (single-file FastAPI app)

Features:
- Live preview (JPEG snapshot endpoint)
- Start/Stop capture to unique project folders
- Choose resolution, interval, file format (png/jpg), jpg quality
- Autosave state every N frames (default 5)
- Gallery + MP4 builder (ffmpeg)
- MP4 player with speed + loop
- File manager (delete images/mp4/projects safely)
- Themes (including "Star Trek" style)
"""

from __future__ import annotations

import io
import json
import re
import secrets
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles


# -----------------------------
# Paths / persistence
# -----------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
PROJECTS_DIR = ROOT_DIR / "projects"
WWW_DIR = ROOT_DIR / "www"
STATE_FILE = ROOT_DIR / "app_state.json"

PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
WWW_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Time-Lapse Web Pro", version="3.0")
app.mount("/www", StaticFiles(directory=str(WWW_DIR), html=False), name="www")


# -----------------------------
# Defaults
# -----------------------------

RESOLUTION_OPTIONS: Dict[int, Tuple[int, int]] = {
    1: (640, 480),
    2: (800, 600),
    3: (1024, 768),
    4: (1280, 720),
    5: (1280, 1024),
    6: (1920, 1080),
    7: (1920, 1440),
    8: (2560, 1440),
    9: (3840, 2160),
    10: (4096, 2160),
}

THEMES: List[Dict[str, str]] = [
    {"id": "neon", "name": "Neon Arcade"},
    {"id": "trek", "name": "Star Trek LCARS"},
    {"id": "matrix", "name": "Matrix Terminal"},
    {"id": "cyber", "name": "Cyberpunk Holo"},
    {"id": "ocean", "name": "Deep Ocean"},
    {"id": "sunset", "name": "Synth Sunset"},
    {"id": "ice", "name": "Ice Core"},
    {"id": "ember", "name": "Ember Forge"},
    {"id": "forest", "name": "Forest Lab"},
    {"id": "mono", "name": "Mono Minimal"},
]

ALLOWED_IMAGE_EXTS = ["png", "jpg", "jpeg"]

DEFAULTS = {
    "camera_index": 0,
    "resolution_key": 4,
    "interval_seconds": 60.0,
    "project_name": "project",
    "image_ext": "png",
    "jpg_quality": 92,
    "autosave_every": 5,
    "retention_limit": 0,
    "theme": "neon",
    "overlay_timestamp": True,
    "overlay_project": True,
    "text_scale": 1.0,
    "text_thickness": 2,
}


# -----------------------------
# State model
# -----------------------------

@dataclass
class RunConfig:
    camera_index: int = DEFAULTS["camera_index"]
    resolution_key: int = DEFAULTS["resolution_key"]
    interval_seconds: float = DEFAULTS["interval_seconds"]
    project_name: str = DEFAULTS["project_name"]
    image_ext: str = DEFAULTS["image_ext"]
    jpg_quality: int = DEFAULTS["jpg_quality"]
    autosave_every: int = DEFAULTS["autosave_every"]
    retention_limit: int = DEFAULTS["retention_limit"]
    theme: str = DEFAULTS["theme"]
    overlay_timestamp: bool = DEFAULTS["overlay_timestamp"]
    overlay_project: bool = DEFAULTS["overlay_project"]
    text_scale: float = DEFAULTS["text_scale"]
    text_thickness: int = DEFAULTS["text_thickness"]

@dataclass
class AppState:
    running: bool = False
    current_project_id: str = ""
    current_project_name: str = DEFAULTS["project_name"]
    started_at: str = ""
    last_saved: str = ""
    total_saved: int = 0
    last_error: str = ""
    config: RunConfig = RunConfig()

_state_lock = threading.RLock()
_state: AppState = AppState()


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _safe_name(name: str) -> str:
    name = (name or "project").strip()
    name = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name[:40] or "project"

def _new_project_id(project_name: str) -> str:
    base = _safe_name(project_name)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tok = secrets.token_hex(3)
    return f"{base}_{ts}_{tok}"

def _project_paths(project_id: str) -> Dict[str, Path]:
    pid = (project_id or "").strip()
    if not pid or "/" in pid or "\\" in pid or ".." in pid:
        raise HTTPException(400, "Invalid project id")
    pdir = PROJECTS_DIR / pid
    imgdir = pdir / "images"
    mp4dir = pdir / "mp4"
    webdir = WWW_DIR / "projects" / pid
    webimg = webdir / "images"
    webmp4 = webdir / "mp4"
    return {"pdir": pdir, "imgdir": imgdir, "mp4dir": mp4dir, "webdir": webdir, "webimg": webimg, "webmp4": webmp4}

def _ensure_project_dirs(project_id: str) -> Dict[str, Path]:
    paths = _project_paths(project_id)
    for k in ["pdir", "imgdir", "mp4dir", "webdir"]:
        paths[k].mkdir(parents=True, exist_ok=True)

    paths["webimg"].parent.mkdir(parents=True, exist_ok=True)
    paths["webmp4"].parent.mkdir(parents=True, exist_ok=True)
    try:
        if not paths["webimg"].exists():
            paths["webimg"].symlink_to(paths["imgdir"], target_is_directory=True)
        if not paths["webmp4"].exists():
            paths["webmp4"].symlink_to(paths["mp4dir"], target_is_directory=True)
    except Exception:
        paths["webimg"].mkdir(parents=True, exist_ok=True)
        paths["webmp4"].mkdir(parents=True, exist_ok=True)
    return paths


def _load_state() -> None:
    global _state
    if not STATE_FILE.exists():
        return
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        cfg = data.get("config", {})
        config = RunConfig(**{**DEFAULTS, **cfg})
        _state = AppState(
            running=bool(data.get("running", False)),
            current_project_id=str(data.get("current_project_id", "")),
            current_project_name=str(data.get("current_project_name", DEFAULTS["project_name"])),
            started_at=str(data.get("started_at", "")),
            last_saved=str(data.get("last_saved", "")),
            total_saved=int(data.get("total_saved", 0)),
            last_error=str(data.get("last_error", "")),
            config=config,
        )
        _state.running = False
    except Exception as e:
        _state.last_error = f"State load failed: {e}"

def _save_state() -> None:
    with _state_lock:
        payload = asdict(_state)
        payload["config"] = asdict(_state.config)
        tmp = STATE_FILE.with_name(STATE_FILE.stem + ".__tmp__" + STATE_FILE.suffix)
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(STATE_FILE)

_load_state()
_save_state()


# -----------------------------
# Camera worker (single open handle)
# -----------------------------

class CameraWorker:
    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._frame_lock = threading.Lock()
        self._last_jpeg: Optional[bytes] = None
        self._jpeg_ts = 0.0
        self._capture: Optional[cv2.VideoCapture] = None
        self._last_open_params: Tuple[int, int, int] = (-1, -1, -1)
        self._next_save_t = 0.0
        self._saved_since_autosave = 0

    def start_thread(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="CameraWorker", daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        self._stop.set()
        try:
            if self._thread:
                self._thread.join(timeout=2.0)
        finally:
            self._close_camera()

    def get_jpeg(self) -> Optional[bytes]:
        with self._frame_lock:
            return self._last_jpeg

    def _open_camera(self, cam_index: int, width: int, height: int) -> None:
        if self._capture is not None and self._last_open_params == (cam_index, width, height):
            return
        self._close_camera()

        cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {cam_index}. Check /dev/video* and permissions.")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))

        self._capture = cap
        self._last_open_params = (cam_index, width, height)

    def _close_camera(self) -> None:
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:
                pass
        self._capture = None
        self._last_open_params = (-1, -1, -1)

    def _overlay(self, frame: np.ndarray, project_label: str) -> np.ndarray:
        with _state_lock:
            cfg = _state.config
            overlay_ts = cfg.overlay_timestamp
            overlay_proj = cfg.overlay_project
            scale = float(cfg.text_scale)
            thick = int(cfg.text_thickness)

        if not overlay_ts and not overlay_proj:
            return frame

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 255, 255)
        shadow = (0, 0, 0)

        def put(line: str, y: int) -> None:
            cv2.putText(frame, line, (10, y), font, scale, shadow, thick + 2, cv2.LINE_AA)
            cv2.putText(frame, line, (10, y), font, scale, color, thick, cv2.LINE_AA)

        y = int(30 * max(0.6, scale))
        if overlay_ts:
            put(ts, y)
            y += int(32 * max(0.6, scale))
        if overlay_proj:
            put(project_label, y)
        return frame

    def _encode_preview_jpeg(self, frame: np.ndarray) -> bytes:
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        return buf.tobytes()

    def _save_frame(self, frame: np.ndarray, project_id: str, project_name: str) -> None:
        with _state_lock:
            cfg = _state.config
            ext = (cfg.image_ext or "png").lower()
            jpgq = int(cfg.jpg_quality)
            autosave_n = max(1, int(cfg.autosave_every))
            keep = int(cfg.retention_limit)

        if ext == "jpeg":
            ext = "jpg"
        if ext not in ALLOWED_IMAGE_EXTS:
            ext = "png"

        paths = _ensure_project_dirs(project_id)
        imgdir = paths["imgdir"]

        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fn = f"{project_name}_{ts}.{ext}"
        out_path = imgdir / fn
        tmp = out_path.with_name(out_path.stem + ".__tmp__" + out_path.suffix)

        params = []
        if ext in ("jpg", "jpeg"):
            params = [int(cv2.IMWRITE_JPEG_QUALITY), max(1, min(100, jpgq))]

        ok = cv2.imwrite(str(tmp), frame, params)
        if not ok:
            raise RuntimeError(f"cv2.imwrite failed for {tmp}")
        tmp.replace(out_path)

        with _state_lock:
            _state.total_saved += 1
            _state.last_saved = _now()
            _state.last_error = ""

        self._saved_since_autosave += 1
        if self._saved_since_autosave >= autosave_n:
            self._saved_since_autosave = 0
            _save_state()

        if keep and keep > 0:
            try:
                files = sorted([p for p in imgdir.iterdir() if p.is_file()], key=lambda p: p.stat().st_mtime)
                if len(files) > keep:
                    for p in files[: max(0, len(files) - keep)]:
                        try:
                            p.unlink()
                        except Exception:
                            pass
            except Exception:
                pass

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                with _state_lock:
                    cfg = _state.config
                    cam_index = int(cfg.camera_index)
                    rk = int(cfg.resolution_key)
                    interval = float(cfg.interval_seconds)
                    pname = _safe_name(cfg.project_name)
                    pid = _state.current_project_id or ""
                    running = bool(_state.running)

                width, height = RESOLUTION_OPTIONS.get(rk, RESOLUTION_OPTIONS[DEFAULTS["resolution_key"]])
                self._open_camera(cam_index, width, height)
                assert self._capture is not None

                ret, frame = self._capture.read()
                if not ret or frame is None:
                    raise RuntimeError("Camera read failed")

                label = f"{pname} [{pid}]" if pid else pname
                frame2 = self._overlay(frame.copy(), label)

                t = time.time()
                if t - self._jpeg_ts >= 0.45:
                    jpg = self._encode_preview_jpeg(frame2)
                    with self._frame_lock:
                        self._last_jpeg = jpg
                        self._jpeg_ts = t

                if running and pid:
                    if self._next_save_t == 0.0:
                        self._next_save_t = t
                    if t >= self._next_save_t:
                        self._save_frame(frame2, pid, pname)
                        self._next_save_t = t + max(0.2, interval)
                else:
                    self._next_save_t = 0.0
                    self._saved_since_autosave = 0

                time.sleep(0.02)

            except Exception as e:
                with _state_lock:
                    _state.last_error = str(e)
                self._close_camera()
                time.sleep(0.6)


camera = CameraWorker()
camera.start_thread()

def _stop_background_on_exit(*_a: Any) -> None:
    camera.shutdown()

signal.signal(signal.SIGTERM, _stop_background_on_exit)
signal.signal(signal.SIGINT, _stop_background_on_exit)


# -----------------------------
# Project listing helpers
# -----------------------------

def list_projects() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for pdir in sorted([p for p in PROJECTS_DIR.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True):
        pid = pdir.name
        imgdir = pdir / "images"
        mp4dir = pdir / "mp4"
        imgs = [x for x in imgdir.iterdir() if x.is_file()] if imgdir.exists() else []
        mp4s = sorted([x.name for x in mp4dir.glob("*.mp4")], reverse=True) if mp4dir.exists() else []
        out.append({"id": pid, "images": len(imgs), "mp4": mp4s, "mtime": pdir.stat().st_mtime})
    return out

def list_images(project_id: str, limit: int = 300) -> List[str]:
    paths = _project_paths(project_id)
    imgdir = paths["imgdir"]
    if not imgdir.exists():
        return []
    files = sorted([p.name for p in imgdir.iterdir() if p.is_file()], reverse=True)
    return files[: max(1, min(2000, int(limit)))]

def list_mp4(project_id: str) -> List[str]:
    paths = _project_paths(project_id)
    mp4dir = paths["mp4dir"]
    if not mp4dir.exists():
        return []
    return sorted([p.name for p in mp4dir.glob("*.mp4")], reverse=True)

def _ffmpeg_exists() -> bool:
    return shutil.which("ffmpeg") is not None

def build_mp4(project_id: str, fps: int, pattern: str) -> Dict[str, Any]:
    paths = _ensure_project_dirs(project_id)
    imgdir = paths["imgdir"]
    mp4dir = paths["mp4dir"]

    if not _ffmpeg_exists():
        raise RuntimeError("ffmpeg not found. Install with: sudo apt-get install -y ffmpeg")

    fps = max(1, min(120, int(fps)))
    imgs = sorted([p for p in imgdir.glob(pattern) if p.is_file()])
    if not imgs:
        raise RuntimeError(f"No images matching {pattern} in {imgdir}")

    list_file = mp4dir / f"frames_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    lines = []
    for p in imgs:
        path = str(p.resolve()).replace("'", r"'\''")
        lines.append(f"file '{path}'")
    list_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    out_name = f"{project_id}_{fps}fps_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    out_path = mp4dir / out_name

    cmd = ["ffmpeg", "-y", "-r", str(fps), "-f", "concat", "-safe", "0", "-i", str(list_file),
           "-vf", "format=yuv420p", "-movflags", "+faststart", str(out_path)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed (rc={proc.returncode}).\n{proc.stderr[-2000:]}")
    return {"mp4": out_name, "fps": fps}


# -----------------------------
# API
# -----------------------------

@app.get("/api/state")
def api_state() -> Dict[str, Any]:
    with _state_lock:
        d = asdict(_state)
        d["config"] = asdict(_state.config)
        d["themes"] = THEMES
        d["resolutions"] = [{"k": k, "w": v[0], "h": v[1]} for k, v in RESOLUTION_OPTIONS.items()]
        return d

@app.post("/api/config")
def api_config(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    with _state_lock:
        cfg = _state.config

        def set_if(key: str, cast, allow=None, clamp=None):
            if key not in payload:
                return
            try:
                val = cast(payload[key])
            except Exception:
                return
            if allow and val not in allow:
                return
            if clamp:
                lo, hi = clamp
                try:
                    val = max(lo, min(hi, val))
                except Exception:
                    pass
            setattr(cfg, key, val)

        set_if("camera_index", int, clamp=(0, 16))
        set_if("resolution_key", int, allow=set(RESOLUTION_OPTIONS.keys()))
        set_if("interval_seconds", float, clamp=(0.2, 365*24*3600))
        if "project_name" in payload:
            cfg.project_name = _safe_name(str(payload["project_name"]))
        if "image_ext" in payload:
            ext = str(payload["image_ext"]).lower().strip(".")
            if ext == "jpeg":
                ext = "jpg"
            if ext in ALLOWED_IMAGE_EXTS:
                cfg.image_ext = ext
        set_if("jpg_quality", int, clamp=(1, 100))
        set_if("autosave_every", int, clamp=(1, 1000))
        set_if("retention_limit", int, clamp=(0, 5_000_000))
        if "theme" in payload:
            th = str(payload["theme"])
            if any(t["id"] == th for t in THEMES):
                cfg.theme = th
        if "overlay_timestamp" in payload:
            cfg.overlay_timestamp = bool(payload["overlay_timestamp"])
        if "overlay_project" in payload:
            cfg.overlay_project = bool(payload["overlay_project"])
        if "text_scale" in payload:
            try:
                cfg.text_scale = max(0.4, min(3.0, float(payload["text_scale"])))
            except Exception:
                pass
        if "text_thickness" in payload:
            try:
                cfg.text_thickness = max(1, min(6, int(payload["text_thickness"])))
            except Exception:
                pass

        _state.config = cfg
        _save_state()
    return {"ok": True}

@app.post("/api/new_project")
def api_new_project(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    name = _safe_name(str(payload.get("project_name", DEFAULTS["project_name"])))
    pid = _new_project_id(name)
    _ensure_project_dirs(pid)
    with _state_lock:
        _state.current_project_id = pid
        _state.current_project_name = name
        _state.config.project_name = name
        _save_state()
    return {"ok": True, "project_id": pid}

@app.post("/api/start")
def api_start() -> Dict[str, Any]:
    with _state_lock:
        if not _state.current_project_id:
            pid = _new_project_id(_state.config.project_name)
            _ensure_project_dirs(pid)
            _state.current_project_id = pid
            _state.current_project_name = _state.config.project_name
        _state.running = True
        _state.started_at = _now()
        _state.last_error = ""
        _save_state()
        pid = _state.current_project_id
    return {"ok": True, "project_id": pid}

@app.post("/api/stop")
def api_stop() -> Dict[str, Any]:
    with _state_lock:
        _state.running = False
        _save_state()
    return {"ok": True}

@app.get("/preview.jpg")
def preview_jpg() -> Response:
    jpg = camera.get_jpeg()
    if jpg is None:
        return Response(content=b"", media_type="image/jpeg", status_code=404)
    return Response(content=jpg, media_type="image/jpeg")

@app.get("/api/projects")
def api_projects() -> Dict[str, Any]:
    return {"ok": True, "projects": list_projects()}

@app.get("/api/project/{project_id}/images")
def api_project_images(project_id: str, limit: int = 300) -> Dict[str, Any]:
    imgs = list_images(project_id, limit=limit)
    return {"ok": True, "images": imgs, "count": len(imgs)}

@app.get("/api/project/{project_id}/mp4")
def api_project_mp4(project_id: str) -> Dict[str, Any]:
    mp4s = list_mp4(project_id)
    return {"ok": True, "mp4": mp4s, "count": len(mp4s)}

@app.post("/api/project/{project_id}/build_mp4")
def api_build_mp4(project_id: str, payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    fps = int(payload.get("fps", 30))
    ext = str(payload.get("ext", "png")).lower().strip(".")
    if ext == "jpeg":
        ext = "jpg"
    if ext not in ALLOWED_IMAGE_EXTS:
        ext = "png"
    try:
        r = build_mp4(project_id, fps=fps, pattern=f"*.{ext}")
        return {"ok": True, **r}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def _safe_rel(name: str) -> str:
    name = (name or "").strip()
    if not name or "/" in name or "\\" in name or ".." in name:
        raise HTTPException(400, "Invalid filename")
    return name

@app.post("/api/project/{project_id}/delete_image")
def api_delete_image(project_id: str, payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    fn = _safe_rel(str(payload.get("filename", "")))
    paths = _project_paths(project_id)
    p = paths["imgdir"] / fn
    if not p.exists() or not p.is_file():
        raise HTTPException(404, "Not found")
    p.unlink()
    return {"ok": True}

@app.post("/api/project/{project_id}/delete_mp4")
def api_delete_mp4(project_id: str, payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    fn = _safe_rel(str(payload.get("filename", "")))
    paths = _project_paths(project_id)
    p = paths["mp4dir"] / fn
    if not p.exists() or not p.is_file():
        raise HTTPException(404, "Not found")
    p.unlink()
    return {"ok": True}

@app.post("/api/project/{project_id}/delete_project")
def api_delete_project(project_id: str) -> Dict[str, Any]:
    paths = _project_paths(project_id)
    if not paths["pdir"].exists():
        raise HTTPException(404, "Not found")
    with _state_lock:
        if _state.current_project_id == project_id:
            _state.running = False
            _state.current_project_id = ""
            _save_state()
    shutil.rmtree(paths["pdir"], ignore_errors=True)
    try:
        if paths["webdir"].exists():
            if paths["webdir"].is_symlink():
                paths["webdir"].unlink()
            else:
                shutil.rmtree(paths["webdir"], ignore_errors=True)
    except Exception:
        pass
    return {"ok": True}


# -----------------------------
# Web UI
# -----------------------------

def _theme_css() -> str:
    return r"""
:root{
  --bg0:#070a12; --bg1:#0c1224; --card:#0f1a33; --ink:#e7f0ff; --muted:#9fb3d1;
  --accent:#36d6ff; --accent2:#ff4fd8; --good:#2bff88; --warn:#ffd166; --bad:#ff4f4f;
  --border: rgba(255,255,255,.12);
  --shadow: 0 14px 40px rgba(0,0,0,.45);
  --radius: 22px;
  --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  --sans: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji","Segoe UI Emoji";
}
body{ margin:0; font-family:var(--sans); color:var(--ink); background:
 radial-gradient(1200px 700px at 15% 10%, rgba(54,214,255,.18), transparent 55%),
 radial-gradient(1100px 600px at 85% 25%, rgba(255,79,216,.12), transparent 58%),
 linear-gradient(180deg, var(--bg0), var(--bg1));
 min-height:100vh;
}
.container{ max-width:1300px; margin:18px auto; padding:14px; }
a{ color:inherit; text-decoration:none; }
.topbar{
  display:flex; align-items:center; justify-content:space-between; gap:14px;
  padding:14px 16px; border:1px solid var(--border); border-radius:var(--radius);
  background: linear-gradient(180deg, rgba(255,255,255,.06), rgba(255,255,255,.02));
  box-shadow: var(--shadow);
  position:sticky; top:12px; z-index:50; backdrop-filter: blur(10px);
}
.brand{ display:flex; align-items:center; gap:12px; }
.badge{
  display:inline-flex; align-items:center; gap:10px;
  padding:8px 12px; border-radius:999px; border:1px solid var(--border);
  background: rgba(0,0,0,.22);
  font-family:var(--mono); font-size:12px; color:var(--muted);
}
.dot{ width:10px; height:10px; border-radius:999px; background: var(--bad); box-shadow:0 0 14px rgba(255,79,79,.45); }
.dot.on{ background: var(--good); box-shadow:0 0 16px rgba(43,255,136,.55); }
.btn{
  appearance:none; border:none; cursor:pointer;
  padding:10px 14px; border-radius:14px;
  color:var(--ink); font-weight:700;
  background: linear-gradient(180deg, rgba(54,214,255,.22), rgba(54,214,255,.10));
  border:1px solid rgba(54,214,255,.35);
  box-shadow: 0 10px 26px rgba(0,0,0,.35);
  transition: transform .08s ease, filter .15s ease, opacity .15s ease;
}
.btn:hover{ filter:brightness(1.06); }
.btn:active{ transform: translateY(1px); }
.btn.secondary{
  background: linear-gradient(180deg, rgba(255,79,216,.18), rgba(255,79,216,.08));
  border-color: rgba(255,79,216,.35);
}
.btn.ghost{
  background: rgba(255,255,255,.04); border-color: var(--border);
}
.btn.danger{
  background: linear-gradient(180deg, rgba(255,79,79,.18), rgba(255,79,79,.08));
  border-color: rgba(255,79,79,.35);
}
.btn:disabled{ opacity:.55; cursor:not-allowed; }
.grid{ display:grid; grid-template-columns: 1.6fr 1fr; gap:14px; margin-top:14px; }
.card{
  border:1px solid var(--border);
  border-radius:var(--radius);
  background: linear-gradient(180deg, rgba(255,255,255,.06), rgba(255,255,255,.02));
  box-shadow: var(--shadow);
  overflow:hidden;
}
.card .hd{ display:flex; align-items:center; justify-content:space-between; padding:12px 14px; border-bottom:1px solid rgba(255,255,255,.08); }
.card .hd h2{ margin:0; font-size:14px; letter-spacing:.4px; color:var(--muted); text-transform:uppercase; }
.card .bd{ padding:14px; }
.preview-wrap{
  aspect-ratio: 16/9;
  width:100%;
  border-radius: 18px;
  overflow:hidden;
  border:1px solid rgba(255,255,255,.14);
  background: rgba(0,0,0,.30);
  position:relative;
}
.preview-wrap img{ width:100%; height:100%; object-fit:cover; display:block; }
.preview-overlay{ position:absolute; left:10px; bottom:10px; display:flex; gap:8px; flex-wrap:wrap; }
.pill{
  font-family:var(--mono); font-size:12px;
  padding:6px 10px; border-radius:999px;
  border:1px solid rgba(255,255,255,.14);
  background: rgba(0,0,0,.35);
  color: var(--muted);
}
.form{ display:grid; grid-template-columns: 1fr 1fr; gap:10px; }
.field{ display:flex; flex-direction:column; gap:6px; }
label{ font-size:12px; color:var(--muted); }
input, select{
  background: rgba(0,0,0,.22);
  border:1px solid rgba(255,255,255,.14);
  color:var(--ink);
  border-radius: 14px;
  padding:10px 12px;
  outline:none;
}
input:focus, select:focus{ border-color: rgba(54,214,255,.55); box-shadow:0 0 0 3px rgba(54,214,255,.12); }
.row{ display:flex; gap:10px; flex-wrap:wrap; }
.msg{
  border:1px solid rgba(255,255,255,.14);
  border-radius:14px;
  padding:10px 12px;
  background: rgba(0,0,0,.25);
  color: var(--muted);
  font-family: var(--mono);
  font-size: 12px;
  white-space: pre-wrap;
}
.small{ font-size:12px; color:var(--muted); font-family: var(--mono); }
.tabs{ display:flex; gap:8px; flex-wrap:wrap; }
.tab{
  padding:8px 10px; border-radius: 999px;
  border:1px solid var(--border); background: rgba(255,255,255,.04);
  cursor:pointer; font-weight:700; font-size:12px; color:var(--muted);
}
.tab.on{ color:var(--ink); border-color: rgba(54,214,255,.35); background: rgba(54,214,255,.10); }
.gallery{ display:grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap:10px; }
.thumb{ border-radius:16px; overflow:hidden; border:1px solid rgba(255,255,255,.14); background: rgba(0,0,0,.25); cursor:pointer; }
.thumb img{ width:100%; height:110px; object-fit:cover; display:block; }
.player{ display:flex; flex-direction:column; gap:10px; }
.player video{ width:100%; border-radius:18px; border:1px solid rgba(255,255,255,.14); background: rgba(0,0,0,.35); }
@media (max-width: 1100px){ .grid{ grid-template-columns: 1fr; } .gallery{ grid-template-columns: repeat(3, minmax(0, 1fr)); } }
@media (max-width: 620px){ .form{ grid-template-columns: 1fr; } .gallery{ grid-template-columns: repeat(2, minmax(0, 1fr)); } }

body.theme-trek{ --bg0:#0f0b12; --bg1:#1a1120; --card:#231425; --ink:#ffe8c6; --muted:#f3b567; --accent:#ff9f1c; --accent2:#c77dff; --good:#7ae582; --warn:#ffd166; --bad:#ff595e; }
body.theme-matrix{ --bg0:#000a00; --bg1:#001300; --card:#001d00; --ink:#b6ffb6; --muted:#66ff66; --accent:#33ff33; --accent2:#00b300; }
body.theme-cyber{ --bg0:#070612; --bg1:#0d0b22; --card:#121036; --ink:#e9e2ff; --muted:#b7a8ff; --accent:#7c4dff; --accent2:#00e5ff; }
body.theme-ocean{ --bg0:#03121a; --bg1:#052331; --card:#072a3a; --ink:#e9fbff; --muted:#8bd3dd; --accent:#2ec4b6; --accent2:#00bbf9; }
body.theme-sunset{ --bg0:#120612; --bg1:#1a0823; --card:#2a0c2d; --ink:#fff0f7; --muted:#ffb3c6; --accent:#ff5d8f; --accent2:#fca311; }
body.theme-ice{ --bg0:#06111a; --bg1:#081e2d; --card:#0a2536; --ink:#eaf6ff; --muted:#a7d8ff; --accent:#4cc9f0; --accent2:#bde0fe; }
body.theme-ember{ --bg0:#110805; --bg1:#1c0f08; --card:#2b140b; --ink:#ffe6d5; --muted:#ffb385; --accent:#ff7a18; --accent2:#ffd60a; }
body.theme-forest{ --bg0:#071007; --bg1:#0c1b0c; --card:#102510; --ink:#eaffea; --muted:#a0d6a0; --accent:#55a630; --accent2:#80b918; }
body.theme-mono{ --bg0:#0b0b0b; --bg1:#121212; --card:#191919; --ink:#f3f3f3; --muted:#c8c8c8; --accent:#ffffff; --accent2:#9a9a9a; }
"""

def _html() -> str:
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Time-Lapse Web Pro</title>
  <style>{_theme_css()}</style>
</head>
<body class="theme-neon">
  <div class="container">
    <div class="topbar">
      <div class="brand">
        <div style="width:14px;height:14px;border-radius:999px;background:linear-gradient(180deg,var(--accent),var(--accent2));"></div>
        <div style="display:flex;flex-direction:column;gap:2px;">
          <div style="font-weight:900; letter-spacing:.5px;">Time-Lapse Web Pro</div>
          <div class="small" id="host_line">Host: —</div>
        </div>
      </div>
      <div class="row">
        <span class="badge"><span class="dot" id="run_dot"></span><span id="run_label">STOPPED</span></span>
        <button class="btn" id="btn_start">Start</button>
        <button class="btn secondary" id="btn_stop">Stop</button>
        <button class="btn ghost" id="btn_new">New Project</button>
        <select id="theme_pick" style="min-width:180px"></select>
      </div>
    </div>

    <div class="grid">
      <div class="card">
        <div class="hd">
          <h2>Live Preview</h2>
          <div class="row">
            <span class="pill" id="pill_project">Project: —</span>
            <span class="pill" id="pill_saved">Saved: 0</span>
            <span class="pill" id="pill_last">Last: —</span>
          </div>
        </div>
        <div class="bd">
          <div class="preview-wrap">
            <img id="preview" src="/preview.jpg" alt="preview"/>
            <div class="preview-overlay">
              <span class="pill" id="pill_res">—</span>
              <span class="pill" id="pill_int">—</span>
              <span class="pill" id="pill_fmt">—</span>
              <span class="pill" id="pill_err">—</span>
            </div>
          </div>

          <div style="margin-top:12px;" class="tabs">
            <div class="tab on" data-tab="gallery">Gallery</div>
            <div class="tab" data-tab="mp4">MP4</div>
            <div class="tab" data-tab="files">File Manager</div>
          </div>

          <div id="tab_gallery" style="margin-top:12px;">
            <div class="row" style="justify-content:space-between; align-items:center;">
              <div class="small" id="gal_status">—</div>
              <div class="row"><button class="btn ghost" id="btn_refresh_gallery">Refresh</button></div>
            </div>
            <div style="margin-top:10px;" class="gallery" id="gallery_grid"></div>
          </div>

          <div id="tab_mp4" style="margin-top:12px; display:none;">
            <div class="row" style="justify-content:space-between; align-items:center;">
              <div class="small" id="mp4_status">—</div>
              <div class="row">
                <select id="mp4_ext">
                  <option value="png">Use PNG images</option>
                  <option value="jpg">Use JPG images</option>
                </select>
                <input id="mp4_fps" type="number" min="1" max="120" value="30" style="width:100px" />
                <button class="btn" id="btn_make_mp4">Create MP4</button>
                <button class="btn ghost" id="btn_refresh_mp4">Refresh</button>
              </div>
            </div>
            <div class="player" style="margin-top:10px;">
              <video id="vid" controls playsinline></video>
              <div class="row" style="align-items:center;">
                <label class="small">Speed</label>
                <input id="speed" type="range" min="0.1" max="4" step="0.1" value="1" style="flex:1" />
                <span class="pill" id="speed_label">1.0x</span>
                <label class="small"><input type="checkbox" id="loop"/> Loop</label>
              </div>
              <div class="row" id="mp4_list"></div>
            </div>
          </div>

          <div id="tab_files" style="margin-top:12px; display:none;">
            <div class="row" style="justify-content:space-between; align-items:center;">
              <div class="small">Cleanup tools for projects/images/mp4.</div>
              <div class="row">
                <button class="btn ghost" id="btn_refresh_files">Refresh</button>
                <button class="btn danger" id="btn_delete_project">Delete Project</button>
              </div>
            </div>
            <div style="margin-top:10px;" class="msg" id="files_status">—</div>
            <div class="row" style="margin-top:10px;">
              <div style="flex:1; min-width:280px;">
                <div class="small">Images (click to delete):</div>
                <div class="msg" id="file_images" style="max-height:220px; overflow:auto;"></div>
              </div>
              <div style="flex:1; min-width:280px;">
                <div class="small">MP4 (click to delete):</div>
                <div class="msg" id="file_mp4" style="max-height:220px; overflow:auto;"></div>
              </div>
            </div>
          </div>

        </div>
      </div>

      <div class="card">
        <div class="hd">
          <h2>Controls</h2>
          <span class="badge">Autosave every <b id="auto_n">—</b> images</span>
        </div>
        <div class="bd">
          <div class="form">
            <div class="field"><label>Project name</label><input id="project_name" placeholder="project"/></div>
            <div class="field"><label>Camera index</label><input id="camera_index" type="number" min="0" max="16" value="0"/></div>

            <div class="field"><label>Resolution</label><select id="resolution_key"></select></div>
            <div class="field"><label>Interval seconds</label><input id="interval_seconds" type="number" min="0.2" step="0.1"/></div>

            <div class="field"><label>Image format</label>
              <select id="image_ext"><option value="png">PNG</option><option value="jpg">JPG</option></select>
            </div>
            <div class="field"><label>JPG quality (1-100)</label><input id="jpg_quality" type="number" min="1" max="100"/></div>

            <div class="field"><label>Retention limit (0 disables)</label><input id="retention_limit" type="number" min="0" step="1"/></div>
            <div class="field"><label>Autosave every N images</label><input id="autosave_every" type="number" min="1" step="1"/></div>

            <div class="field"><label>Overlay timestamp</label>
              <select id="overlay_timestamp"><option value="true">On</option><option value="false">Off</option></select>
            </div>
            <div class="field"><label>Overlay project</label>
              <select id="overlay_project"><option value="true">On</option><option value="false">Off</option></select>
            </div>

            <div class="field"><label>Text scale</label><input id="text_scale" type="number" min="0.4" max="3" step="0.1"/></div>
            <div class="field"><label>Text thickness</label><input id="text_thickness" type="number" min="1" max="6" step="1"/></div>
          </div>

          <div class="row" style="margin-top:12px;">
            <button class="btn ghost" id="btn_apply">Save Settings</button>
            <button class="btn ghost" id="btn_reload">Reload</button>
          </div>

          <div class="msg" style="margin-top:12px;" id="msg">Ready.</div>

          <div style="margin-top:12px;" class="row">
            <div class="badge" style="flex:1; justify-content:space-between;">
              <span>Current project id:</span>
              <b id="proj_id">—</b>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

<script>
function qs(id){ return document.getElementById(id); }
function setMsg(t){ qs("msg").textContent = t; }
function setHostLine(){ qs("host_line").textContent = "Host: " + window.location.origin; }
function isEditingForm(){
  const ae = document.activeElement;
  if (!ae) return false;
  const t = ae.tagName;
  return (t === 'INPUT' || t === 'SELECT' || t === 'TEXTAREA');
}

window.__TL_PREVIEW_PAUSED = false;
function __tl_pause_preview(on){ window.__TL_PREVIEW_PAUSED = !!on; }

let STATE = null;
let CURRENT_TAB = "gallery";

async function apiGet(url){
  const r = await fetch(url, {cache:"no-store"});
  return await r.json();
}
async function apiPost(url, payload){
  const r = await fetch(url, {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(payload||{})});
  return await r.json();
}

function setRunningUI(running){
  const dot = qs("run_dot");
  dot.classList.toggle("on", !!running);
  qs("run_label").textContent = running ? "RUNNING" : "STOPPED";
}

function tickPreview(){
  if (window.__TL_PREVIEW_PAUSED) return;
  qs("preview").src = "/preview.jpg?t=" + Date.now();
}
setInterval(tickPreview, 900);

function setTheme(id){
  document.body.className = "theme-" + id;
  try{ localStorage.setItem("tl_theme", id); }catch(e){}
}

async function refreshState(){
  const r = await apiGet("/api/state");
  STATE = r;

  // Themes
  const tp = qs("theme_pick");
  if (tp.options.length === 0){
    (r.themes||[]).forEach(t=>{
      const o = document.createElement("option");
      o.value = t.id; o.textContent = t.name;
      tp.appendChild(o);
    });
    tp.onchange = ()=>{ setTheme(tp.value); apiPost("/api/config", {theme: tp.value}); setMsg("Theme set to: "+tp.value); };
  }
  // Apply stored theme
  let local = null;
  try{ local = localStorage.getItem("tl_theme"); }catch(e){}
  const th = local || (r.config.theme || "neon");
  tp.value = th;
  setTheme(th);

  // Resolutions
  const rd = qs("resolution_key");
  if (rd.options.length === 0){
    (r.resolutions||[]).forEach(x=>{
      const o = document.createElement("option");
      o.value = x.k; o.textContent = x.k + ": " + x.w + "x" + x.h;
      rd.appendChild(o);
    });
  }

  if (!isEditingForm()){
    qs("project_name").value = r.config.project_name || "";
    qs("camera_index").value = r.config.camera_index;
    rd.value = r.config.resolution_key;
    qs("interval_seconds").value = r.config.interval_seconds;
    qs("image_ext").value = (r.config.image_ext === "jpeg") ? "jpg" : (r.config.image_ext || "png");
    qs("jpg_quality").value = r.config.jpg_quality;
    qs("retention_limit").value = r.config.retention_limit;
    qs("autosave_every").value = r.config.autosave_every;
    qs("overlay_timestamp").value = String(!!r.config.overlay_timestamp);
    qs("overlay_project").value = String(!!r.config.overlay_project);
    qs("text_scale").value = r.config.text_scale;
    qs("text_thickness").value = r.config.text_thickness;
  }

  setRunningUI(!!r.running);
  qs("proj_id").textContent = r.current_project_id || "—";
  qs("pill_project").textContent = "Project: " + (r.current_project_id || "—");
  qs("pill_saved").textContent = "Saved: " + (r.total_saved ?? 0);
  qs("pill_last").textContent = "Last: " + (r.last_saved || "—");
  qs("pill_res").textContent = "Res: " + r.config.resolution_key;
  qs("pill_int").textContent = "Int: " + (r.config.interval_seconds + "s");
  qs("pill_fmt").textContent = "Fmt: " + (r.config.image_ext || "png");
  qs("pill_err").textContent = r.last_error ? ("Error: " + r.last_error) : "OK";
  qs("auto_n").textContent = r.config.autosave_every;

  // Requested behavior: pause preview while running
  __tl_pause_preview(!!r.running);

  if (CURRENT_TAB === "gallery") await refreshGallery(false);
  if (CURRENT_TAB === "mp4") await refreshMp4(false);
  if (CURRENT_TAB === "files") await refreshFiles(false);
}

function setTab(tab){
  CURRENT_TAB = tab;
  document.querySelectorAll(".tab").forEach(x=>x.classList.toggle("on", x.dataset.tab===tab));
  qs("tab_gallery").style.display = (tab==="gallery") ? "" : "none";
  qs("tab_mp4").style.display = (tab==="mp4") ? "" : "none";
  qs("tab_files").style.display = (tab==="files") ? "" : "none";
  if (tab==="gallery") refreshGallery(true);
  if (tab==="mp4") refreshMp4(true);
  if (tab==="files") refreshFiles(true);
}

async function refreshGallery(showMsg=true){
  if (!STATE || !STATE.current_project_id){
    qs("gal_status").textContent = "No project selected.";
    qs("gallery_grid").innerHTML = "";
    return;
  }
  const pid = STATE.current_project_id;
  const r = await apiGet(`/api/project/${pid}/images?limit=50`);
  qs("gal_status").textContent = `Project: ${pid} | Images shown: ${r.count}`;
  const grid = qs("gallery_grid");
  grid.innerHTML = "";
  (r.images||[]).forEach(fn=>{
    const d = document.createElement("div");
    d.className = "thumb";
    const img = document.createElement("img");
    img.src = `/www/projects/${pid}/images/${encodeURIComponent(fn)}`;
    img.loading = "lazy";
    d.appendChild(img);
    d.onclick = ()=>{ window.open(img.src, "_blank"); };
    grid.appendChild(d);
  });
  if (showMsg) setMsg("Gallery refreshed.");
}

async function refreshMp4(showMsg=true){
  if (!STATE || !STATE.current_project_id){
    qs("mp4_status").textContent = "No project selected.";
    qs("mp4_list").innerHTML = "";
    return;
  }
  const pid = STATE.current_project_id;
  const r = await apiGet(`/api/project/${pid}/mp4`);
  qs("mp4_status").textContent = `Project: ${pid} | MP4: ${r.count}`;
  const list = qs("mp4_list");
  list.innerHTML = "";
  (r.mp4||[]).forEach(fn=>{
    const b = document.createElement("button");
    b.className = "btn ghost";
    b.textContent = fn;
    b.onclick = ()=>{
      const url = `/www/projects/${pid}/mp4/${encodeURIComponent(fn)}`;
      const v = qs("vid");
      v.src = url;
      v.playbackRate = parseFloat(qs("speed").value||"1");
      v.loop = qs("loop").checked;
      v.play().catch(()=>{});
      setMsg("Playing: " + fn);
    };
    list.appendChild(b);
  });
  if (showMsg) setMsg("MP4 list refreshed.");
}

async function refreshFiles(showMsg=true){
  if (!STATE || !STATE.current_project_id){
    qs("files_status").textContent = "No project selected.";
    qs("file_images").textContent = "";
    qs("file_mp4").textContent = "";
    return;
  }
  const pid = STATE.current_project_id;
  const imgs = await apiGet(`/api/project/${pid}/images?limit=200`);
  const mp4 = await apiGet(`/api/project/${pid}/mp4`);
  qs("files_status").textContent = `Project: ${pid} | images: ${imgs.count} | mp4: ${mp4.count} (click a filename to delete)`;

  const imgBox = qs("file_images"); imgBox.innerHTML = "";
  (imgs.images||[]).forEach(fn=>{
    const a = document.createElement("div");
    a.textContent = fn;
    a.style.cursor = "pointer";
    a.style.padding = "2px 0";
    a.onclick = async ()=>{
      if (!confirm("Delete image: " + fn + " ?")) return;
      await apiPost(`/api/project/${pid}/delete_image`, {filename: fn});
      setMsg("Deleted image: " + fn);
      refreshFiles(false);
      refreshGallery(false);
    };
    imgBox.appendChild(a);
  });

  const mp4Box = qs("file_mp4"); mp4Box.innerHTML = "";
  (mp4.mp4||[]).forEach(fn=>{
    const a = document.createElement("div");
    a.textContent = fn;
    a.style.cursor = "pointer";
    a.style.padding = "2px 0";
    a.onclick = async ()=>{
      if (!confirm("Delete mp4: " + fn + " ?")) return;
      await apiPost(`/api/project/${pid}/delete_mp4`, {filename: fn});
      setMsg("Deleted mp4: " + fn);
      refreshFiles(false);
      refreshMp4(false);
    };
    mp4Box.appendChild(a);
  });

  if (showMsg) setMsg("File manager refreshed.");
}

// Buttons
qs("btn_apply").onclick = async ()=>{
  const payload = {
    project_name: qs("project_name").value,
    camera_index: parseInt(qs("camera_index").value||"0",10),
    resolution_key: parseInt(qs("resolution_key").value||"4",10),
    interval_seconds: parseFloat(qs("interval_seconds").value||"60"),
    image_ext: qs("image_ext").value,
    jpg_quality: parseInt(qs("jpg_quality").value||"92",10),
    retention_limit: parseInt(qs("retention_limit").value||"0",10),
    autosave_every: parseInt(qs("autosave_every").value||"5",10),
    overlay_timestamp: (qs("overlay_timestamp").value === "true"),
    overlay_project: (qs("overlay_project").value === "true"),
    text_scale: parseFloat(qs("text_scale").value||"1"),
    text_thickness: parseInt(qs("text_thickness").value||"2",10),
  };
  await apiPost("/api/config", payload);
  setMsg("Settings saved.");
  await refreshState();
};

qs("btn_reload").onclick = async ()=>{ setMsg("Reloading..."); await refreshState(); };

qs("btn_new").onclick = async ()=>{
  const name = prompt("New project name:", qs("project_name").value || "project");
  if (!name) return;
  const r = await apiPost("/api/new_project", {project_name: name});
  if (r.ok){
    setMsg("New project created: " + r.project_id);
    await refreshState();
    await refreshGallery(false);
  } else {
    setMsg("Failed.");
  }
};

qs("btn_start").onclick = async ()=>{
  setMsg("Starting...");
  const r = await apiPost("/api/start", {});
  if (r.ok){
    __tl_pause_preview(true);
    setMsg("Started. Project: " + r.project_id);
    await refreshState();
  } else {
    setMsg("Start failed.");
  }
};

qs("btn_stop").onclick = async ()=>{
  setMsg("Stopping...");
  const r = await apiPost("/api/stop", {});
  if (r.ok){
    __tl_pause_preview(false);
    setMsg("Stopped.");
    await refreshState();
  } else {
    setMsg("Stop failed.");
  }
};

qs("btn_refresh_gallery").onclick = ()=>refreshGallery(true);
qs("btn_refresh_mp4").onclick = ()=>refreshMp4(true);
qs("btn_refresh_files").onclick = ()=>refreshFiles(true);

qs("btn_make_mp4").onclick = async ()=>{
  if (!STATE || !STATE.current_project_id){
    setMsg("No project selected.");
    return;
  }
  const pid = STATE.current_project_id;
  const fps = parseInt(qs("mp4_fps").value||"30",10);
  const ext = qs("mp4_ext").value;
  setMsg("Building MP4...");
  const r = await apiPost(`/api/project/${pid}/build_mp4`, {fps: fps, ext: ext});
  if (r.ok){
    setMsg("MP4 created: " + r.mp4);
    await refreshMp4(false);
  } else {
    setMsg("MP4 failed: " + (r.error||"unknown"));
  }
};

qs("speed").oninput = ()=>{
  const v = qs("vid");
  const sp = parseFloat(qs("speed").value||"1");
  qs("speed_label").textContent = sp.toFixed(1) + "x";
  v.playbackRate = sp;
};
qs("loop").onchange = ()=>{
  qs("vid").loop = qs("loop").checked;
};

qs("btn_delete_project").onclick = async ()=>{
  if (!STATE || !STATE.current_project_id) return;
  const pid = STATE.current_project_id;
  if (!confirm("DELETE ENTIRE PROJECT: " + pid + " ?")) return;
  const r = await apiPost(`/api/project/${pid}/delete_project`, {});
  if (r.ok){
    setMsg("Project deleted: " + pid);
    await refreshState();
    qs("gallery_grid").innerHTML = "";
    qs("mp4_list").innerHTML = "";
    qs("file_images").textContent = "";
    qs("file_mp4").textContent = "";
  } else {
    setMsg("Delete failed.");
  }
};

// Tabs
document.querySelectorAll(".tab").forEach(t=>{ t.onclick = ()=> setTab(t.dataset.tab); });

setHostLine();
setTimeout(async ()=>{
  await refreshState();
  setInterval(()=>{ refreshState().catch(()=>{}); }, 1500);
}, 50);
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    return HTMLResponse(_html())
