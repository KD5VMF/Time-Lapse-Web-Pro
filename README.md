# Time-Lapse Web Pro (FastAPI + OpenCV)

A reliable, long-running **web-controlled time-lapse** system for Linux using:
- **OpenCV** for camera capture + overlays
- **FastAPI/Uvicorn** for the web UI + API
- **ffmpeg** for MP4 creation

Designed for “set it and forget it” capture that can run **for months** with autosave + safe file organization.

---

## Features

### Core
- Live **preview** (`/preview.jpg`)
- Start/Stop capture from the browser
- Create a **unique project folder** each run (no overwrites)
- Resolution presets (10 options)
- Interval in seconds
- Save as **PNG** or **JPG**
- Optional overlay: timestamp + project label

### Reliability
- **Single camera worker thread** to avoid lockups/double-open issues
- Autosave state every **N images** (default: **5**)
- Optional retention limit (keep last N images; 0 disables)

### MP4 + Playback
- MP4 builder (ffmpeg concat list for stable ordering)
- MP4 player with:
  - speed slider
  - loop option

### File Manager
- Delete images
- Delete MP4
- Delete whole project

### Themes
10 built-in themes including **Star Trek LCARS**.

---

## Folder Layout

```
projects/<project_id>/
  images/
  mp4/

www/projects/<project_id>/
  images/  -> symlink to projects/<project_id>/images
  mp4/     -> symlink to projects/<project_id>/mp4
```

---

## Install & Run

```bash
./scripts/install.sh
./start.sh
```

The server prints a LAN URL you can open from another machine.

---

## Camera Tips

List devices:

```bash
ls -l /dev/video*
v4l2-ctl --list-devices
```

If permissions fail:

```bash
sudo usermod -aG video $USER
# log out/in
```

---

## Run as a Service (optional)

Edit and install the included unit:

`./scripts/timelapse-web-pro.service`

```bash
sudo cp scripts/timelapse-web-pro.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now timelapse-web-pro
```

---

## Cleanup (safe)

Removes caches and backups (does not delete `projects/`):

```bash
./cleanup.sh
```

---

## License
MIT
