# 📸 Time-Lapse Web Pro 🚀  
A **reliable, long-running** time-lapse capture system with a **futuristic web UI**, live preview, project folders, gallery, MP4 creation + player, themes, and file cleanup tools.

---

## ✨ What this project does

✅ **Live camera preview** (for aiming/focus/testing)  
✅ **Start / Stop** capture from the browser  
✅ **Projects** with unique folders (organized, safe for long runs)  
✅ **Gallery** to browse images  
✅ **Create MP4** from captured frames (FFmpeg)  
✅ **MP4 player** with **speed control** + **loop forever**  
✅ **Themes** (pick from multiple looks)  
✅ **Reliability features** for long-running use (autosaves / safe writes)  
✅ **Cleanup tools** for removing old junk/test scripts without breaking the working app  

---

## 🧱 Repo layout

- `src/timelapse_web_pro.py` → the main app (FastAPI + OpenCV + UI)
- `scripts/install.sh` → installs system packages + creates venv + installs Python deps
- `start.sh` → starts the web UI (prints the LAN URL)
- `cleanup.sh` → removes junk/backups/test scripts (safe cleanup)
- `scripts/timelapse-web-pro.service` → optional systemd service to auto-start at boot
- `projects/` → your saved projects (each project has its own folder)
- `www/` / `web/` → static assets (if present)

---

## ✅ Requirements

### 🐧 OS
- Ubuntu / Debian recommended

### 🎥 Hardware
- USB camera or any V4L2 camera device
- You should see `/dev/video0`

### 📦 Packages
The install script will install what you need, including:
- `ffmpeg` (MP4 creation)
- `v4l-utils` (camera debugging)
- `python3-venv`, build tools

---

## ⚡ Quick Start (recommended)

### 1) Clone
```bash
cd ~
git clone https://github.com/<YOURNAME>/time-lapse-web-pro.git
cd time-lapse-web-pro
```

### 2) Install dependencies + create venv
```bash
bash scripts/install.sh
```

### 3) Run it
```bash
./start.sh
```

You’ll see something like:
- `Local: http://127.0.0.1:8090`
- `LAN:   http://192.168.x.x:8090`

Open the **LAN URL** from another PC/phone on your network.

---

## 🌐 How to connect to the Web UI

✅ Same machine:
- `http://127.0.0.1:8090`

✅ From another device on your LAN:
- `http://<LAN_IP>:8090`  (the app prints this for you)

> 🔒 **Security note:** This is meant for LAN use. Don’t expose it directly to the Internet unless you know what you’re doing (reverse proxy + auth).

---

## 🕹️ Using the Web UI

### 🎯 Live Preview
- Used for aiming/focus/testing.
- **Auto-stops when capture starts**, and **returns when capture stops** (to avoid camera conflicts).

### 🧪 Create a project
- Enter a project name
- Choose resolution
- Choose interval (HH:MM:SS)
- Choose image format (if your UI supports it)

### ▶️ Start capture
- Press **Start**
- The status at the top should change to **Running**
- Images start saving into the project folder

### ⏹️ Stop capture
- Press **Stop**
- Status returns to **Stopped**
- Live preview resumes

### 🖼️ Gallery
- Pick project
- Browse frames
- Use file tools (delete image / delete folder / etc. if enabled)

### 🎞️ Create MP4
- Choose your project
- Click **Create MP4**
- Once created, use the **Player** section

### 🎬 MP4 Player (speed + loop)
- Set playback speed
- Enable **Loop**
- Scrub timeline

---

## 🧹 Cleaning up junk/test scripts (safe)

This repo often includes backup scripts from development. To clean them safely:

```bash
./cleanup.sh
```

✅ Keeps your working app + projects  
✅ Removes old patch scripts/backups you no longer need  
✅ Doesn’t delete your important folders unless you choose options that do  

---

## 🔁 Run at boot (optional systemd)

### 1) Install service
```bash
sudo cp scripts/timelapse-web-pro.service /etc/systemd/system/timelapse-web-pro.service
sudo systemctl daemon-reload
sudo systemctl enable timelapse-web-pro
sudo systemctl start timelapse-web-pro
```

### 2) Check status/logs
```bash
sudo systemctl status timelapse-web-pro --no-pager
journalctl -u timelapse-web-pro -f
```

---

## 🛠️ Troubleshooting

### ❌ “No camera” / Preview is blank
Check device exists:
```bash
ls -l /dev/video*
```

Check camera capabilities:
```bash
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --all
```

### 🔐 Permission denied on /dev/video0
Add your user to the `video` group:
```bash
sudo usermod -aG video $USER
newgrp video
```
(Or log out/in.)

### 🎞️ MP4 creation fails: “ffmpeg not found”
Install ffmpeg:
```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

### 🌍 Can’t connect from another device
- Make sure you’re using the LAN IP shown by `start.sh`
- Confirm port is open on local firewall:
```bash
sudo ufw status
```
If needed (LAN only):
```bash
sudo ufw allow 8090/tcp
```

### 🧨 If something gets weird
The safest “restore” flow is:
```bash
# stop the server
# Ctrl+C

# reinstall venv deps
bash scripts/install.sh

# run again
./start.sh
```

---

## 🧠 Notes on reliability (long runs)

This project is designed to run for **months/years**:
- Uses safe writes (avoid partial/corrupt images)
- Saves state/config
- Avoids camera conflicts (preview pauses while capture runs)

Still, you should:
✅ Use a stable storage disk  
✅ Ensure enough free space  
✅ Consider retention cleanup (manual or scheduled)

---

## 📌 Roadmap ideas (optional)
- 🕒 Scheduled start/stop  
- 🎯 Motion-trigger capture  
- 🎚️ Exposure/brightness controls (camera-dependent)  
- 🧹 Automatic retention rules  

---

## 📄 License
Pick a license and drop it in `LICENSE` (MIT is common).

---

## 🙌 Credits
Built for makers who want a **hands-off, reliable time-lapse rig** with a **fun futuristic UI**.
