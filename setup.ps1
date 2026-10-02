# Lipflow GPT RU for Windows: lightweight ChatGPT mode by default; legacy English VSR is optional.
#   powershell -ExecutionPolicy Bypass -File setup.ps1 [-Legacy] [-Samples] [-NoShortcut]
param([switch]$Legacy, [switch]$Samples, [switch]$NoShortcut)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"  # Invoke-WebRequest is 10x slower with the progress bar
Set-Location $PSScriptRoot
$env:PYTHONUTF8 = "1"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv (Python package manager)..."
    powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}
if ($Legacy) {
    uv sync --extra legacy
} else {
    uv sync
}
if ($LASTEXITCODE -ne 0) { throw "uv sync failed" }

function Get-File($url, $dest) {
    if ((Test-Path $dest) -and ((Get-Item $dest).Length -gt 0)) { Write-Host "ok  $dest"; return }
    New-Item -ItemType Directory -Force -Path (Split-Path $dest) | Out-Null
    Write-Host "get $dest"
    curl.exe -fL --progress-bar -o "$dest.part" $url
    if ($LASTEXITCODE -ne 0) { throw "download failed: $url" }
    Move-Item -Force "$dest.part" $dest
}

# MediaPipe face landmarker is the only model required by ChatGPT mode.
Get-File "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" `
    "models/face_landmarker.task"

if ($Legacy) {
    $HF = "https://huggingface.co"
    # Optional original English Auto-AVSR stack (~1.2 GB; research-use model weights).
    Get-File "$HF/Amanvir/LRS3_V_WER19.1/resolve/main/model.json" "models/vsr/model.json"
    Get-File "$HF/Amanvir/LRS3_V_WER19.1/resolve/main/model.pth"  "models/vsr/model.pth"
    Get-File "$HF/Amanvir/lm_en_subword/resolve/main/model.json"  "models/lm/model.json"
    Get-File "$HF/Amanvir/lm_en_subword/resolve/main/model.pth"   "models/lm/model.pth"
    Get-File "https://github.com/mpc001/auto_avsr/raw/main/spm/unigram/unigram5000.model" "models/lm/unigram5000.model"
}

if ($Samples) {
    # Public-domain White House weekly addresses (Wikimedia Commons), used by tests/test_pipeline.py
    $C = "https://upload.wikimedia.org/wikipedia/commons/transcoded"
    Get-File "$C/c/ce/2016-03-12_President_Obama%27s_Weekly_Address.webm/2016-03-12_President_Obama%27s_Weekly_Address.webm.360p.mpeg4.mov" "samples/2016-03-12.mov"
    Get-File "$C/2/29/2017-01-07_President_Obama%27s_Weekly_Address.webm/2017-01-07_President_Obama%27s_Weekly_Address.webm.360p.mpeg4.mov" "samples/2017-01-07.mov"
}

if (-not $NoShortcut) {
    # Start menu entry that runs the tray app without a console window
    $data = Join-Path $env:APPDATA "LipflowGPT"
    New-Item -ItemType Directory -Force -Path $data | Out-Null
    $ico = Join-Path $data "LipflowGPT.ico"
    uv run python -c "import sys; from lipflow.win.hud import tray_image; tray_image(size=256).save(sys.argv[1], sizes=[(16,16),(32,32),(48,48),(256,256)])" $ico
    $lnk = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\Lipflow GPT RU.lnk"
    $sh = (New-Object -ComObject WScript.Shell).CreateShortcut($lnk)
    $sh.TargetPath = Join-Path $PSScriptRoot ".venv\Scripts\pythonw.exe"
    $sh.Arguments = "-X utf8 -m lipflow"
    $sh.WorkingDirectory = $PSScriptRoot
    $sh.IconLocation = $ico
    $sh.Description = "Silent Russian and English dictation with ChatGPT"
    $sh.Save()
    Write-Host "ok  Start menu shortcut: $lnk"
}

Write-Host ""
Write-Host "Done. Open Lipflow GPT RU from the Start menu (it lives in the system tray)."
Write-Host "Then choose Continue with ChatGPT and authorize your ChatGPT Plus/Pro account."
if ($Legacy) { Write-Host "Legacy English VSR was also installed." }
