"""Lipflow for Windows: a tray icon. Tap the key to start, mouth the words, tap again to finish.

Threads: tk owns the main thread (overlay, setup window); pynput's hook thread reports the key;
pystray runs the tray menu on its own thread; one model thread reads lips. Everything that touches
tk goes through ui(), which queues it for the main thread.
"""
from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass

from ..camera import Camera, Recording, mouth_view
from ..chatgpt_auth import ChatGPTAuthError, ChatGPTSession
from ..chatgpt_vision import ChatGPTInferenceError, ChatGPTLipReader
from ..dictation import (
    HISTORY, JOIN_WINDOW, MAX_SECONDS, PREVIEW_EVERY, TAIL_SECONDS, clip_problem, keep_clip, load_settings,
    log_history, rois_for, save_settings, train_on_face,
)
from ..paths import HOME
from .hotkey import DEFAULT_KEY, KEYS, PushToTalk
from .hud import HUD, tray_image
from .paste import copy_text, paste_text

LOG = os.path.join(HOME, "Lipflow.log")
CAMERAS = 4  # Windows can't name cameras through OpenCV: offer the first few by number
RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"


@dataclass
class Options:
    key: str = DEFAULT_KEY
    beam: int = 4
    backend: str = "auto"
    camera: "int | str" = "auto"
    paste: bool = True
    live_preview: bool = True
    onboard: bool = False


def key_label(key: str) -> str:
    return key.replace("_", " ").title().replace("Control", "Ctrl")


class Lipflow:
    def __init__(self, opts: Options):
        self.opts = opts
        self.settings = load_settings()
        if opts.key == DEFAULT_KEY and self.settings.get("key") in KEYS:
            opts.key = self.settings["key"]
        self.root = tk.Tk()
        self.root.withdraw()
        self._q: "queue.Queue" = queue.Queue()
        self.reader = None
        self.cleaner = None
        self.chatgpt = ChatGPTSession()
        self.gpt_reader = ChatGPTLipReader(self.chatgpt)
        self.settings.setdefault("engine", "chatgpt")
        self.settings.setdefault("language", "ru")
        self.jobs: "queue.Queue" = queue.Queue()
        self.session = 0          # bumps on every start/cancel so stale previews are dropped
        self.preview_busy = False
        self.last_output = ""
        self.last_paste_at = 0.0
        self.context: list[str] = []
        self.hands_free = False
        self.pending_stop = None
        from ..mic import Mic
        self.mic = Mic()
        self.av_reader = None
        self._ui_busy = False
        self.onboarding = None
        self.onboarding_text = ""
        self.setup = None
        self.loading = True
        self.state_text = "Loading model…"
        self.ctx = None
        cam = opts.camera if opts.camera != "auto" else self.settings.get("camera", "auto")
        self.camera = Camera(cam, on_frame=self.on_frame)

    @property
    def key_name(self) -> str:
        return key_label(self.opts.key)

    # -- threading -------------------------------------------------------------------------
    def ui(self, fn, *args, **kw):
        """Run fn on the tk thread (safe from any thread)."""
        self._q.put((fn, args, kw))

    def _pump(self):
        try:
            while True:
                fn, args, kw = self._q.get_nowait()
                try:
                    fn(*args, **kw)
                except Exception:
                    import traceback
                    traceback.print_exc()
        except queue.Empty:
            pass
        self.root.after(15, self._pump)

    # -- setup -----------------------------------------------------------------------------
    def start(self):
        self.hud = HUD(self.root)
        self._install_key()
        self._build_tray()
        self.hud.show("reading", "Lipflow", "Loading the lip-reading model…")
        # ChatGPT mode has no local VSR warmup; pre-open the camera once so the first
        # real dictation does not lose its opening words to DirectShow/MediaPipe cold start.
        if self.settings.get("engine", "chatgpt") == "chatgpt":
            self.camera.ensure_open()
        threading.Thread(target=self._worker, name="lipflow-model", daemon=True).start()
        self.jobs.put(("load",))
        self.root.after(15, self._pump)

    def _install_key(self):
        # The hook thread only queues: all state changes happen on the tk thread.
        self.ptt = PushToTalk(self.opts.key,
                              lambda hands_free: self.ui(self.on_start, hands_free),
                              lambda: self.ui(self.on_stop),
                              lambda silent=False: self.ui(self.on_cancel, silent))
        self.ptt.install()

    def _build_tray(self):
        import pystray
        from pystray import Menu, MenuItem as Item

        def toggle(name, default=True, then=None):
            def act(icon, item):
                self.settings[name] = not self.settings.get(name, default)
                save_settings(self.settings)
                if then:
                    then()
            return Item(name_labels[name], act, checked=lambda item: self.settings.get(name, default))

        name_labels = {"whisper": "Whisper mode (lips + a soft whisper)",
                       "use_context": "Use the window title for names",
                       "save_clips": "Keep my last 100 clips (local)"}

        def pick_camera(value):
            return Item("Automatic" if value == "auto" else f"Camera {value + 1}",
                        lambda icon, item: self.ui(self._pick_camera, value),
                        checked=lambda item: self.settings.get("camera", "auto") == value, radio=True)

        def pick_key(name):
            return Item(key_label(name), lambda icon, item: self.ui(self._pick_key, name),
                        checked=lambda item: self.opts.key == name, radio=True)

        def pick_language(value, label):
            return Item(label, lambda icon, item: self.ui(self._pick_language, value),
                        checked=lambda item: self.settings.get("language", "ru") == value, radio=True)

        def pick_engine(value, label):
            return Item(label, lambda icon, item: self.ui(self._pick_engine, value),
                        checked=lambda item: self.settings.get("engine", "chatgpt") == value, radio=True)

        menu = Menu(
            Item(lambda item: self.state_text, None, enabled=False),
            Item(lambda item: f"Tap {self.key_name} to start · tap again to finish", None, enabled=False),
            Item(lambda item: f"ChatGPT: {self.chatgpt.label()}", None, enabled=False),
            Item(lambda item: f"Recognition: {self.settings.get('engine', 'chatgpt')} / {self.settings.get('language', 'ru').upper()}", None, enabled=False),
            Item("Continue with ChatGPT", lambda icon, item: self.ui(self._chatgpt_signin)),
            Item("Disconnect ChatGPT", lambda icon, item: self.ui(self._chatgpt_signout),
                 enabled=lambda item: self.chatgpt.connected()),
            Menu.SEPARATOR,
            Item("Language", Menu(
                pick_language("ru", "Russian"),
                pick_language("en", "English"),
                pick_language("auto", "Auto RU / EN"),
            )),
            Item("Recognition engine", Menu(
                pick_engine("chatgpt", "ChatGPT Vision"),
                pick_engine("legacy", "Original Lipflow (English)"),
            )),
            Menu.SEPARATOR,
            Item("Copy last dictation", lambda icon, item: self.ui(self._copy_last)),
            Item("Practice && train more…", lambda icon, item: self.ui(self.show_setup, "practice")),
            Item("Run setup again…", lambda icon, item: self.ui(self.show_setup)),
            Menu.SEPARATOR,
            Item("Camera", Menu(*[pick_camera(v) for v in ["auto", *range(CAMERAS)]])),
            Item("Push-to-talk key", Menu(*[pick_key(k) for k in KEYS])),
            toggle("whisper", False, then=lambda: self.ui(self._whisper_changed)),
            toggle("use_context"),
            toggle("save_clips", False),
            Item("Start with Windows", lambda icon, item: self._toggle_autostart(),
                 checked=lambda item: self._autostart_enabled()),
            Menu.SEPARATOR,
            Item("Open history", lambda icon, item: self._notepad(HISTORY)),
            Item("Edit custom words…", lambda icon, item: self._edit_words()),
            Item("Open log", lambda icon, item: self._notepad(LOG)),
            Menu.SEPARATOR,
            Item("Quit Lipflow", lambda icon, item: self.ui(self.quit)),
        )
        self.icon = pystray.Icon("Lipflow", tray_image(False), "Lipflow", menu)
        threading.Thread(target=self.icon.run, name="lipflow-tray", daemon=True).start()

    def _set_icon(self, listening: bool):
        try:
            self.icon.icon = tray_image(listening)
        except Exception:
            pass

    def _set_state(self, text: str):
        self.state_text = text
        try:
            self.icon.update_menu()
        except Exception:
            pass

    # -- menu actions (tk thread unless noted) ---------------------------------------------------
    def _copy_last(self):
        if self.last_output:
            copy_text(self.last_output)

    def _pick_language(self, value):
        self.settings["language"] = value
        save_settings(self.settings)
        self.icon.update_menu()
        self.hud.show("done", "Language", value.upper(), 1.5)

    def _pick_engine(self, value):
        if value == "legacy" and self.reader is None:
            self.hud.show("error", "Legacy engine is not loaded",
                          "Install with -Legacy and restart Lipflow in legacy mode.", 4.0)
            return
        self.settings["engine"] = value
        save_settings(self.settings)
        self.icon.update_menu()
        label = "ChatGPT Vision" if value == "chatgpt" else "Original Lipflow"
        self.hud.show("done", "Recognition engine", label, 1.5)

    def _chatgpt_signin(self):
        self.hud.show("reading", "ChatGPT", "Opening secure sign-in in your browser…")
        threading.Thread(target=self._chatgpt_signin_worker, name="chatgpt-signin", daemon=True).start()

    def _chatgpt_signin_worker(self):
        try:
            profile = self.chatgpt.sign_in()
            models = self.chatgpt.models()
            if models:
                available = {m["slug"] for m in models}
                current = self.settings.get("chatgpt_model")
                preferred = ("gpt-6.1-sol", "gpt-6-sol", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.5")
                if current not in available or "astra" in (current or "").lower():
                    self.settings["chatgpt_model"] = next((m for m in preferred if m in available), None)
                save_settings(self.settings)
            label = profile.get("email") or profile.get("name") or "Connected"
            self.ui(self.icon.update_menu)
            self.ui(self.hud.show, "done", "ChatGPT connected", label, 3.0)
        except Exception as e:
            print(f"[lipflow] ChatGPT sign-in failed: {e}")
            self.ui(self.hud.show, "error", "ChatGPT sign-in failed", str(e)[:100], 5.0)

    def _chatgpt_signout(self):
        self.chatgpt.sign_out()
        self.settings.pop("chatgpt_model", None)
        save_settings(self.settings)
        self.icon.update_menu()
        self.hud.show("done", "ChatGPT disconnected", "", 2.0)

    def _pick_camera(self, value):
        self.settings["camera"] = value
        save_settings(self.settings)
        self.camera.set_source(value)
        self.icon.update_menu()  # pystray rebuilt it before this queued change ran
        print(f"[lipflow] camera: {value}")

    def _pick_key(self, name):
        self.ptt.stop()
        self.opts.key = name
        self.settings["key"] = name
        save_settings(self.settings)
        self._install_key()
        self.icon.update_menu()
        self.hud.show("done", "Push-to-talk key", f"Tap {self.key_name} to start · tap again to finish", 2.0)

    def _whisper_changed(self):
        if self.settings.get("whisper") and self.av_reader is None and not self.loading:
            self.jobs.put(("whisper",))

    @staticmethod
    def _notepad(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "a", encoding="utf-8").close()
        subprocess.Popen(["notepad.exe", path])

    def _edit_words(self):
        from .. import vocab
        vocab.load()
        self._notepad(vocab.PATH)

    @staticmethod
    def _autostart_command() -> str:
        exe = sys.executable
        if exe.lower().endswith("python.exe"):  # no console window at login
            exe = exe[:-len("python.exe")] + "pythonw.exe"
        return f'"{exe}" -X utf8 -m lipflow'

    def _autostart_enabled(self) -> bool:
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as k:
                winreg.QueryValueEx(k, "Lipflow")
                return True
        except OSError:
            return False

    def _toggle_autostart(self):  # tray thread; registry only
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as k:
            if self._autostart_enabled():
                winreg.DeleteValue(k, "Lipflow")
            else:
                winreg.SetValueEx(k, "Lipflow", 0, winreg.REG_SZ, self._autostart_command())

    def show_setup(self, start_at: str = "welcome"):
        from .setup import Setup
        if self.loading and start_at == "practice":
            self.hud.show("error", "Still loading", "Try again in a moment", 1.5)
            return
        if self.setup is None:
            self.setup = Setup(self)
        self.setup.show(start_at)

    def quit(self):
        self.camera.close()
        self.ptt.stop()
        try:
            self.icon.stop()
        except Exception:
            pass
        self.root.destroy()

    # -- push-to-talk (tk thread) -------------------------------------------------------------
    def on_start(self, hands_free: bool):
        print(f"[lipflow] PTT start (key={self.opts.key}, hands_free={hands_free})")
        if self.loading:
            self.hud.show("error", "Still loading", "The model is almost ready…", hide_after=1.5)
            return
        if self.pending_stop is not None:  # pressed again during the tail: finish the last one now
            self._finish_stop(self.pending_stop)
        if hands_free and self.camera.recording is not None:
            self.hands_free = True  # Windows toggle mode: recording stays active until the next tap
            self.hud.show("listening", "Hands-free · tap to finish", self.hud.body_text)
            return
        self.session += 1
        self.hands_free = hands_free
        from ..context import Context, capture
        # the app you're typing into is in front right now
        self.ctx = capture() if self.settings.get("use_context", False) else Context()
        rec = self.camera.start_recording()
        if self.whisper_on:
            self.mic.start()
        self._set_icon(True)
        title = "Hands-free · tap to finish" if hands_free else "Listening"
        self.hud.show("listening", title, "" if self.camera.ready.is_set() else "Starting camera…")
        threading.Thread(target=self._preview_loop, args=(self.session, rec), daemon=True).start()

    def on_stop(self):
        print(f"[lipflow] PTT stop (key={self.opts.key})")
        self.hands_free = False
        if self.camera.recording is None:
            return
        self.session += 1
        self.pending_stop = self.session
        self._set_icon(False)
        self.hud.show("reading", "Reading your lips", self.hud.body_text)
        token = self.session
        self._finish_stop(token)

    def _finish_stop(self, token):
        if self.pending_stop != token:
            return
        self.pending_stop = None
        rec = self.camera.stop_recording()
        audio = self.mic.stop() if self.whisper_on else []
        if rec is not None:
            rec.audio = audio
            self.jobs.put(("final", rec))

    def on_cancel(self, silent: bool = False):
        self.pending_stop = None
        self.mic.stop()
        self._set_icon(False)
        self.session += 1
        self.hands_free = False
        self.camera.stop_recording()
        if silent:
            self.hud.hide()
        else:
            self.hud.show("error", "Cancelled", "", hide_after=0.8)

    # -- camera thread ------------------------------------------------------------------------
    def on_frame(self, frame, obs, recording):
        """At most one video frame waits for the tk thread at a time, so frames never pile up in
        front of the key handling."""
        rec = self.camera.recording
        if recording and rec is not None and rec.duration > MAX_SECONDS:
            self.ui(self.on_stop)
            return
        if self._ui_busy or (not recording and self.onboarding is None):
            return
        setup = mouth_view(frame, obs, 208, 130) if self.onboarding is not None else None
        pill = mouth_view(frame, obs, 112, 70) if recording else None
        self._ui_busy = True
        self.ui(self._show_frame, setup, pill)

    def _show_frame(self, setup, pill):
        try:
            if setup is not None and self.onboarding is not None:
                self.onboarding.set_frame(setup)
            if pill is not None:
                self.hud.set_frame(pill)
        finally:
            self._ui_busy = False

    # -- model thread -------------------------------------------------------------------------
    def _preview_loop(self, session: int, rec: Recording):
        if not self.opts.live_preview:
            return
        waited = 0.0
        while self.session == session:
            time.sleep(PREVIEW_EVERY)
            waited += PREVIEW_EVERY
            if not rec.ts and (self.camera.error or waited > 6):
                msg = self.camera.error or "The camera isn't sending frames"
                print(f"[lipflow] camera problem: {msg}")
                self.ui(self.hud.show, "error", "Camera problem", msg, 6.0)
                return
            if self.session != session or self.preview_busy or len(rec.ts) < 15:
                continue
            self.preview_busy = True
            self.jobs.put(("preview", session, rec))

    def _worker(self):
        while True:
            job = self.jobs.get()
            try:
                if job[0] == "load":
                    self._load()
                elif job[0] == "preview":
                    self._preview(*job[1:])
                elif job[0] == "final":
                    self._final(job[1])
                elif job[0] == "train":
                    self._train(job[1])
                elif job[0] == "whisper":
                    self._load_whisper()
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.ui(self.hud.show, "error", "Something went wrong", str(e)[:80], 3.0)
            finally:
                if job[0] == "preview":
                    self.preview_busy = False

    def _load(self):
        t = time.time()
        if self.settings.get("engine", "chatgpt") == "legacy":
            from ..cleanup import Cleaner
            from ..vsr import LipReader
            self.reader = LipReader(beam_size=self.opts.beam)
            self.reader.warmup()
            self.cleaner = Cleaner(self.opts.backend)
            print(f"[lipflow] legacy model ready in {time.time() - t:.1f}s "
                  f"(encoder on {self.reader.enc_device}, cleanup: {self.cleaner.describe()})")
        else:
            print(f"[lipflow] ChatGPT mode ready in {time.time() - t:.2f}s; no local VSR loaded")
        self.loading = False
        self.ui(self._set_state, "Ready")
        if self.settings.get("engine", "chatgpt") == "legacy" and self.settings.get("whisper"):
            self.jobs.put(("whisper",))
        if self.settings.get("engine", "chatgpt") == "legacy" and (self.opts.onboard or not self.settings.get("onboarded")):
            self.ui(self.hud.hide)
            self.ui(self.show_setup)
        else:
            body = ("Connect ChatGPT from the tray menu" if not self.chatgpt.connected()
                    else f"Tap {self.key_name} to finish")
            self.ui(self.hud.show, "done", "Lipflow GPT RU is ready", body, 3.0)

    @property
    def whisper_on(self) -> bool:
        return bool(self.settings.get("whisper")) and self.av_reader is not None and self.onboarding is None

    def _load_whisper(self):
        if self.settings.get("engine", "chatgpt") != "legacy":
            self.ui(self.hud.show, "error", "Whisper mode unavailable",
                    "ChatGPT plan sharing supports images, not audio. Use Silent mode.", 4.0)
            return
        from .. import av
        if not av.available():
            self.ui(self.hud.show, "reading", "Whisper mode", "Downloading the audio-visual model (1.8 GB)…")
            try:
                av.download(lambda pct: self.ui(self.hud.set_text, f"Downloading the audio-visual model… {pct:.0f}%"))
            except Exception as e:
                self.ui(self.hud.show, "error", "Whisper mode", f"Download failed: {e}"[:80], 5.0)
                return
        self.ui(self.hud.show, "reading", "Whisper mode", "Loading…")
        self.av_reader = av.AVReader(beam_size=self.opts.beam)
        self.av_reader.warmup_av()
        print("[lipflow] whisper mode ready (lips + audio)")
        self.ui(self.hud.show, "done", "Whisper mode on", "Whisper or speak softly while you mouth the words", 3.0)

    def _av_candidates(self, rec, rois):
        from ..mic import segment
        if not self.whisper_on or not getattr(rec, "audio", None):
            return None
        ts, _, _ = rec.snapshot()
        wave = segment(rec.audio, ts[0], rois.shape[0])
        if wave is None:
            return None
        return self.av_reader.beam_search(self.av_reader.encode_av(rois, wave), nbest=5)

    def _preview(self, session: int, rec: Recording):
        if self.session != session:
            return
        if self.settings.get("engine", "chatgpt") == "chatgpt":
            if rec.face_ratio < 0.4 and len(rec.ts) >= 15:
                self.ui(self.hud.set_text, "Can't see your face…")
            return
        if self.reader is None:
            return
        rois = rois_for(rec)
        if rois is None:
            self.ui(self.hud.set_text, "Can't see your face…")
            return
        text = self.reader.greedy(self.reader.encode(rois))
        if self.session == session and text:
            self.ui(self.hud.set_text, text.lower())

    def _final(self, rec: Recording):
        print(f"[lipflow] final entered ({rec.duration:.2f}s, {len(rec.ts)} frames, face={rec.face_ratio:.0%})")
        t0 = time.time()
        ob = self.onboarding  # the setup window can be closed meanwhile on the tk thread
        problem = clip_problem(rec)
        if problem and ob is not None:
            print(f"[lipflow] practice clip rejected ({rec.duration:.1f}s, {len(rec.ts)} frames, face in "
                  f"{rec.face_ratio:.0%}): {problem[0]}")
            self.ui(ob.clip_done, False, f"{problem[0]}. {problem[1]}.")
            self.ui(self.hud.hide)
            return
        if problem:
            print(f"[lipflow] skipped {rec.duration:.1f}s clip ({len(rec.ts)} frames, face in "
                  f"{rec.face_ratio:.0%}): {problem[0]}")
            self.ui(self.hud.show, "error", problem[0], problem[1], 2.2)
            return
        rois = rois_for(rec)
        if ob is None and self.settings.get("engine", "chatgpt") == "chatgpt":
            # When local clip retention is explicitly enabled, persist the visual input before
            # any network request so a slow/failed ChatGPT call cannot lose the diagnostic clip.
            if self.settings.get("save_clips", False):
                keep_clip(rois, [], "", self.settings)
            if self.settings.get("capture_only", False):
                print(f"[lipflow] diagnostic clip saved ({rec.duration:.2f}s, {len(rec.ts)} frames)")
                self.ui(self.hud.show, "done", "Diagnostic clip saved", f"{rec.duration:.1f}s · {len(rec.ts)} frames", 2.4)
                return
            if not self.chatgpt.connected():
                self.ui(self.hud.show, "error", "Connect ChatGPT first",
                        "Tray menu → Continue with ChatGPT", 4.0)
                return
            self.ui(self.hud.set_text, "Reading lips with ChatGPT…")
            ctx = self.ctx
            use_context = self.settings.get("use_context", False)
            try:
                result = self.gpt_reader.read(
                    rois,
                    language=self.settings.get("language", "ru"),
                    context=" ".join(self.context[-3:]) if use_context else "",
                    names=ctx.names if use_context and ctx else None,
                    model=self.settings.get("chatgpt_model"),
                )
            except ChatGPTInferenceError as e:
                print(f"[lipflow] ChatGPT lip reading failed: {e}")
                self.ui(self.hud.show, "error", "ChatGPT could not read that", str(e)[:100], 5.0)
                return
            text = result.text
            if not text:
                self.ui(self.hud.show, "error", "Couldn't read that", "Try again, a little slower", 2.2)
                return
            t_all = time.time() - t0
            out = (" " if self.last_paste_at and time.time() - self.last_paste_at < JOIN_WINDOW else "") + text
            self.last_output = text
            self.last_paste_at = time.time()
            self.context.append(text)
            log_history(rec, [text], text, t_all, f"chatgpt:{result.model}")
            print(f"[lipflow] {rec.duration:.1f}s clip → ChatGPT {result.model}: {text!r} ({t_all:.2f}s)")
            self.ui(paste_text if self.opts.paste else copy_text, out if self.opts.paste else text)
            self.ui(self.hud.show, "done", "Pasted" if self.opts.paste else "Copied", text, 2.4)
            return

        enc = self.reader.encode(rois)
        t_enc = time.time() - t0
        if ob is not None:  # practice clip: keep it with its known text, don't paste
            raw = self.reader.greedy(enc)
            print(f"[lipflow] practice clip saved ({rec.duration:.1f}s): {raw!r}")
            self.ui(ob.clip_done, True, "", rois, self.onboarding_text, raw)
            self.ui(self.hud.hide)
            return
        # Whisper mode reads empty when there's no audible whisper (silent mouthing): fall back to lips
        candidates = self._av_candidates(rec, rois)
        if not candidates or not candidates[0]:
            if candidates is not None:
                print("[lipflow] lips + audio read nothing, using lips only")
            candidates = self.reader.beam_search(enc, nbest=5)
        t_beam = time.time() - t0 - t_enc
        if not candidates or not candidates[0]:
            print(f"[lipflow] {rec.duration:.1f}s clip: nothing read")
            self.ui(self.hud.show, "error", "Couldn't read that", "Try again, a little slower", 2.2)
            return
        self.ui(self.hud.set_text, candidates[0].lower())
        ctx = self.ctx
        text = self.cleaner(candidates, context=" ".join(self.context[-3:]), names=ctx.names if ctx else None)
        t_all = time.time() - t0
        print(f"[lipflow] {rec.duration:.1f}s clip → raw: {candidates[0]!r}\n"
              f"          → typed: {text!r}  (encode {t_enc:.2f}s, beam {t_beam:.2f}s, total {t_all:.2f}s)")
        if not text:
            self.ui(self.hud.show, "error", "Couldn't read that", "Try again, a little slower", 2.2)
            return
        out = text
        if self.last_paste_at and time.time() - self.last_paste_at < JOIN_WINDOW:
            out = " " + text
        self.last_output = text
        self.last_paste_at = time.time()
        self.context.append(text)
        log_history(rec, candidates, text, t_all, self.cleaner.describe())
        keep_clip(rois, candidates, text, self.settings)
        self.ui(paste_text if self.opts.paste else copy_text, out if self.opts.paste else text)
        self.ui(self.hud.show, "done", "Pasted" if self.opts.paste else "Copied", text, 2.4)

    def _train(self, ob):
        self.loading = True
        self.ui(self._set_state, "Training on your face…")
        r = train_on_face(self.opts.beam, ob.report)
        if r["after"] is not None:
            self.reader = LipReader(beam_size=self.opts.beam)
            self.reader.warmup()
            self.settings["training"] = {"before": r["before"], "after": r["after"], "kept": r["kept"],
                                         "clips": r["clips"], "at": time.time()}
            save_settings(self.settings)
        self.loading = False
        self.ui(self._set_state, "Ready")
        ob.finished(r["before"], r["after"], r["kept"], r["note"])


def _log_to_file():
    """pythonw has no console: send prints and tracebacks to %APPDATA%\\Lipflow\\Lipflow.log."""
    os.makedirs(HOME, exist_ok=True)
    try:
        if os.path.getsize(LOG) > 5_000_000:
            os.replace(LOG, LOG + ".old")
    except OSError:
        pass
    f = open(LOG, "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = f
    print(f"\n[lipflow] started {time.strftime('%Y-%m-%d %H:%M:%S')}")


def _already_running() -> bool:
    import ctypes
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.restype = ctypes.c_void_p
    _already_running.handle = kernel32.CreateMutexW(None, False, "Local\\LipflowGPTTray")  # held until exit
    return ctypes.get_last_error() == 183  # ERROR_ALREADY_EXISTS


def run(opts: Options):
    if sys.stdout is None or os.environ.get("LIPFLOW_APP"):
        _log_to_file()
    if _already_running():
        import ctypes
        ctypes.WinDLL("user32").MessageBoxW(None, "Lipflow is already running. Look for the pink mouth icon "
                                                  "in the system tray (you may need to click ^ to see it).",
                                            "Lipflow", 0x40)
        return
    lf = Lipflow(opts)
    lf.start()
    import signal
    signal.signal(signal.SIGINT, lambda *a: lf.ui(lf.quit))  # tk would swallow Ctrl-C in its callbacks
    print(f"[lipflow] tap {lf.key_name} to start · tap again to finish · Esc cancels · Ctrl-C quits")
    lf.root.mainloop()
    os._exit(0)  # the hook, tray and model threads don't need a clean shutdown
