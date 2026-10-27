"""
JARVIS v3 - commands only from YOUR voice, friendly conversation with anyone. Works offline: no API key needed.

SETUP   python -m pip install flask numpy scipy resemblyzer faster-whisper pyautogui psutil pillow rapidocr-onnxruntime
RUN     python jarvis.py   (opens http://localhost:5000 - use Chrome/Edge, allow the microphone)

WHO CAN DO WHAT
    You (voiceprint match)  everything: PC control, teaching, your private facts and documents, chat
    Anyone else (guest)     chat and general questions only. Must say "Jarvis" first. No commands, no teaching,
                            and no access to things you told it to remember, or to documents/screens you taught it.
                            (Answers you teach with "when someone asks ... say ..." ARE shared with guests.)

SEE THE TERMINAL / SCREEN (offline OCR, no API key)
    what's on the terminal  /  read the terminal  /  is there an error  /  explain the error
    what's on my screen  /  read the window called chrome  /  learn from the terminal

TALK AND TEACH
    hello / how are you / tell me a joke / what is photosynthesis   (unknown facts: searches DuckDuckGo, Wikipedia, Stack Overflow and the top web pages, saved for offline)
    the answer to <question> is <answer>   /   when someone asks <question> say <answer>
    (right after "I don't know that yet")   the answer is <answer>
    learn that work mode means open chrome and open code   /   when I say focus do mute and open notepad
    remember my name is Deepa   /   what is my name   /   forget work mode   /   what have you learned
    research <anything>  /  how do I fix this error   (searches the web, no key)
    learn from folder C:/notes   /   learn from https://...   /   guest mode off   /   guest mode on

OPTIONAL ENV VARS
    VOICE_THRESHOLD 0.75   WHISPER_MODEL base.en (small.en = more accurate)   REQUIRE_WAKE_WORD 0|1
    ADAPT_VOICE 1 (voiceprint improves as you use it)   GUEST_MODE 1   GUEST_WAKE 1   CONVO_SECS 20
    WEB_LEARN 1 (set 0 to never search the web)   OLLAMA_MODEL llama3.2 (free local AI, optional)
    ANTHROPIC_API_KEY sk-...  (optional extra brain, NOT needed; CLAUDE_MODEL default claude-haiku-5-5)
SAFETY  shutdown/restart/shell/force-kill always ask "yes" first. Mouse to a screen corner aborts pyautogui.
Voice matching is a convenience lock, not strong security.
"""
import ast, ctypes, datetime, difflib, glob, gzip, html, inspect, json, math, operator, os, random, re, secrets
import subprocess, threading, time, urllib.request, webbrowser
from pathlib import Path
from urllib.parse import quote_plus, unquote, urlparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from flask import Flask, Response, abort, jsonify, request
try:
    import pyautogui
    pyautogui.PAUSE = 0.05
except Exception:
    pyautogui = None
try:
    import psutil
except Exception:
    psutil = None

THRESHOLD = float(os.getenv("VOICE_THRESHOLD", "0.75"))
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base.en")
REQUIRE_WAKE_WORD = os.getenv("REQUIRE_WAKE_WORD", "0") == "1"
ADAPT = os.getenv("ADAPT_VOICE", "1") == "1"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-5-5")
GUEST_MODE = os.getenv("GUEST_MODE", "1") == "1"      # other voices may chat (they never get commands)
GUEST_WAKE = os.getenv("GUEST_WAKE", "1") == "1"      # ...but must say "Jarvis" first
WEB_LEARN = os.getenv("WEB_LEARN", "1") == "1"        # unknown facts: free Wikipedia lookup, saved for offline use
CONVO_SECS = float(os.getenv("CONVO_SECS", "20"))     # follow-ups need no wake word for this long
SAMPLES_NEEDED = 5
PORT = 5000
ALLOWED_HOSTS = {f"localhost:{PORT}", f"127.0.0.1:{PORT}"}
HERE = Path(__file__).resolve().parent
VOICEPRINT = str(HERE / "voiceprint.npy")
MEM_F, LOG_F = HERE / "jarvis_memory.json", HERE / "jarvis_log.txt"
TOKEN = secrets.token_urlsafe(16)

app = Flask(__name__)
_encoder = _whisper = None
LOCK = threading.RLock()
EVENTS, HIST = [], []
PENDING = LAST = ASKED = None
AWAKE_UNTIL = GUEST_UNTIL = 0.0
SCREEN = {"text": "", "title": "", "t": 0.0}      # last thing I read off the terminal/screen

def encoder():
    global _encoder
    if _encoder is None:
        from resemblyzer import VoiceEncoder
        _encoder = VoiceEncoder()
    return _encoder


def whisper():
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel
        _whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    return _whisper


def pcm_from_request():
    """Raw little-endian int16 mono 16 kHz -> float32 [-1, 1]."""
    data = request.get_data()
    data = data[: len(data) // 2 * 2]                       # drop a stray odd byte
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def embed(wav):
    from resemblyzer import preprocess_wav
    wav = preprocess_wav(wav, source_sr=16000)
    if len(wav) < 1600:                                      # silence trimmed away
        raise ValueError("no speech detected")
    with LOCK:
        return encoder().embed_utterance(wav)


def load_prints():
    if not os.path.exists(VOICEPRINT):
        return np.zeros((0, 256))
    try:
        p = np.load(VOICEPRINT)
        return p if p.ndim == 2 and p.shape[1] == 256 else np.zeros((0, 256))
    except Exception:
        return np.zeros((0, 256))


def centroid(prints):
    c = prints.mean(axis=0)
    return c / np.linalg.norm(c)


# ---------------------------------------------------------------- hearing (cleanup, speaker match, transcription, wake word)
def clean_audio(wav):
    """Remove DC offset and rumble, then even out loudness (gain is capped so room noise isn't blown up)."""
    if len(wav) == 0:
        return wav
    wav = wav - wav.mean()
    try:
        from scipy.signal import butter, sosfilt
        wav = sosfilt(butter(2, 80, "highpass", fs=16000, output="sos"), wav).astype(np.float32)
    except Exception:
        pass
    rms, peak = float(np.sqrt(np.mean(wav ** 2))) + 1e-9, float(np.abs(wav).max()) + 1e-9
    return (wav * min(10.0, 0.08 / rms, 0.95 / peak)).astype(np.float32)


def voice_score(vec, prints):
    """How much this sounds like you: blend of the average voiceprint and the 3 closest enrolled samples."""
    sims = np.sort(prints @ vec)
    return float(0.6 * np.dot(vec, centroid(prints)) + 0.4 * sims[-3:].mean())


HALLU = re.compile(r"(?i)\W*(?:thanks? for watching|thank you for watching|subtitles? by.*|please subscribe|you|\.+)\W*")


def transcribe(wav):
    """Whisper with the usual hallucination guards (silence/noise often comes back as 'Thank you.')."""
    segs, _ = whisper().transcribe(wav, language="en", beam_size=3, vad_filter=True,
                                   vad_parameters={"min_silence_duration_ms": 400}, temperature=0.0,
                                   condition_on_previous_text=False, no_speech_threshold=0.6,
                                   initial_prompt=vocab())
    out = []
    for s in segs:
        if s.no_speech_prob > 0.6 and s.avg_logprob < -0.8:
            continue
        if s.avg_logprob < -1.6 or s.compression_ratio > 2.4:
            continue
        out.append(s.text.strip())
    text = " ".join(out).strip()
    return "" if HALLU.fullmatch(text) else text


FILLER = {"hey", "ok", "okay", "hi", "hello", "yo"}


def _is_wake(w):
    return difflib.SequenceMatcher(None, re.sub(r"[^a-z]", "", w.lower()), "jarvis").ratio() >= 0.78


def heard_wake(text):
    """True if 'Jarvis' was said (tolerates mis-hearings like 'jarvas' or 'jervis')."""
    return any(_is_wake(w) for w in re.findall(r"[a-z']+", text.lower())[:12])


def strip_wake(t):
    """'hey jarvis, open chrome' -> 'open chrome'  (also 'open chrome jarvis')."""
    parts = t.split()
    for i in range(min(3, len(parts))):
        if _is_wake(parts[i]) and all(re.sub(r"[^a-z]", "", p.lower()) in FILLER for p in parts[:i]):
            return " ".join(parts[i + 1:])
    if len(parts) > 1 and _is_wake(parts[-1]):
        return " ".join(parts[:-1])
    return t


@app.before_request
def guard():
    """Block other websites / DNS rebinding; every POST needs this session's token."""
    if request.host not in ALLOWED_HOSTS:
        abort(403)
    origin = request.headers.get("Origin")
    if origin and urlparse(origin).netloc not in ALLOWED_HOSTS:
        abort(403)
    if request.method == "POST" and request.headers.get("X-Token") != TOKEN:
        abort(403)


# ---------------------------------------------------------------- memory (learning)
def _load():
    try:
        m = json.loads(MEM_F.read_text("utf-8"))
    except Exception:
        m = {}
    for k in ("facts", "macros", "usage", "learned", "qa"):
        m.setdefault(k, {})
    m.setdefault("unknown", [])
    return m


MEM = _load()
def save(): MEM_F.write_text(json.dumps(MEM, indent=1), "utf-8")
def log(s): LOG_F.open("a", encoding="utf-8").write(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} {s}\n")


# ---------------------------------------------------------------- speech (Windows built-in voice)
TTS_OK = os.name == "nt"
_SPK, _CNT, _spk_n = threading.Lock(), threading.Lock(), 0


def speak_async(text):
    global _spk_n
    if not (TTS_OK and text):
        return
    with _CNT:
        _spk_n += 1

    def run():
        global TTS_OK, _spk_n
        try:
            with _SPK:
                r = subprocess.run(
                    ["powershell", "-NoProfile", "-Command",
                     "Add-Type -AssemblyName System.Speech;$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
                     "try{$s.SelectVoiceByHints('Male')}catch{};$s.Speak($env:JARVIS_SAY)"],
                    env=dict(os.environ, JARVIS_SAY=text[:600]), capture_output=True, timeout=90,
                    creationflags=0x08000000)
                if r.returncode != 0:
                    TTS_OK = False                      # browser voice takes over
                    log("tts failed: " + r.stderr.decode(errors="ignore")[:200])
        except Exception as e:
            TTS_OK = False
            log(f"tts error {e}")
        finally:
            with _CNT:
                _spk_n -= 1
    threading.Thread(target=run, daemon=True).start()


def notify(msg):
    EVENTS.append(msg)
    speak_async(msg)


# ---------------------------------------------------------------- knowledge (learn from docs, open source, web pages)
KNOW_F = HERE / "jarvis_knowledge.json"


def _kload():
    try:
        return json.loads(KNOW_F.read_text("utf-8"))
    except Exception:
        return []


KNOW = _kload()
_TOKS = []
PUBLIC_SRC = ("http", "wikipedia:", "stackoverflow:", "duckduckgo:")      # what guests may be told; your folders and screens stay private


def _store(src, text, flush=True):
    global KNOW
    t = re.sub(r"\s+", " ", text)[:200000]
    KNOW = [k for k in KNOW if k["src"] != src] + [{"src": src, "text": t[i:i + 800]} for i in range(0, len(t), 700)]
    _TOKS.clear()
    if flush:
        KNOW_F.write_text(json.dumps(KNOW), "utf-8")


STOP = set("what whats who whos how why when where which does did are was were the and for you your about tell explain "
           "define mean meaning please can could would will this that there their them with from have has not any give "
           "show say said really very just like know".split())


def _kw(s):
    return [w for w in re.findall(r"[a-z0-9]{3,}", s.lower()) if w not in STOP]


def _index():
    global _TOKS
    if len(_TOKS) != len(KNOW):
        _TOKS = [set(re.findall(r"[a-z0-9]{3,}", k["text"].lower())) for k in KNOW]
    return _TOKS


def recall(q, n=3, owner=True):
    """Best-matching learned passages (rare words count more). Guests only see public sources."""
    kws = set(_kw(q))
    toks = _index()
    ids = [i for i, k in enumerate(KNOW) if owner or k["src"].startswith(PUBLIC_SRC)]
    if not kws or not ids:
        return []
    df = {w: sum(w in toks[i] for i in ids) for w in kws}
    wt = {w: math.log(1 + len(ids) / df[w]) for w in kws if df[w]}
    need = min(2, len(kws))
    ranked = sorted(((sum(v for w, v in wt.items() if w in toks[i]), i) for i in ids), reverse=True)[:n]
    return [KNOW[i]["text"] for sc, i in ranked if sc and sum(w in toks[i] for w in wt) >= need]


def best_sentences(q, chunks, k=2):
    """Pick the 1-2 sentences that actually answer the question."""
    kws, num = set(_kw(q)), q.lower().startswith("how ")
    sents = [s.strip() for c in chunks for s in re.split(r"(?<=[.!?])\s+", c) if len(s.strip()) > 25]

    def sc(i):
        s = sents[i].lower()
        return sum(w in s for w in kws) + (0.5 if num and re.search(r"\d", s) else 0)
    rank = sorted(range(len(sents)), key=lambda i: -sc(i))[:k]
    return " ".join(sents[i] for i in sorted(rank) if sc(i) > 0)


# ---------------------------------------------------------------- PC actions
BUILTIN = {"notepad": "notepad", "calculator": "calc", "paint": "mspaint", "cmd": "cmd",
           "command prompt": "cmd", "powershell": "powershell", "task manager": "taskmgr",
           "settings": "ms-settings:", "file explorer": "explorer", "explorer": "explorer",
           "control panel": "control", "snipping tool": "ms-screenclip:", "chrome": "chrome",
           "edge": "msedge", "firefox": "firefox", "word": "winword", "excel": "excel",
           "powerpoint": "powerpnt", "spotify": "spotify", "vs code": "code", "vscode": "code",
           "code": "code"}
PROC = {"calculator": "CalculatorApp.exe", "word": "WINWORD.EXE", "excel": "EXCEL.EXE",
        "powerpoint": "POWERPNT.EXE", "vs code": "Code.exe", "vscode": "Code.exe", "code": "Code.exe",
        "edge": "msedge.exe", "paint": "mspaint.exe", "task manager": "Taskmgr.exe",
        "file explorer": "explorer.exe", "command prompt": "cmd.exe"}
SITES = {"youtube": "youtube.com", "google": "google.com", "gmail": "mail.google.com",
         "github": "github.com", "whatsapp": "web.whatsapp.com", "instagram": "instagram.com",
         "facebook": "facebook.com", "twitter": "x.com", "reddit": "reddit.com", "netflix": "netflix.com",
         "amazon": "amazon.in", "maps": "maps.google.com", "drive": "drive.google.com",
         "chatgpt": "chatgpt.com", "claude": "claude.ai", "linkedin": "linkedin.com"}
SENSITIVE = {"shutdown", "restart", "shell", "kill_app"}
ACT = {}
_START = None


def act(f):
    ACT[f.__name__] = f
    return f


def vk(code, n=1):
    for _ in range(n):
        ctypes.windll.user32.keybd_event(code, 0, 0, 0)
        ctypes.windll.user32.keybd_event(code, 0, 2, 0)


def start(target):
    subprocess.Popen(["cmd", "/c", "start", "", str(target)])


def P():
    if pyautogui is None:
        raise RuntimeError("run: python -m pip install pyautogui")
    return pyautogui


def start_apps():
    global _START
    if _START is None:
        _START = {}
        for root in (r"%ProgramData%\Microsoft\Windows\Start Menu\Programs",
                     r"%AppData%\Microsoft\Windows\Start Menu\Programs"):
            for p in glob.glob(os.path.expandvars(root) + "/**/*.lnk", recursive=True):
                _START[Path(p).stem.lower()] = p
    return _START


def launch(n):
    t = BUILTIN.get(n)
    if not t:
        a = start_apps()
        k = sorted([x for x in a if n in x], key=len) or difflib.get_close_matches(n, list(a), 1, 0.7)
        t = a[k[0]] if k else None
    if t:
        start(t)
    return bool(t)


@act
def open_site(url):
    u = url if re.match(r"https?://", url) else "https://" + url
    if not re.fullmatch(r"https?://[\w.\-]+(:\d+)?(/\S*)?", u):
        return "I can't open that."
    webbrowser.open(u)
    return f"Opening {urlparse(u).netloc}."


@act
def open_app(name):
    n = name.lower().strip()
    if n in SITES:
        return open_site(SITES[n])
    if launch(n):
        return f"Opening {n}."
    if re.fullmatch(r"[a-z0-9\-]+(\.[a-z0-9\-]+)+", n.replace(" ", "")):
        return open_site(n.replace(" ", ""))
    return f"I couldn't find {n}. Say: learn that {n} means open something."


def _kill(name, force):
    n = name.lower().strip()
    exe = PROC.get(n, n.replace(" ", "") + ".exe")
    if not re.fullmatch(r"[\w.\-]+", exe):
        return "I can't close that."
    r = subprocess.run(["taskkill"] + (["/f"] if force else []) + ["/im", exe], capture_output=True)
    return f"Closed {n}." if r.returncode == 0 else f"{n} isn't running."


@act
def close_app(name): return _kill(name, False)       # graceful: apps still ask to save


@act
def kill_app(name): return _kill(name, True)         # force: needs confirmation


@act
def search(query):
    webbrowser.open("https://www.google.com/search?q=" + quote_plus(query))
    return f"Searching for {query}."


@act
def youtube(query):
    webbrowser.open("https://www.youtube.com/results?search_query=" + quote_plus(query))
    return f"Looking up {query} on YouTube."


@act
def type_text(text):
    P().write(text, interval=0.02)
    return "Typed."


@act
def press(keys):
    P().hotkey(*[k for k in re.split(r"[+\s]+", keys.lower()) if k])
    return "Done."


@act
def click(kind="left"):
    g = P()
    {"left": g.click, "double": g.doubleClick, "right": g.rightClick}.get(kind, g.click)()
    return "Clicked."


@act
def scroll(dir="down"):
    P().scroll(600 if dir == "up" else -600)
    return "Scrolling."


@act
def move_mouse(dir):
    dx, dy = {"left": (-200, 0), "right": (200, 0), "up": (0, -200), "down": (0, 200)}[dir]
    P().moveRel(dx, dy, duration=0.2)
    return "Moved."


@act
def volume(dir):
    vk({"up": 0xAF, "down": 0xAE, "mute": 0xAD}[dir], 5 if dir != "mute" else 1)
    return "Done."


@act
def set_volume(level):
    level = max(0, min(100, int(level)))
    vk(0xAE, 50)
    vk(0xAF, round(level / 2))
    return f"Volume {level} percent."


@act
def media(cmd):
    vk({"play": 0xB3, "next": 0xB0, "prev": 0xB1}[cmd])
    return "Done."


@act
def screenshot():
    d = Path.home() / "Pictures" / "Jarvis"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"shot_{datetime.datetime.now():%Y%m%d_%H%M%S}.png"
    P().screenshot(str(f))
    return f"Screenshot saved to {f.name} in Pictures slash Jarvis."


@act
def lock():
    ctypes.windll.user32.LockWorkStation()
    return "Locking."


@act
def sleep_pc():
    subprocess.Popen(["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"])
    return "Going to sleep."


@act
def shutdown():
    subprocess.Popen(["shutdown", "/s", "/t", "10"])
    return "Shutting down in 10 seconds. Say cancel shutdown to stop."


@act
def restart():
    subprocess.Popen(["shutdown", "/r", "/t", "10"])
    return "Restarting in 10 seconds. Say cancel shutdown to stop."


@act
def cancel_shutdown():
    subprocess.run(["shutdown", "/a"], capture_output=True)
    return "Cancelled."


@act
def shell(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20)
    return ((r.stdout or r.stderr).strip()[:200]) or "Done."


@act
def folder(name):
    start(Path.home() / name.capitalize())
    return f"Opening {name}."


@act
def note(text):
    p = Path.home() / "Documents" / "Jarvis_notes.txt"
    p.open("a", encoding="utf-8").write(f"[{datetime.datetime.now():%Y-%m-%d %H:%M}] {text}\n")
    return "Noted."


@act
def read_notes():
    p = Path.home() / "Documents" / "Jarvis_notes.txt"
    return " ".join(p.read_text("utf-8").splitlines()[-3:]) if p.exists() else "You have no notes."


@act
def timer(seconds, label="your timer"):
    t = threading.Timer(float(seconds), lambda: notify(f"Time is up: {label}"))
    t.daemon = True
    t.start()
    return f"Okay, I'll tell you in {int(float(seconds))} seconds."


@act
def sysinfo():
    if not psutil:
        return "Install psutil for system info."
    b = psutil.sensors_battery()
    return (f"CPU {psutil.cpu_percent(0.3):.0f} percent, memory {psutil.virtual_memory().percent:.0f} percent"
            + (f", battery {b.percent:.0f} percent{', charging' if b.power_plugged else ''}." if b else "."))


@act
def clock(what="time"):
    n = datetime.datetime.now()
    return ("It's " + n.strftime("%I:%M %p").lstrip("0") + ".") if what == "time" else \
        n.strftime("Today is %A, %B ") + str(n.day) + "."


@act
def calc(expr):
    v = safe_eval(expr)
    return f"That's {int(v) if isinstance(v, float) and v.is_integer() else round(v, 6)}."


@act
def say(text): return text


@act
def look(question="Describe what is on my screen."):
    """Sees the screen. Offline OCR by default; if ANTHROPIC_API_KEY is set, Claude describes the picture instead."""
    if not ANTHROPIC_KEY:
        title, text = read_screen("window")
        lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 3]
        head = "On the screen: " if title == "the screen" else f"The front window is {title[:50]}. "
        if diagnose(text):
            return head + diagnose(text)
        return head + ("I can read: " + " ... ".join(lines[:4])[:240] if lines else "I couldn't read any text.")
    import base64, io
    img = P().screenshot()
    img.thumbnail((1400, 1400))
    b = io.BytesIO()
    img.save(b, "PNG")
    r = _post("https://api.anthropic.com/v1/messages", {"x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01"},
              {"model": CLAUDE_MODEL, "max_tokens": 400, "messages": [{"role": "user", "content": [
                  {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                               "data": base64.b64encode(b.getvalue()).decode()}},
                  {"type": "text", "text": question + " Answer in two short sentences."}]}]})
    return r["content"][0]["text"]


@act
def focus_window(name):
    n = re.sub(r"[^\w .\-]", "", name)
    r = subprocess.run(["powershell", "-NoProfile", "-Command",
                        f"(New-Object -ComObject WScript.Shell).AppActivate('{n}')"], capture_output=True, text=True, timeout=10)
    return f"Switched to {n}." if "True" in r.stdout else f"I couldn't find a window called {n}."


@act
def learn_docs(path):
    p = Path(path.strip(' "'))
    files = [p] if p.is_file() else [f for f in p.rglob("*") if f.suffix.lower() in (".txt", ".md", ".py", ".json", ".rst", ".csv")][:300]
    n = 0
    for f in files:
        try:
            _store(str(f), f.read_text("utf-8", errors="ignore"), flush=False)
            n += 1
        except Exception:
            pass
    KNOW_F.write_text(json.dumps(KNOW), "utf-8")
    return f"I learned {n} files from {p.name}."


@act
def learn_url(url):
    import ipaddress, socket
    u = urlparse(url)
    if u.scheme != "https" or not u.hostname:
        return "I only read https links."
    try:
        for a in socket.getaddrinfo(u.hostname, None):
            ip = ipaddress.ip_address(a[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local:
                return "I won't read local network addresses."
        raw = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Jarvis"}), timeout=20).read(2_000_000)
    except Exception as e:
        return f"I couldn't read that page: {str(e)[:60]}"
    t = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw.decode("utf-8", "ignore"))
    _store(url, re.sub(r"<[^>]+>", " ", t))
    return f"I learned that page from {u.hostname}."


@act
def export_training():
    """Writes jarvis_training.jsonl (for fine-tuning) and Modelfile (for ollama create)."""
    ex = [(k, [tuple(x) for x in v]) for k, v in MEM["learned"].items()]
    for k in MEM["macros"]:
        st = plan(k)
        if st:
            ex.append((k, st))
    if not ex:
        return "I have nothing to export yet. Use me for a while first."
    sysm, base = system_prompt(), (OLLAMA_MODEL or "llama3.2")
    js = lambda st: json.dumps({"steps": [{"action": a, "args": g} for a, g in st]})
    with (HERE / "jarvis_training.jsonl").open("w", encoding="utf-8") as f:
        for k, st in ex:
            f.write(json.dumps({"messages": [{"role": "system", "content": sysm}, {"role": "user", "content": k},
                                             {"role": "assistant", "content": js(st)}]}) + "\n")
    q3 = chr(34) * 3
    mf = [f"FROM {base}", "SYSTEM " + q3 + sysm.replace(q3, "") + q3, "PARAMETER temperature 0.2"]
    for k, st in ex[-40:]:
        mf += [f"MESSAGE user {k}", f"MESSAGE assistant {js(st)}"]
    (HERE / "Modelfile").write_text("\n".join(mf) + "\n", "utf-8")
    return f"Exported {len(ex)} examples to the Jarvis folder."


@act
def wait(seconds):
    time.sleep(min(float(seconds), 10))
    return ""


@act
def brightness(level):
    level = max(0, min(100, int(level)))
    subprocess.run(["powershell", "-NoProfile", "-Command",
                    f"(Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods).WmiSetBrightness(1,{level})"],
                   capture_output=True, timeout=15)
    return f"Brightness {level} percent."


@act
def find_file(name):
    n = name.lower()
    for d in ("Desktop", "Documents", "Downloads", "Pictures", "Music", "Videos"):
        for root, _, files in os.walk(Path.home() / d):
            for f in files:
                if n in f.lower():
                    start(Path(root) / f)
                    return f"Opening {f}."
    return f"I couldn't find a file named {name}."


# ---------------------------------------------------------------- eyes (offline OCR: reads the terminal / screen, no API key)
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)      # window rectangles == real screen pixels
except Exception:
    pass


class Friendly(RuntimeError):
    """An error whose message is fit to be spoken as-is."""


class RECT(ctypes.Structure):
    _fields_ = [("l", ctypes.c_long), ("t", ctypes.c_long), ("r", ctypes.c_long), ("b", ctypes.c_long)]


TERM_CLASSES = {"CASCADIA_HOSTING_WINDOW_CLASS", "ConsoleWindowClass"}
TERM_WORDS = ("terminal", "powershell", "command prompt", "cmd.exe", "bash", "wsl", "ubuntu", "python.exe", "anaconda", "mingw")
BROWSERS = ("chrome", "edge", "firefox", "brave", "opera")
GENERIC = ("", "terminal", "console", "command prompt", "cmd", "powershell")
SHELL_CLASSES = {"Shell_TrayWnd", "Shell_SecondaryTrayWnd", "Progman", "WorkerW"}
_ocr = None


def _u32():
    u = ctypes.windll.user32
    u.GetForegroundWindow.restype = ctypes.c_void_p
    for f in (u.IsWindowVisible, u.IsIconic, u.GetWindowTextLengthW, u.SetForegroundWindow):
        f.argtypes = [ctypes.c_void_p]
    u.ShowWindow.argtypes = [ctypes.c_void_p, ctypes.c_int]
    u.GetWindowTextW.argtypes = u.GetClassNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_int]
    u.GetWindowRect.argtypes = [ctypes.c_void_p, ctypes.POINTER(RECT)]
    return u


def _cloaked(h):
    v = ctypes.c_int(0)
    try:
        ctypes.windll.dwmapi.DwmGetWindowAttribute(ctypes.c_void_p(h), 14, ctypes.byref(v), 4)
    except Exception:
        pass
    return v.value != 0


def _rect(h):
    r = RECT()
    return (r.l, r.t, r.r, r.b) if _u32().GetWindowRect(h, ctypes.byref(r)) else None


def windows():
    """[(hwnd, title, class)] of visible top-level windows, front-most first."""
    u, out = _u32(), []

    def cb(h, _):
        if h and u.IsWindowVisible(h) and u.GetWindowTextLengthW(h) and not _cloaked(h):
            t, c = ctypes.create_unicode_buffer(300), ctypes.create_unicode_buffer(100)
            u.GetWindowTextW(h, t, 300)
            u.GetClassNameW(h, c, 100)
            out.append((h, t.value, c.value))
        return True
    u.EnumWindows(ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)(cb), ctypes.c_void_p(0))
    return out


def find_terminal(name=""):
    """A terminal window (Windows Terminal, cmd, PowerShell, ...) or any window whose title contains `name`."""
    n = name.lower().strip()
    for h, t, c in windows():
        tl = t.lower()
        if n not in GENERIC:
            if n in tl:
                return h, t
        elif c in TERM_CLASSES or (any(w in tl for w in TERM_WORDS) and not any(b in tl for b in BROWSERS)):
            return h, t
    return None


def grab_window(h):
    """Bring a window to the front, photograph it, then hand focus back to where it was."""
    from PIL import ImageGrab
    u = _u32()
    back = u.GetForegroundWindow()
    if u.IsIconic(h):
        u.ShowWindow(h, 9)
    vk(0x12)                                    # tap Alt: lets us steal focus from the browser tab
    u.SetForegroundWindow(h)
    time.sleep(0.6)
    r = _rect(h)
    img = ImageGrab.grab(bbox=r, all_screens=True) if r else ImageGrab.grab()
    if back and back != h:
        vk(0x12)
        u.SetForegroundWindow(back)
    return img


def _group_lines(res):
    """OCR boxes -> text lines (top to bottom, left to right)."""
    rows = sorted((sum(p[1] for p in b) / 4, min(p[0] for p in b), max(p[1] for p in b) - min(p[1] for p in b), tx)
                  for b, tx, _s in res)
    if not rows:
        return ""
    hs = sorted(r[2] for r in rows)[len(rows) // 2] or 10
    lines, cur = [], []
    for r in rows:
        if cur and r[0] - cur[0][0] > hs * 0.6:
            lines.append(cur)
            cur = []
        cur.append(r)
    lines.append(cur)
    return "\n".join(" ".join(x[3] for x in sorted(l, key=lambda x: x[1])) for l in lines)


def ocr(img):
    """Offline text recognition: rapidocr-onnxruntime (pip only) or Tesseract if you have it."""
    global _ocr
    w, h = img.size
    if w < 1100:
        img = img.resize((w * 2, h * 2))        # small console text reads better enlarged
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        RapidOCR = None
    if RapidOCR:
        if _ocr is None:
            _ocr = RapidOCR()
        res, _ = _ocr(np.ascontiguousarray(np.array(img.convert("RGB"))[:, :, ::-1]))
        return _group_lines(res or [])
    try:
        import pytesseract
        return pytesseract.image_to_string(img, config="--psm 6")
    except Exception:
        raise Friendly("I need an offline reader. Run: python -m pip install rapidocr-onnxruntime")


def read_screen(target="window", name=""):
    """-> (title, text). target: terminal | window (front-most app that isn't Jarvis) | screen. Remembered in SCREEN."""
    from PIL import ImageGrab
    img = title = None
    if target == "terminal":
        f = find_terminal(name)
        if not f:
            raise Friendly("I can't find a terminal window. Open one, or say: read the window called <name>.")
        title, img = f[1], grab_window(f[0])
    elif target == "window":
        for h, t, c in windows():
            if c in SHELL_CLASSES or "jarvis" in t.lower():
                continue
            r = _rect(h)
            if r and r[2] - r[0] > 200 and r[3] - r[1] > 120:
                title, img = t, grab_window(h)
                break
    if img is None:
        title, img = "the screen", ImageGrab.grab()
    text = ocr(img)
    SCREEN.update(text=text, title=title, t=time.time())
    return title, text


def _fresh_terminal():
    if not SCREEN["text"] or time.time() - SCREEN["t"] > 20:
        try:
            read_screen("terminal")
        except Friendly:
            read_screen("window")
    return SCREEN["text"]


ERRORS = [
    (r"no module named '?([\w.\-]+)", lambda m: f"Python can't find the module {m[1].split('.')[0]}. Try: pip install {m[1].split('.')[0]}."),
    (r"cannot import name '?(\w+)", lambda m: f"Python can't import {m[1]}. Check the spelling or update that package."),
    (r"name '?(\w+)'? is not defined", lambda m: f"The name {m[1]} is used before it is defined. Check the spelling or an import."),
    (r"syntaxerror|invalid syntax", lambda m: "Syntax error. Look just above the caret for a missing bracket, quote or colon."),
    (r"indentationerror|unexpected indent", lambda m: "Indentation error. Check tabs versus spaces and alignment."),
    (r"filenotfounderror|no such file or directory|cannot find the (?:path|file)", lambda m: "A file or folder wasn't found. Check the path and the current directory."),
    (r"permissionerror|permission denied|access is denied", lambda m: "Permission denied. Close other programs using it, or run as administrator."),
    (r"is not recognized as an internal or external command|command not found", lambda m: "That command isn't installed or isn't on the PATH."),
    (r"address already in use|errno 10048|eaddrinuse", lambda m: "That port is already in use. Close the other program or change the port."),
    (r"connection refused|econnrefused|errno 10061|actively refused", lambda m: "Nothing is listening at that address. Start the server first."),
    (r"keyerror: (.{1,40})", lambda m: f"The key {m[1]} doesn't exist."),
    (r"\b(typeerror|valueerror|attributeerror|indexerror|zerodivisionerror|unicodedecodeerror|unicodeencodeerror|importerror"
     r"|runtimeerror|assertionerror|timeouterror|oserror|jsondecodeerror): (.{1,80})", lambda m: f"{m[1]}: {m[2]}."),
    (r"npm (?:err!|error) (.{1,80})", lambda m: f"npm reports: {m[1]}."),
    (r"fatal: (.{1,80})", lambda m: f"Git says: {m[1]}."),
    (r"traceback \(most recent call last\)", lambda m: "A Python traceback is showing. The error is on its last line."),
]


def diagnose(text):
    """Plain-English hint for the most recent error visible in the OCR text (offline, no AI)."""
    low = re.sub(r"[\u2018\u2019`\u00b4]", "'", text.lower())
    best = None
    for pat, fn in ERRORS:
        for m in re.finditer(pat, low):
            if best is None or m.start() >= best[0]:
                best = (m.start(), fn(m))
    if best:
        return best[1]
    m = list(re.finditer(r"\b(?:error|exception|failed)\b[:\s]+(.{5,80})", low))
    return f"I see an error: {m[-1][1].strip()}." if m else None


def summarize_terminal(text):
    lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 2]
    if not lines:
        return "The terminal looks empty, or I couldn't read it."
    d = diagnose(text)
    return "The last lines say: " + " ... ".join(lines[-3:])[:240] + "." + (" " + d if d else "")


@act
def see_terminal(name=""):
    """Reads the terminal (or the window called `name`) with offline OCR and says what it shows."""
    return summarize_terminal(read_screen("terminal", name)[1])


@act
def terminal_errors():
    """Looks for an error on the terminal and explains it in plain words."""
    return diagnose(_fresh_terminal()) or "I don't see an error on it."


@act
def learn_screen():
    """Saves what is on the terminal into my private knowledge."""
    _store("screen:" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), _fresh_terminal())
    return "Learned what is on that window."


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv}


def safe_eval(expr):
    """Arithmetic only (+ - * /); no eval(), no exponent bombs."""
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and type(n.value) in (int, float):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            v = ev(n.operand)
            return -v if isinstance(n.op, ast.USub) else v
        raise ValueError("unsupported expression")
    if len(expr) > 100:
        raise ValueError("too long")
    return ev(ast.parse(expr.strip(), mode="eval"))




# ---------------------------------------------------------------- understanding commands
HOT = {r"(?:minimi[sz]e all|show (?:the )?desktop)": "win+d", r"(?:switch|next) window": "alt+tab",
       r"close window": "alt+f4", r"new tab": "ctrl+t", r"close tab": "ctrl+w", r"next tab": "ctrl+tab",
       r"previous tab": "ctrl+shift+tab", r"(?:refresh|reload)(?: page)?": "f5", r"copy(?: that)?": "ctrl+c",
       r"paste(?: that)?": "ctrl+v", r"cut(?: that)?": "ctrl+x", r"select all": "ctrl+a", r"undo": "ctrl+z",
       r"redo": "ctrl+y", r"save(?: it| file)?": "ctrl+s", r"maximi[sz]e(?: window)?": "win+up",
       r"minimi[sz]e(?: window)?": "win+down", r"snap left": "win+left", r"snap right": "win+right",
       r"task view": "win+tab", r"(?:press )?enter": "enter", r"(?:press )?escape": "esc",
       r"go back": "alt+left", r"zoom in": "ctrl+=", r"zoom out": "ctrl+-", r"find": "ctrl+f"}
RULES = []


def R(p, a, f=lambda m: {}):
    RULES.append((re.compile(p), a, f))


for _p, _k in HOT.items():
    R(_p, "press", lambda m, k=_k: {"keys": k})
R(r"(?:what(?:'s| is) (?:on|in|happening in|going on in)|read|look at|check|describe|show me|tell me what(?:'s| is) on) (?:my |the )?(?:terminal|console|command prompt|powershell|cmd)(?: (?:output|window|screen))?", "see_terminal")
R(r"what does (?:my |the )?(?:terminal|console) (?:say|show)", "see_terminal")
R(r"(?:read|look at|check) (?:the )?window (?:called |named )?(.+)", "see_terminal", lambda m: {"name": m[1]})
R(r"(?:is there (?:an? )?error|any errors?|what(?:'s| is| was) the (?:error|problem|issue)|explain (?:the |this )?error|why did (?:it|that) (?:fail|crash)|fix (?:the |this )?error)(?: (?:on|in) (?:the |my )?(?:terminal|screen|console))?", "terminal_errors")
R(r"learn (?:from |about )?(?:this|the|my) (?:screen|terminal|window)(?: output)?", "learn_screen")
R(r"(?:how (?:do|can) i fix (?:this|the) error|(?:find|search for|look up) (?:a |the )?fix(?: for (?:this|the) error)?)", "terminal_fix")
R(r"(?:find out|research|look into|search the (?:web|internet) for|ask the internet(?: about)?) (.+)", "research", lambda m: {"question": m[1]})
R(r"(?:run|execute) (?:the )?command (.+)", "shell", lambda m: {"cmd": m[1]})
R(r"(?:open|show)(?: my)? (downloads|documents|desktop|pictures|music|videos) folder", "folder", lambda m: {"name": m[1]})
R(r"(?:what(?:'s| is) on (?:my )?screen|(?:look at|read|describe) (?:my )?screen)", "look")
R(r"(?:switch to|focus on|focus) (.+)", "focus_window", lambda m: {"name": m[1]})
R(r"(?:export|build|create) (?:my )?(?:training|model)(?: data)?", "export_training")
R(r"(?:set )?(?:the )?brightness (?:to )?(\d{1,3})(?: percent)?", "brightness", lambda m: {"level": m[1]})
R(r"(?:find|open|show) (?:the |my )?file (.+)", "find_file", lambda m: {"name": m[1]})
R(r"(?:go to|take me to|bring up|pull up|show me) (.+)", "open_app", lambda m: {"name": m[1]})
R(r"(?:open|launch|start) (.+)", "open_app", lambda m: {"name": m[1]})
R(r"(?:force close|force quit|kill) (.+)", "kill_app", lambda m: {"name": m[1]})
R(r"(?:close|quit|exit) (.+)", "close_app", lambda m: {"name": m[1]})
R(r"(?:play|pause|resume|stop)(?: (?:the )?(?:music|song|video|media))?", "media", lambda m: {"cmd": "play"})
R(r"(?:next|skip)(?: (?:track|song|video))?", "media", lambda m: {"cmd": "next"})
R(r"previous(?: (?:track|song))?", "media", lambda m: {"cmd": "prev"})
R(r"(?:search|google|look up)(?: for)? (.+)", "search", lambda m: {"query": m[1]})
R(r"(?:play|search youtube for|youtube) (.+)", "youtube", lambda m: {"query": m[1]})
R(r"type (.+)", "type_text", lambda m: {"text": m[1]})
R(r"press (.+)", "press", lambda m: {"keys": m[1].replace("control", "ctrl").replace(" plus ", "+").replace(" ", "+")})
R(r"(?:set )?(?:the )?volume (?:to )?(\d{1,3})(?: percent)?", "set_volume", lambda m: {"level": m[1]})
R(r"(?:volume up|louder|turn (?:it |the volume )?up|increase (?:the )?volume)", "volume", lambda m: {"dir": "up"})
R(r"(?:volume down|quieter|turn (?:it |the volume )?down|(?:decrease|lower) (?:the )?volume)", "volume", lambda m: {"dir": "down"})
R(r"(?:un)?mute(?: (?:the )?(?:sound|volume|audio))?", "volume", lambda m: {"dir": "mute"})
R(r"(?:take (?:a )?)?screenshot", "screenshot")
R(r"lock(?: the)?(?: computer| pc| screen)?", "lock")
R(r"(?:go to )?sleep(?: mode)?|put (?:the )?(?:pc|computer) to sleep", "sleep_pc")
R(r"(?:shut ?down|turn off)(?: the)?(?: computer| pc)?", "shutdown")
R(r"(?:restart|reboot)(?: the)?(?: computer| pc)?", "restart")
R(r"(?:cancel|abort) (?:the )?(?:shutdown|restart)", "cancel_shutdown")
R(r"(double |right )?click", "click", lambda m: {"kind": (m[1] or "left").strip()})
R(r"scroll (up|down)", "scroll", lambda m: {"dir": m[1]})
R(r"move (?:the )?(?:mouse|cursor) (up|down|left|right)", "move_mouse", lambda m: {"dir": m[1]})
R(r"(?:take a note|note|write down|add note)(?: that)? (.+)", "note", lambda m: {"text": m[1]})
R(r"read (?:my )?notes", "read_notes")
R(r"(?:set (?:a )?)?timer for (\d+) (second|minute|hour)s?",
  "timer", lambda m: {"seconds": int(m[1]) * {"second": 1, "minute": 60, "hour": 3600}[m[2]]})
R(r"remind me in (\d+) (second|minute|hour)s? to (.+)", "timer",
  lambda m: {"seconds": int(m[1]) * {"second": 1, "minute": 60, "hour": 3600}[m[2]], "label": m[3]})
R(r".*\b(?:battery|cpu|ram|memory usage|system status)\b.*", "sysinfo")
R(r".*\b(?:weather|news)\b.*", "search", lambda m: {"query": m.string})
R(r".*\btime\b.*", "clock", lambda m: {"what": "time"})
R(r".*\b(?:date|day is it|today)\b.*", "clock", lambda m: {"what": "date"})
R(r".*\bjoke\b.*", "say", lambda m: {"text": "I would tell you a UDP joke, but you might not get it."})
R(r".*\b(?:who are you|your name)\b.*", "say", lambda m: {"text": "I'm Jarvis, and I only listen to you."})
R(r".*\b(?:thanks|thank you)\b.*", "say", lambda m: {"text": "Anytime."})

VERBS = (r"(?:open|launch|start|close|quit|search|google|play|type|press|set|turn|take|lock|scroll|click|"
         r"minimi[sz]e|maximi[sz]e|switch|mute|unmute|pause|resume|next|previous|remind|note|run|volume|"
         r"increase|decrease|lower|new|copy|paste|save|refresh|screenshot|shut|restart|sleep|read|kill)")
SOFT = re.compile(r"\s*(?:,|;|\band then\b|\bthen\b|\bafter that\b|\band\b)\s+(?=%s\b)" % VERBS)
HARD = re.compile(r"\s*(?:;|\band then\b|\bthen\b|\bafter that\b)\s+(?=%s\b)" % VERBS)
FREE = re.compile(r"(?:type|learn|teach|remember|when i say|if i say|note|take a note|write down|remind|run (?:the )?command)\b")


def split(t):
    if FREE.match(t):
        return [t]
    return (HARD if re.match(r"(?:search|google|look up|play|youtube)\b", t) else SOFT).split(t)


def parse(t):
    c = t
    for pat, rep in ((r"\bplus\b", "+"), (r"\bminus\b", "-"), (r"\b(?:times|multiplied by)\b", "*"),
                     (r"\bdivided by\b", "/"), (r"(?<=\d)\s*x\s*(?=\d)", "*")):
        c = re.sub(pat, rep, c)
    m = re.fullmatch(r"(?:calculate|what is|what's|whats)\s+([\d\s+\-*/().]+)", c)
    if m and re.search(r"\d", m[1]):
        return ("calc", {"expr": m[1]})
    for rx, a, f in RULES:
        m = rx.fullmatch(t)
        if m:
            return (a, f(m))


def plan(t, d=0):
    k = t if t in MEM["macros"] else (difflib.get_close_matches(t, list(MEM["macros"]), 1, 0.85) or [None])[0]
    if k and d < 3:
        out = []
        for c in MEM["macros"][k]:
            p = plan(c, d + 1)
            if p is None:
                return None
            out += p
        return out
    out = []
    for part in split(t):
        s = parse(part.strip(" .,!?"))
        if not s:
            return None
        out.append(s)
    return out


def _post(url, headers, body, timeout=60):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", **headers})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def system_prompt(q=""):
    docs = "; ".join(f"{n}({', '.join(inspect.signature(f).parameters)})" for n, f in ACT.items())
    s = ('You control a Windows laptop. Reply ONLY with JSON {"steps":[{"action":"name","args":{}}]}. '
         f"Allowed actions: {docs}. Use say(text) to answer questions or chat briefly. "
         "After open_app, a 3 second pause is automatic. Use focus_window(name) before type_text/press if the target app "
         "may not be in front, and look(question) when you need to see the screen. "
         "For anything the actions don't cover, use shell(cmd) with a cmd or PowerShell command. "
         f"Known facts about the user: {json.dumps(MEM['facts'])}")
    ex = difflib.get_close_matches(q, list(MEM["learned"]), 4, 0.3) if q else []
    if ex:
        s += " Examples of this user's commands: " + " ".join(
            f'"{k}" -> ' + json.dumps({"steps": [{"action": a, "args": g} for a, g in MEM["learned"][k]]}) for k in ex)
    notes = recall(q) if q else []
    if notes:
        s += " Reference notes the user taught you (use them to answer questions): " + " | ".join(notes)
    return s


def ask_llm(text):
    """AI brain (Claude API or local Ollama): turns ANY sentence into the safe actions above."""
    if not (ANTHROPIC_KEY or OLLAMA_MODEL):
        return None
    sysmsg = system_prompt(text)
    msgs = []
    for u, a in HIST:
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": text})
    try:
        if ANTHROPIC_KEY:
            r = _post("https://api.anthropic.com/v1/messages", {"x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01"},
                      {"model": CLAUDE_MODEL, "max_tokens": 800, "system": sysmsg, "messages": msgs})
            raw = r["content"][0]["text"]
        else:
            r = _post("http://localhost:11434/api/chat", {}, {"model": OLLAMA_MODEL, "stream": False, "format": "json",
                      "messages": [{"role": "system", "content": sysmsg}] + msgs}, 90)
            raw = r["message"]["content"]
        out = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
        return [(x["action"], x.get("args") or {}) for x in out["steps"] if x.get("action") in ACT] or None
    except Exception as e:
        log(f"llm error {e}")
        return None


def execute(steps):
    global LAST
    LAST, out = steps, []
    for i, (a, args) in enumerate(steps):
        try:
            out.append(ACT[a](**args))
            if a == "open_app" and i < len(steps) - 1:
                time.sleep(3)
        except Friendly as e:
            out.append(str(e))
        except Exception as e:
            out.append(f"{a} failed: {e}")
        MEM["usage"][a] = MEM["usage"].get(a, 0) + 1
        log(f"{a} {args}")
    save()
    return " ".join(filter(None, out)) or "Done."


def meta(t, orig=None):
    """Learning commands."""
    global GUEST_MODE

    def val(m, i):      # the same words as spoken, keeping capital letters (names, places)
        o = re.fullmatch(m.re.pattern, orig or "", re.I)
        return o[i] if o else m[i]

    m = re.fullmatch(r"(?:learn(?: that)?|teach(?: yourself)?|when i say|if i say)\s+(.+?)\s+(?:means|mean|do|then)\s+(.+)", t)
    if m:
        ph, cmd = m[1].strip(" ,"), m[2]
        if plan(cmd) is None:
            return f"I don't understand '{cmd}' yet, so I can't learn that."
        MEM["macros"][ph] = [c.strip(" .,") for c in split(cmd)]
        save()
        return f"Learned. When you say {ph}, I'll {cmd}."
    m = re.fullmatch(r"(?:turn |switch |set )?guest mode (on|off)|(allow|block) guests", t)
    if m:
        GUEST_MODE = (m[1] or m[2]) in ("on", "allow")
        return "Guests can chat with me." if GUEST_MODE else "Okay, I will only listen to you."
    m = (re.fullmatch(r"(?:when|if) (?:someone|anyone|somebody|i) (?:asks?|asked) (.+?),? (?:say|answer|reply|tell (?:them|me)) (.+)", t)
         or re.fullmatch(r"(?:learn |remember )?(?:that )?the answer to (.+?) (?:is|=) (.+)", t))
    if m:
        MEM["qa"][m[1].strip(" ?,")] = val(m, 2).strip()
        save()
        return f"Got it. When someone asks {m[1]}, I will say {val(m, 2)}."
    m = re.fullmatch(r"remember (?:that )?(.+?) (?:is|are) (.+)", t)
    if m:
        MEM["facts"][m[1]] = val(m, 2)
        save()
        return f"Okay, {m[1]} is {val(m, 2)}."
    m = re.fullmatch(r"(?:what is|what's|whats) (my .+)", t)
    if m and m[1] in MEM["facts"]:
        return f"{m[1]} is {MEM['facts'][m[1]]}."
    m = re.fullmatch(r"forget (?:that |about )?(.+)", t)
    if m:
        hit = MEM["macros"].pop(m[1], None) or MEM["facts"].pop(m[1], None) or MEM["learned"].pop(m[1], None) or MEM["qa"].pop(m[1], None)
        save()
        return f"Forgot {m[1]}." if hit else f"I don't have {m[1]}."
    if re.fullmatch(r"what (?:have you|did you) learn(?:ed)?|list (?:my )?(?:commands|macros)", t):
        items = [f"{k}: {', '.join(v)}" for k, v in MEM["macros"].items()] + [f"{k} is {v}" for k, v in MEM["facts"].items()] + [f"{k} (auto-learned)" for k in MEM["learned"]]
        items += [f"{k}? {v}" for k, v in MEM["qa"].items()]
        n = len({k["src"] for k in KNOW})
        if n:
            items.append(f"{n} documents or pages")
        return " | ".join(items) or "Nothing yet. Say: learn that work mode means open chrome."


def _handle(text):
    global PENDING, ASKED
    t = text.lower().replace("\u2019", "'").strip()
    t = strip_wake(t).strip(" .!?,")
    t = re.sub(r"^(?:(?:please|can you|could you|will you|would you|i want you to|i need you to|go ahead and|just)\s+)+", "", t).strip(" .!?,")
    orig = re.sub(r"^(?:(?:please|can you|could you|will you|would you|i want you to|i need you to|go ahead and|just)\s+)+", "",
                  strip_wake(text.replace("\u2019", "'").strip()), flags=re.I).strip(" .!?,")
    if not t:
        return _hello(True) if re.match(r"(?:hi|hello|hey|yo|namaste)\b", text.lower()) else "Yes?"
    if PENDING and time.time() < PENDING[1]:
        steps, PENDING = PENDING[0], None
        if re.fullmatch(r"(?:yes|yeah|yep|confirm|do it|sure|go ahead|proceed)", t):
            return execute(steps)
        if re.fullmatch(r"(?:no|nope|cancel|stop|never mind|don't)", t):
            return "Cancelled."
    PENDING = None
    if ASKED and time.time() - ASKED[1] < 120:          # I asked for the answer to something; you are giving it
        m = re.fullmatch(r"(?:the answer is|answer is|it(?:'s| is)|its)\s+(.+)", t)
        if m:
            MEM["qa"][ASKED[0]] = (re.fullmatch(m.re.pattern, orig, re.I) or m)[1]
            save()
            ASKED = None
            return "Thanks, I will remember that."
    m = re.search(r"learn (?:from|about) (https://\S+)", text, re.I)
    if m:
        return execute([("learn_url", {"url": m[1].rstrip(".,")})])
    m = re.search(r"learn from (?:folder|file) (.+)", text, re.I)
    if m:
        return execute([("learn_docs", {"path": m[1].strip(" .")})])
    r = meta(t, orig)
    if r:
        return r
    if re.fullmatch(r"(?:again|repeat(?: that)?|do that again)", t) and LAST:
        return execute(LAST)
    cached = MEM["learned"].get(t)
    steps = plan(t) or ([(a, g) for a, g in cached] if cached else None)
    ai = False
    if not steps:                                       # talk: small talk, taught answers, learned documents
        r = local_answer(t, True)
        if r:
            return r
        steps = ask_llm(text)                           # optional extra brain (Ollama / Claude) if you set one up
        ai = bool(steps)
    if not steps:
        r = web_answer(t)                               # free Wikipedia lookup, saved so it works offline next time
        if r:
            return r
        MEM["unknown"] = (MEM["unknown"] + [t])[-30:]
        save()
        if WH.match(t):
            ASKED = (t, time.time())
            return "I don't know that yet. If you tell me, I will remember it. Say: the answer is, and then the answer."
        return f"I don't know '{t}' yet. Teach me: say learn that {t} means, followed by a command."
    risky = [a for a, _ in steps if a in SENSITIVE]
    if risky:
        PENDING = (steps, time.time() + 30)
        return f"That will run {', '.join(risky)}. Say yes to confirm."
    reply = execute(steps)
    if ai and "failed" not in reply and not any(a in SENSITIVE or a == "say" for a, _ in steps):
        MEM["learned"][t] = [list(x) for x in steps]      # next time: instant, offline
        save()
    return reply


def handle(text):
    with LOCK:
        reply = _handle(text)
    HIST.append((text, reply))
    del HIST[:-6]
    speak_async(reply)
    return reply


# ---------------------------------------------------------------- conversation (offline chat, taught answers, learned documents)
GUEST_SAFE = {"clock", "calc", "say"}                  # the only things a guest's words may ever run
WH = re.compile(r"(?:what|who|when|where|why|how|which|tell me)\b.*")
WEBQ = re.compile(r"(?:what|who|when|where|which|how(?! to\b| do\b| can\b| should\b)|tell me about|explain|define)\b.*")
TOPIC = re.compile(r"(?:what(?:'s| is| are| was)|who(?:'s| is| was)|tell me about|explain|define|what do you know about"
                   r"|do you know about)\s+(?:an? |the )?(.+)")
JOKES = ["Why do programmers prefer dark mode? Because light attracts bugs.",
         "There are ten kinds of people: those who understand binary and those who don't.",
         "I would tell you a UDP joke, but you might not get it.",
         "A SQL query walks into a bar, sees two tables and asks: can I join you?"]


def _nk(s):
    return re.sub(r"^(?:the|a|an)\s+", "", s.strip())


def _hello(owner):
    h = datetime.datetime.now().hour
    name = MEM["facts"].get("my name") if owner else None
    return f"Good {'morning' if h < 12 else 'afternoon' if h < 18 else 'evening'}{', ' + name if name else ''}. How can I help?"


SMALL = [(re.compile(p), f) for p, f in [
    (r"(?:hi|hello|hey|yo|hola|namaste|good (?:morning|afternoon|evening))(?: there)?", _hello),
    (r"how are you(?: doing| today)?|how(?:'s| is) it going|what(?:'s| is) up|sup",
     lambda o: random.choice(["Running smoothly, thanks. How are you?", "All systems normal, and glad to talk. How about you?"])),
    (r"(?:i(?:'m| am) )?(?:fine|good|great|doing well|okay|ok)(?: thanks| thank you)?", lambda o: "Glad to hear it. What shall we do?"),
    (r"i(?:'m| am) (?:sad|tired|bored|stressed|upset|angry)", lambda o: "I'm sorry to hear that. I'm here if you want to talk about it."),
    (r"what(?:'s| is) your name|who are you", lambda o: "I'm Jarvis, a voice assistant running on this computer. I take commands only from my owner, but I'm happy to talk with anyone."),
    (r"who (?:made|built|created|programmed) you", lambda o: "I was built as a personal assistant that runs on this computer, offline."),
    (r"are you (?:a )?(?:robot|human|real|an ai|ai)", lambda o: "I'm a program running on this computer, not a person. But I do enjoy the conversation."),
    (r"what can you do|what are your (?:features|abilities)|help", lambda o: "I can chat and answer questions for anyone. For my owner I also control the computer, read the terminal, and learn new commands and facts."),
    (r"(?:tell me |say |another )*(?:a |one )?joke(?: please)?|make me laugh", lambda o: random.choice(JOKES)),
    (r"(?:thanks|thank you)(?: so much| very much)?", lambda o: random.choice(["Anytime.", "You're welcome.", "Happy to help."])),
    (r"(?:good ?night|goodbye|bye(?: bye)?|see you(?: later| soon)?)", lambda o: "Goodbye. I'll be here when you need me."),
    (r"are you (?:there|awake|listening)|you there", lambda o: "Yes, I'm here and listening."),
    (r"(?:i love you|you(?:'re| are) (?:great|awesome|smart|amazing|the best))", lambda o: "That's kind of you. I'll keep trying to deserve it."),
    (r"what(?:'s| is) the meaning of life", lambda o: "Forty-two, according to a famous computer. Personally, I'd say good company and curiosity."),
]]


def local_answer(t, owner=True):
    """Everything I can answer without the internet or any AI: small talk, taught answers, facts, learned documents."""
    for rx, fn in SMALL:
        if rx.fullmatch(t):
            return fn(owner)
    qa = MEM["qa"]
    k = t if t in qa else (difflib.get_close_matches(t, list(qa), 1, 0.8) or [None])[0]
    if k:
        return qa[k]
    m = TOPIC.fullmatch(t)
    if m:
        facts = {_nk(a): (a, b) for a, b in MEM["facts"].items()} if owner else {}
        key = _nk(m[1])
        hit = key if key in facts else (difflib.get_close_matches(key, list(facts), 1, 0.8) or [None])[0]
        if hit:
            return f"{facts[hit][0]} is {facts[hit][1]}."
    if WH.match(t) or len(_kw(t)) >= 2:
        return best_sentences(t, recall(t, 3, owner))[:350] or None
    return None


HEADERS = {"User-Agent": "Mozilla/5.0 (Jarvis local assistant)"}
TECH = re.compile(r"\b(?:error|exception|traceback|python|pip|npm|node|javascript|git|docker|install|code|function|module|sql|react|windows|powershell)\b")


def _check_url(url):
    """Only public https addresses: never your own network (also re-checked on every redirect)."""
    import ipaddress, socket
    u = urlparse(url)
    if u.scheme != "https" or not u.hostname:
        raise ValueError("https only")
    for a in socket.getaddrinfo(u.hostname, None):
        ip = ipaddress.ip_address(a[4][0].split("%")[0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise ValueError("local address")


class _Guard(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _check_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def http_get(url, limit=1_500_000, timeout=10):
    _check_url(url)
    data = urllib.request.build_opener(_Guard).open(urllib.request.Request(url, headers=HEADERS), timeout=timeout).read(limit)
    if data[:2] == b"\x1f\x8b":                       # StackExchange always gzips
        data = gzip.decompress(data)
    return data.decode("utf-8", "ignore")


def _plain(h):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", h))).strip()


def page_text(h):
    h = re.sub(r"(?is)<(script|style|noscript|nav|header|footer|aside|form|svg)\b.*?</\1>", " ", h)
    return _plain(h)


def parse_ddg(h):
    """DuckDuckGo html results -> [(url, snippet)]"""
    out = []
    for chunk in h.split('class="result__a"')[1:]:
        m = re.search(r'href="([^"]+)"', chunk)
        if not m:
            continue
        u = html.unescape(m[1])
        q = re.search(r"uddg=([^&]+)", u)
        u = unquote(q[1]) if q else u
        if u.startswith("//"):
            u = "https:" + u
        if not u.startswith("https://") or "duckduckgo.com" in u:
            continue
        sn = re.search(r'class="result__snippet"[^>]*>(.*?)</a>', chunk, re.S)
        out.append((u, _plain(sn[1]) if sn else ""))
    return out


def wiki(query):
    """Key-less lookup of the top Wikipedia article -> [(label, store key, text)]."""
    url = ("https://en.wikipedia.org/w/api.php?action=query&generator=search&gsrlimit=1&prop=extracts&exintro=1"
           "&explaintext=1&exsentences=4&redirects=1&format=json&gsrsearch=" + quote_plus(query))
    r = json.loads(http_get(url, 300_000, 8))
    p = next(iter(r.get("query", {}).get("pages", {}).values()), None)
    return [("Wikipedia", "wikipedia:" + p["title"], p["extract"])] if p and p.get("extract") else []


def ddg_instant(q):
    """DuckDuckGo instant answers (calculations, definitions, short facts)."""
    r = json.loads(http_get("https://api.duckduckgo.com/?format=json&no_html=1&skip_disambig=1&q=" + quote_plus(q), 300_000, 8))
    t = next((r[k] for k in ("Answer", "AbstractText", "Definition") if isinstance(r.get(k), str) and r.get(k)), "")
    return [(r.get("AbstractSource") or "DuckDuckGo", "duckduckgo:" + q[:60], t)] if t else []


def stack(q):
    """Accepted Stack Overflow answer for programming and terminal-error questions."""
    base = "https://api.stackexchange.com/2.3/"
    r = json.loads(http_get(base + "search/advanced?order=desc&sort=relevance&accepted=True&pagesize=1&site=stackoverflow&q=" + quote_plus(q), 300_000, 8))
    if not r.get("items"):
        return []
    qid = r["items"][0]["question_id"]
    a = json.loads(http_get(base + f"questions/{qid}/answers?order=desc&sort=votes&pagesize=1&site=stackoverflow&filter=withbody", 300_000, 8))
    body = (a.get("items") or [{}])[0].get("body", "")
    t = _plain(re.sub(r"(?s)<pre.*?</pre>", " ", body))
    return [("Stack Overflow", f"stackoverflow:{qid}", t)] if t else []


def web_pages(q, n=3):
    """Search the open web (DuckDuckGo) and read the top pages. No key, no account."""
    links = parse_ddg(http_get("https://html.duckduckgo.com/html/?q=" + quote_plus(q), 600_000))[:n]

    def get(l):
        host = urlparse(l[0]).netloc.removeprefix("www.")
        try:
            return (host, l[0], (l[1] + " " + page_text(http_get(l[0])))[:20000])
        except Exception as e:
            log(f"page error {host}: {e}")
            return (host, l[0], l[1]) if l[1] else None
    with ThreadPoolExecutor(max(1, len(links))) as ex:
        return [p for p in ex.map(get, links) if p]


def web_answer(t, force=False, tech=False):
    """Unknown question -> ask every free source at once (DuckDuckGo, Wikipedia, Stack Overflow, the top web pages),
    pick the best-matching sentences, and save what was found so next time it works offline."""
    if not WEB_LEARN:
        return None
    if not force and (not WEBQ.fullmatch(t) or re.search(r"\b(?:you|your|yours)\b", t)):
        return None
    m = TOPIC.fullmatch(t)
    query = t if tech else (m[1] if m else " ".join(_kw(t)))
    if not query:
        return None
    jobs = {"instant": lambda: ddg_instant(query), "wiki": lambda: wiki(query), "pages": lambda: web_pages(query)}
    if tech or TECH.search(t):
        jobs["stack"] = lambda: stack(query)
    found = []
    with ThreadPoolExecutor(len(jobs)) as ex:
        futs = {k: ex.submit(f) for k, f in jobs.items()}
        for k, f in futs.items():
            try:
                found += [(k, *x) for x in f.result(timeout=25)]
            except Exception as e:
                log(f"web {k} error {e}")
    kws = set(_kw(query)) or set(_kw(t))
    best = None
    for kind, label, key, text in found:
        if not kws & set(_kw(text)):
            continue
        _store(key, text[:20000])
        s = best_sentences(t, [text])
        score = sum(w in s.lower() for w in kws) + (2 if kind == "instant" else 0) + (1 if kind in ("wiki", "stack") else 0)
        if s and (best is None or score > best[0]):
            best = (score, label, s)
    return f"According to {best[1]}: {best[2]}"[:380] if best else None


@act
def research(question):
    """Searches all free web sources (no API key), answers, and saves what it learned."""
    return web_answer(question.lower().strip(" .?"), force=True) or "I couldn't find anything reliable on that."


@act
def terminal_fix():
    """Looks up the error on the terminal on Stack Overflow and the web."""
    m = list(re.finditer(r"^.*(?:error|exception|failed|fatal)[^\n]{5,}$", _fresh_terminal(), re.I | re.M))
    q = re.sub(r"^[\w:\\/. ()-]*>\s*", "", m[-1][0].strip())[:150] if m else ""
    if not q:
        return "I don't see an error to look up."
    return web_answer(q.lower(), force=True, tech=True) or f"I couldn't find a fix for {q[:80]}."


def llm_chat(q):
    """Plain conversation from a local Ollama model (or Claude if a key is set). It is never given any actions."""
    if not (ANTHROPIC_KEY or OLLAMA_MODEL):
        return None
    sysm = "You are Jarvis, a friendly voice assistant. Answer in one or two short spoken sentences."
    notes = recall(q, 2, owner=False)
    if notes:
        sysm += " Reference notes: " + " | ".join(notes)
    try:
        if ANTHROPIC_KEY:
            r = _post("https://api.anthropic.com/v1/messages", {"x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01"},
                      {"model": CLAUDE_MODEL, "max_tokens": 300, "system": sysm, "messages": [{"role": "user", "content": q}]})
            return r["content"][0]["text"].strip()
        r = _post("http://localhost:11434/api/chat", {}, {"model": OLLAMA_MODEL, "stream": False,
                  "messages": [{"role": "system", "content": sysm}, {"role": "user", "content": q}]}, 60)
        return r["message"]["content"].strip()
    except Exception as e:
        log(f"llm chat error {e}")
        return None


def handle_guest(text, near=False):
    """Someone who isn't you: can chat and ask questions. Never gets PC control, teaching, or your private data."""
    global GUEST_UNTIL
    t = strip_wake(text.lower().replace("\u2019", "'").strip()).strip(" .!?,")
    t = re.sub(r"^(?:(?:please|can you|could you|will you|would you)\s+)+", "", t).strip(" .!?,")
    GUEST_UNTIL = time.time() + CONVO_SECS
    with LOCK:
        steps = plan(t) if t else None
        if not t:
            reply = _hello(False) if re.match(r"(?:hi|hello|hey|yo|namaste)\b", text.lower()) else "Yes? I'm listening."
        elif steps and all(a in GUEST_SAFE for a, _ in steps):
            out = []
            for a, g in steps:
                try:
                    out.append(ACT[a](**g))
                except Exception:
                    out.append("I couldn't work that out.")
            reply = " ".join(filter(None, out)) or "Done."
        elif steps:
            reply = ("I wasn't sure that was my owner's voice. Say it again closer to the microphone, or type it."
                     if near else "Sorry, I only take commands from my owner. I'm happy to chat or answer questions though.")
        else:
            reply = (local_answer(t, False) or llm_chat(t) or web_answer(t)
                     or "I'm not sure about that one. Ask me something else.")
    log(f"guest: {text!r} -> {reply[:80]}")
    speak_async(reply)
    return reply


def vocab():
    return ", ".join(["Jarvis"] + sorted(BUILTIN)[:25] + list(MEM["macros"])[:15])


# ---------------------------------------------------------------- web page
INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Jarvis</title>
<style>
:root{--bg:#07111f;--ink:#d8ecf7;--dim:#7d98ab;--arc:#4fd8ff;--bad:#ff6b6b;--ok:#6dffb0;--line:#17314a}
*{box-sizing:border-box}
body{margin:0;min-height:100vh;background:radial-gradient(circle at 50% 35%,#0d2238,var(--bg) 65%);color:var(--ink);
 font-family:"Segoe UI",system-ui,sans-serif;display:flex;flex-direction:column;align-items:center;padding:32px 20px;gap:22px}
h1{margin:0;font-weight:300;font-size:1.6rem;letter-spacing:.35em}
#orb{width:220px;height:220px;border-radius:50%;border:2px solid var(--arc);display:grid;place-items:center;
 box-shadow:0 0 calc(10px + var(--lvl,0)*70px) var(--arc),inset 0 0 30px #4fd8ff33;transition:border-color .2s,box-shadow .08s;position:relative}
#orb::after{content:"";position:absolute;inset:18px;border-radius:50%;border:1px dashed #4fd8ff66}
#orb.speaking{border-color:var(--ok)} #orb.rejected{border-color:var(--bad)} #orb.off{border-color:var(--line);box-shadow:none}
#state{font-size:.95rem;color:var(--dim);text-align:center;padding:0 24px}
button{background:transparent;color:var(--ink);border:1px solid var(--arc);border-radius:8px;padding:11px 20px;font:inherit;cursor:pointer}
button:hover,button:focus-visible{background:#4fd8ff22;outline:none} button:disabled{opacity:.4;cursor:default}
.panel{width:min(560px,100%);border:1px solid var(--line);border-radius:12px;padding:18px;background:#0a1a2c99}
.panel h2{margin:0 0 8px;font-size:1rem;font-weight:600}
.panel p{margin:6px 0;color:var(--dim);line-height:1.5}
#bar{height:6px;background:var(--line);border-radius:3px;overflow:hidden;margin:12px 0}
#bar i{display:block;height:100%;width:0;background:var(--arc);transition:width .3s}
#run{display:flex;gap:10px}
#log{width:min(560px,100%);display:flex;flex-direction:column;gap:8px;font-size:.95rem}
.row{padding:9px 12px;border-left:3px solid var(--line);background:#0a1a2c77;border-radius:0 8px 8px 0}
.row.you{border-color:var(--arc)} .row.jarvis{border-color:var(--ok)} .row.blocked{border-color:var(--bad);color:var(--dim)} .row.guest{border-color:#ffc857}
input{flex:1;background:#0a1a2c;color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:11px;font:inherit}
.hide{display:none!important}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
</head>
<body>
<h1>JARVIS</h1>
<div id="orb" class="off"></div>
<div id="state" aria-live="polite">Loading…</div>

<section id="enroll" class="panel hide">
  <h2>Teach Jarvis your voice</h2>
  <p>Press Record, then read the sentence aloud for 3–4 seconds. Do 5 samples, in a quiet room, at your normal volume.</p>
  <p id="phrase" style="color:var(--ink)"></p>
  <div id="bar"><i></i></div>
  <button id="rec">Record sample</button>
</section>

<section id="run" class="hide">
  <button id="toggle">Start listening</button>
  <button id="re">Re-enroll my voice</button>
</section>
<section id="typed" class="panel" style="display:flex;gap:10px"><input id="txt" placeholder="Type a command, e.g. open chrome and search for cats" autocomplete="off"><button id="send">Send</button></section>
<div id="log"></div>

<script>
const PHRASES=["The quick brown fox jumps over the lazy dog.","Jarvis, what time is it right now?","Open the pod bay doors and play some music.",
"Seven silver swans swam across the still blue lake.","Please search for the best coffee near me."];
const $=s=>document.querySelector(s), orb=$("#orb"), stateEl=$("#state"), logEl=$("#log");
let ctx, stream, proc, listening=false, loopId=0, stat;

const say=(t)=>stateEl.textContent=t;
const add=(cls,t)=>{const d=document.createElement("div");d.className="row "+cls;d.textContent=t;logEl.prepend(d)};
const TOKEN="__TOKEN__";
const post=(url,body)=>fetch(url,{method:"POST",headers:{"Content-Type":"application/octet-stream","X-Token":TOKEN},body});

async function refresh(){
  stat=await (await fetch("/api/status")).json();
  $("#enroll").classList.toggle("hide",stat.enrolled);
  $("#run").classList.toggle("hide",!stat.enrolled);
  if(!stat.enrolled){orb.className="off";$("#phrase").textContent=PHRASES[stat.samples%PHRASES.length];
    $("#bar i").style.width=stat.samples/stat.needed*100+"%";say(`Voice samples: ${stat.samples} of ${stat.needed}`)}
  else say(listening?"Listening for your voice…":"Ready. Press Start listening.");
}

async function openMic(){
  if(!ctx){
    stream=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:true,noiseSuppression:true}});
    ctx=new AudioContext(); const src=ctx.createMediaStreamSource(stream);
    proc=ctx.createScriptProcessor(4096,1,1); src.connect(proc); proc.connect(ctx.destination);
  }
  if(ctx.state==="suspended") await ctx.resume();
}

function toPCM16(chunks,inRate){
  let len=chunks.reduce((a,c)=>a+c.length,0), all=new Float32Array(len), o=0;
  for(const c of chunks){all.set(c,o);o+=c.length}
  const ratio=inRate/16000, n=Math.floor(len/ratio), out=new Int16Array(n);
  for(let i=0;i<n;i++){const a=Math.floor(i*ratio), b=Math.min(len,Math.max(a+1,Math.floor((i+1)*ratio)));
    let s=0;for(let j=a;j<b;j++)s+=all[j]; s/=Math.max(1,b-a); out[i]=Math.max(-1,Math.min(1,s))*32767}
  return out;
}

// Records until onChunk returns true
function capture({onChunk}){
  return new Promise(res=>{
    const chunks=[]; proc.onaudioprocess=e=>{
      const d=new Float32Array(e.inputBuffer.getChannelData(0)); const stop=onChunk(d,chunks);
      if(stop){proc.onaudioprocess=null;res(toPCM16(chunks,ctx.sampleRate))}};
  });
}
const rms=d=>{let s=0;for(let i=0;i<d.length;i++)s+=d[i]*d[i];return Math.sqrt(s/d.length)};
const level=r=>orb.style.setProperty("--lvl",Math.min(1,r*8));

// ---- enrollment
$("#rec").onclick=async()=>{
  $("#rec").disabled=true;
  try{
    await openMic(); say("Recording… read the sentence now."); const t0=performance.now();
    const pcm=await capture({onChunk:(d,c)=>{c.push(d);level(rms(d));return performance.now()-t0>4500}});
    level(0); say("Saving…");
    const r=await post("/api/enroll",pcm.buffer); const j=await r.json();
    await refresh();
    if(!r.ok) say(j.error||"Enrollment failed.");
    else if(stat.enrolled) add("jarvis","Voice enrolled. I will only respond to you.");
  }catch(e){say("Error: "+e.message)}
  $("#rec").disabled=false;
};
$("#re").onclick=async()=>{
  if(!confirm("Delete your voice profile and enroll again?"))return;
  listening=false;loopId++;$("#toggle").textContent="Start listening";
  await post("/api/reset");logEl.innerHTML="";refresh()};

// ---- listening loop (adaptive voice-activity detection: learns the room noise, ignores clicks and coughs)
$("#toggle").onclick=async()=>{
  if(listening){listening=false;loopId++;$("#toggle").textContent="Start listening";orb.className="off";say("Paused.");return}
  try{await openMic()}catch(e){say("Microphone unavailable: "+e.message);return}
  listening=true; $("#toggle").textContent="Stop listening"; loop(++loopId);
};
let floor=0.004;   // running estimate of the room's background level
async function loop(id){
  const alive=()=>listening&&id===loopId;
  while(alive()){
    orb.className=""; say("Listening…");
    let speaking=false,hot=0,lastVoice=0,start=0,voiced=0,pre=[];
    const pcm=await capture({onChunk:(d,c)=>{
      const r=rms(d),now=performance.now(); level(r);
      if(!alive())return true;
      const on=Math.max(0.012,floor*3.2), off=Math.max(0.008,floor*1.8);
      if(!speaking){
        floor=floor*0.97+Math.min(r,0.05)*0.03;              // keep learning the room while idle
        pre.push(d); if(pre.length>5)pre.shift();             // ~0.4 s pre-roll so the first syllable isn't clipped
        hot=r>on?hot+1:0;
        if(hot>=2){speaking=true;start=lastVoice=now;c.push(...pre)}
        return false}
      c.push(d);
      if(r>off){lastVoice=now;voiced++}
      return now-lastVoice>(voiced>8?850:550)||now-start>15000}});
    level(0); if(!alive())break;
    if(voiced<3||pcm.length<16000*0.6)continue;               // a click, cough or bump - not speech
    say("Checking voice…");
    try{
      const j=await (await post("/api/command",pcm.buffer)).json();
      if(!alive())break;
      if(j.error){say(j.error);await new Promise(r=>setTimeout(r,1000))}
      else if(j.accepted){ const g=j.role==="guest";
        add(g?"guest":"you",(g?"Guest: ":"")+j.heard); add("jarvis",j.reply); await speak(j.reply) }
      else{ orb.className="rejected"; if(j.score!==undefined) add("blocked",`Ignored (voice match ${j.score}${j.reason?`, ${j.reason}`:""})`);
        await new Promise(r=>setTimeout(r,600)) }
    }catch(e){say("Server error: "+e.message);await new Promise(r=>setTimeout(r,1000))}
  }
}
async function speak(t){
  orb.className="speaking"; say(t);
  try{let s=await (await fetch("/api/speaking")).json();
    if(s.server_tts){while(s.speaking){await new Promise(r=>setTimeout(r,200));s=await (await fetch("/api/speaking")).json()}return}}catch(e){}
  await new Promise(res=>{if(!("speechSynthesis" in window))return setTimeout(res,1000);
    speechSynthesis.cancel();const u=new SpeechSynthesisUtterance(t);u.rate=1.02;u.onend=u.onerror=res;setTimeout(res,15000);speechSynthesis.speak(u)})}
if("speechSynthesis" in window)speechSynthesis.getVoices();
async function sendText(){const v=$("#txt").value.trim();if(!v)return;$("#txt").value="";add("you",v);
  try{const j=await (await post("/api/text",v)).json();add("jarvis",j.reply);await speak(j.reply)}catch(e){say("Server error: "+e.message)}}
$("#send").onclick=sendText;$("#txt").onkeydown=e=>{if(e.key==="Enter")sendText()};
setInterval(async()=>{try{const j=await (await fetch("/api/events")).json();for(const m of j.events){add("jarvis",m);await speak(m)}}catch(e){}},3000);
refresh();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------- routes
@app.get("/")
def index():
    return Response(INDEX_HTML.replace("__TOKEN__", TOKEN), mimetype="text/html")


@app.get("/api/status")
def status():
    n = len(load_prints())
    return jsonify(samples=n, needed=SAMPLES_NEEDED, enrolled=n >= SAMPLES_NEEDED, threshold=THRESHOLD, guests=GUEST_MODE)


@app.get("/api/speaking")
def speaking():
    return jsonify(speaking=_spk_n > 0, server_tts=TTS_OK)


@app.get("/api/events")
def events():
    out = EVENTS[:]
    EVENTS.clear()
    return jsonify(events=out)


@app.post("/api/enroll")
def enroll():
    wav = clean_audio(pcm_from_request())
    if len(wav) < 16000 * 2:
        return jsonify(error="Sample too short. Speak for at least 2 seconds."), 400
    try:
        vec = embed(wav)
    except Exception:
        return jsonify(error="Couldn't hear speech in that sample. Try again, a bit louder."), 400
    with LOCK:
        prints = load_prints()
        if len(prints) >= SAMPLES_NEEDED:
            return jsonify(error="Already enrolled. Use re-enroll first."), 409
        prints = np.vstack([prints, vec[None]])
        np.save(VOICEPRINT, prints)
    return jsonify(samples=len(prints), needed=SAMPLES_NEEDED, enrolled=len(prints) >= SAMPLES_NEEDED)


@app.post("/api/reset")
def reset():
    with LOCK:
        if os.path.exists(VOICEPRINT):
            os.remove(VOICEPRINT)
    return jsonify(ok=True)


@app.post("/api/text")
def text_cmd():
    return jsonify(reply=handle(request.get_data(as_text=True)[:500]))


@app.post("/api/command")
def command():
    global AWAKE_UNTIL
    prints = load_prints()
    if len(prints) < SAMPLES_NEEDED:
        return jsonify(error="Not enrolled yet."), 400
    wav = pcm_from_request()
    if len(wav) < 16000 * 0.5:
        return jsonify(accepted=False, reason="too short")
    wav = clean_audio(wav)
    try:
        vec = embed(wav)
        score = voice_score(vec, prints)
    except Exception:
        return jsonify(accepted=False, reason="no speech")
    owner = score >= THRESHOLD
    if not owner and not GUEST_MODE:
        return jsonify(accepted=False, score=round(score, 2))
    with LOCK:
        text = transcribe(wav)
    if not text:
        return jsonify(accepted=False, score=round(score, 2), reason="no speech")
    now, woke = time.time(), heard_wake(text)
    if owner:                                   # your voice: full control
        if REQUIRE_WAKE_WORD and not woke and now > AWAKE_UNTIL:
            return jsonify(accepted=False, score=round(score, 2), heard=text, reason="no wake word")
        AWAKE_UNTIL = now + CONVO_SECS
        if ADAPT and score >= 0.82:             # learn your voice over time
            with LOCK:
                p = np.vstack([load_prints(), vec[None]])
                np.save(VOICEPRINT, np.vstack([p[:SAMPLES_NEEDED], p[-25:]]) if len(p) > 30 else p)
        return jsonify(accepted=True, role="owner", score=round(score, 2), heard=text, reply=handle(text))
    if GUEST_WAKE and not woke and now > GUEST_UNTIL:       # someone else: chat only, and only if addressed
        return jsonify(accepted=False, score=round(score, 2), heard=text, reason="not you, no wake word")
    return jsonify(accepted=True, role="guest", score=round(score, 2), heard=text,
                   reply=handle_guest(text, near=score >= THRESHOLD - 0.08))


if __name__ == "__main__":
    print("Loading models (first run downloads them)...")
    encoder(); whisper()
    if pyautogui is None:
        print("NOTE: pyautogui missing -> typing/mouse/hotkeys disabled. python -m pip install pyautogui psutil")
    if not (ANTHROPIC_KEY or OLLAMA_MODEL):
        try:
            OLLAMA_MODEL = json.loads(urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2).read())["models"][0]["name"]
        except Exception:
            pass
    print("Built-in offline brain: ON (chat, learning, terminal reading).  Extra AI brain:",
          "Claude API" if ANTHROPIC_KEY else f"Ollama {OLLAMA_MODEL}" if OLLAMA_MODEL else "none (not needed)")
    try:
        import rapidocr_onnxruntime
        threading.Thread(target=lambda: ocr(__import__("PIL.Image", fromlist=["Image"]).new("RGB", (400, 100), "white")), daemon=True).start()
        print("Terminal reading: ON (offline OCR)")
    except ImportError:
        print("Terminal reading: OFF -> python -m pip install rapidocr-onnxruntime")
    print("Guests may chat:", "YES (they must say Jarvis first)" if GUEST_MODE else "no", "| only your voice can give commands")
    print(f"Jarvis running at http://localhost:{PORT}")
    threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()
    app.run(host="127.0.0.1", port=PORT, debug=False)
