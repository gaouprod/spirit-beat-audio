"""COPIE CONFORME de SPIRIT-BEAT workers/audio/spirit_audio_worker/beat_grid.py (2026-11-27).

Mêmes fonctions que le worker d'analyse, pour que les grilles calculées ici
soient identiques à celles du catalogue. Toute modification se fait d'abord
dans SPIRIT-BEAT, puis est recopiée ici.

Grille rythmique d'un morceau (table TrackBeatGrid), calculée à chaque analyse.

Sert aux enchaînements calés sur le temps des séances 40fit : on garde les
TEMPS DÉTECTÉS du début (30 s) et de la fin (45 s) du morceau, pas une grille
rectiligne. Le tempo des morceaux générés dérive légèrement et leur calage peut
sauter aux raccords d'un morceau prolongé ; un enchaînement n'a besoin que du
début de l'entrant et de la fin du sortant.

Moteur de détection (`AUDIO_BEAT_ENGINE`) :
  - `beat_this` (CPJKU 2024 : temps ET premiers temps de mesure, très fiable sur
    l'électro) s'il est installé (image construite avec INSTALL_BEAT_THIS=1) ;
  - sinon `librosa` (temps au pas de 5,8 ms ; premier temps de mesure estimé
    par l'énergie grave, la grosse caisse) ;
  - `auto` (défaut) : beat_this s'il est disponible, sinon librosa.

Profil des bords (`edges`) : énergie moyenne de chacune des 16 premières et
16 dernières mesures (pas d'une mesure = 4 temps au tempo d'ensemble, depuis le
début du son et jusqu'à sa fin), en rapport à l'énergie moyenne du morceau :
intro calme ou attaque directe, fin franche ou fondu. 40fit choisit d'après lui
la façon d'enchaîner deux morceaux.

Format compact d'un passage : { bpm, t0, d (écarts entre temps, ms entiers),
phase (index du premier temps de mesure, puis tous les 4), n, regularity,
dbAgree }. Les temps se retrouvent par cumul : t0 + Σ d / 1000.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

BEATS_PER_BAR = 4
HEAD_SEC = 30.0
TAIL_SEC = 45.0
_BEAT_THIS = None


def sound_bounds(y: np.ndarray, sr: int, hop_sec: float = 0.05, floor_db: float = -40.0) -> tuple[float, float]:
    """Début et fin du son utile (au-dessus de −40 dB du maximum, trames de 50 ms)."""
    hop = max(1, int(sr * hop_sec))
    frames = len(y) // hop
    if frames == 0:
        return 0.0, len(y) / sr
    rms = np.sqrt(np.mean(y[: frames * hop].reshape(frames, hop) ** 2, axis=1) + 1e-12)
    loud = np.nonzero(rms > rms.max() * 10 ** (floor_db / 20))[0]
    if not len(loud):
        return 0.0, len(y) / sr
    return round(float(loud[0] * hop_sec), 2), round(float((loud[-1] + 1) * hop_sec), 2)


def local_beats(beats: list[float], downbeats: list[float], start: float, end: float) -> dict[str, Any] | None:
    """Temps de la fenêtre [start, end], trous comblés, phase de la mesure par vote."""
    win = np.asarray([b for b in beats if start <= b <= end], dtype=float)
    if len(win) < 16:
        return None
    med = float(np.median(np.diff(win)))
    # Temps manqués : un intervalle de ~2 ou 3 temps est recoupé régulièrement.
    filled = [win[0]]
    for a, b in zip(win[:-1], win[1:]):
        k = int(round((b - a) / med))
        if 2 <= k <= 3:
            filled.extend(a + (b - a) * j / k for j in range(1, k))
        filled.append(b)
    win = np.asarray(filled)
    ibi = np.diff(win)
    regularity = float(np.mean(np.abs(ibi / med - 1) < 0.05))
    idx = []
    for d in downbeats or []:
        if start <= d <= end:
            i = int(np.argmin(np.abs(win - d)))
            if abs(win[i] - d) < 0.07:
                idx.append(i)
    if idx:
        counts = np.bincount(np.asarray(idx) % BEATS_PER_BAR, minlength=BEATS_PER_BAR)
        phase = int(counts.argmax())
        agree = float(counts[phase] / counts.sum())
    else:
        phase, agree = 0, 0.0
    ms = np.round(win * 1000).astype(int)
    d = np.diff(ms)
    return {
        # Tempo d'après l'intervalle moyen : la médiane d'intervalles tramés est biaisée.
        "bpm": round(60000.0 / float(np.mean(d)), 3),
        "t0": round(float(ms[0]) / 1000, 3),
        "d": [int(x) for x in d],
        "phase": phase,
        "n": int(len(win)),
        "regularity": round(regularity, 3),
        "dbAgree": round(agree, 3),
    }


EDGE_BARS = 16


def bar_energies(y: np.ndarray, sr: int, bpm: float, sound_start: float, sound_end: float, bars: int = EDGE_BARS) -> dict[str, Any] | None:
    """Énergie RMS moyenne par mesure au début et à la fin du son, rapportée à la moyenne du morceau."""
    if not (bpm and bpm > 40) or len(y) == 0:
        return None
    hop = 512
    frames = len(y) // hop
    if frames < 8:
        return None
    rms = np.sqrt(np.mean(y[: frames * hop].reshape(frames, hop) ** 2, axis=1) + 1e-12)
    times = (np.arange(frames) + 0.5) * hop / sr
    sound = (times >= sound_start) & (times < sound_end)
    mean = float(rms[sound].mean()) if sound.any() else float(rms.mean())
    if mean <= 0:
        return None
    bar = 240.0 / bpm

    def window(a: float, b: float) -> float | None:
        sel = (times >= a) & (times < b)
        return round(float(rms[sel].mean()) / mean, 2) if sel.any() else None

    head = [window(sound_start + k * bar, sound_start + (k + 1) * bar) for k in range(bars)]
    tail = [window(sound_end - (bars - k) * bar, sound_end - (bars - k - 1) * bar) for k in range(bars)]
    if any(v is None for v in head + tail):
        return None
    return {"barSec": round(bar, 4), "head": head, "tail": tail}


def _beat_this_model():
    global _BEAT_THIS
    if _BEAT_THIS is None:
        from beat_this.inference import File2Beats  # import paresseux : torch n'est chargé qu'au besoin

        _BEAT_THIS = File2Beats(checkpoint_path="final0", device="cpu", dbn=False)
    return _BEAT_THIS


def beat_this_available() -> bool:
    try:
        import beat_this  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def detect_beat_this(audio_path: Path) -> tuple[list[float], list[float]]:
    beats, downbeats = _beat_this_model()(str(audio_path))
    return [float(b) for b in beats], [float(d) for d in downbeats]


def detect_librosa(y: np.ndarray, sr: int) -> tuple[list[float], list[float]]:
    """Temps au pas de 128 échantillons ; premier temps de mesure = phase où la grosse caisse frappe le plus."""
    import librosa

    hop = 128
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop, aggregate=np.median)
    _, frames = librosa.beat.beat_track(onset_envelope=onset, sr=sr, hop_length=hop, units="frames", trim=False)
    beats = librosa.frames_to_time(frames, sr=sr, hop_length=hop)
    if len(beats) < 8:
        return beats.tolist(), []
    low = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop, fmax=200, n_mels=8)
    strength = np.asarray([low[min(len(low) - 1, int(f))] for f in frames])
    phase = int(np.argmax([strength[p::BEATS_PER_BAR].mean() for p in range(BEATS_PER_BAR)]))
    return beats.tolist(), beats[phase::BEATS_PER_BAR].tolist()


def compute_beat_grid(audio_path: Path, y: np.ndarray, sr: int) -> dict[str, Any] | None:
    """Grille rythmique complète d'un morceau, ou None si les temps sont introuvables."""
    choice = os.environ.get("AUDIO_BEAT_ENGINE", "auto").strip().lower()
    use_beat_this = choice == "beat_this" or (choice == "auto" and beat_this_available())
    if use_beat_this:
        beats, downbeats = detect_beat_this(audio_path)
        engine = "beat_this"
    else:
        beats, downbeats = detect_librosa(y, sr)
        engine = "librosa"
    sound_start, sound_end = sound_bounds(y, sr)
    head = local_beats(beats, downbeats, max(0.0, sound_start - 0.1), sound_start + HEAD_SEC)
    tail = local_beats(beats, downbeats, max(0.0, sound_end - TAIL_SEC), sound_end + 0.1)
    if not head and not tail:
        return None
    spans = [g for g in (head, tail) if g]
    total = sum(sum(g["d"]) for g in spans)
    count = sum(len(g["d"]) for g in spans)
    first = head or tail
    bpm = round(60000.0 / (total / count), 3)
    return {
        "engine": engine,
        "bpm": bpm,
        "beat0Sec": first["t0"],
        "downbeatSec": round(first["t0"] + sum(first["d"][: first["phase"]]) / 1000, 3),
        "beatsPerBar": BEATS_PER_BAR,
        "downbeatAgree": first["dbAgree"],
        "soundStartSec": sound_start,
        "soundEndSec": sound_end,
        "head": head,
        "tail": tail,
        "edges": bar_energies(y, sr, bpm, sound_start, sound_end),
    }
