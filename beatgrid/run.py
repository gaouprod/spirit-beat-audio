"""Grille rythmique des morceaux (workflow beatgrid.yml de ce dépôt public).

Entrée : TRACKS_FILE (JSON [{id, title, url}] écrit par fetch_tracks.py).
Sortie : RESULTS_FILE = {"results": [{id, beatGrid, fit, windows}]}, renvoyé à
SPIRIT-BEAT (POST /api/pipeline/beatgrid), qui l'enregistre dans TrackBeatGrid
sans jamais remplacer une grille d'un moteur plus précis.

Calcul identique au worker d'analyse de SPIRIT-BEAT (worker_grid.py, copie
conforme) : temps et premiers temps de mesure détectés par beat_this (CPJKU,
repli librosa), temps du début (30 s) et de la fin (45 s), début et fin du son,
profil d'énergie des 16 premières et 16 dernières mesures. En plus, la qualité
d'une grille régulière (`fit`, `windows`), comme le workflow « Grille rythmique »
de SPIRIT-BEAT. Le journal public ne montre que des identifiants et des nombres.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from worker_grid import (  # noqa: E402
    BEATS_PER_BAR,
    HEAD_SEC,
    TAIL_SEC,
    bar_energies,
    detect_librosa,
    local_beats,
    sound_bounds,
)

# Un temps détecté à moins de 25 ms de la grille régulière la confirme.
FIT_TOLERANCE_SEC = 0.025


def is_blob_url(url):
    parsed = urllib.parse.urlparse(url)
    return parsed.scheme == "https" and parsed.hostname and parsed.hostname.endswith(".blob.vercel-storage.com")


def fit_grid(beats):
    """Part des temps détectés à moins de 25 ms d'une grille régulière (moindres carrés robustes)."""
    beats = np.asarray(sorted(beats), dtype=float)
    if len(beats) < 16:
        return None
    period = float(np.median(np.diff(beats)))
    origin = beats[0]
    keep = np.ones(len(beats), dtype=bool)
    for _ in range(4):
        n = np.round((beats - origin) / period)
        A = np.vstack([np.ones(keep.sum()), n[keep]]).T
        (origin, period), *_ = np.linalg.lstsq(A, beats[keep], rcond=None)
        n = np.round((beats - origin) / period)
        keep = np.abs(beats - (origin + n * period)) < max(period / 6, 0.04)
        if keep.sum() < 16:
            return None
    n = np.round((beats - origin) / period)
    return round(float(np.mean(np.abs(beats - (origin + n * period)) < FIT_TOLERANCE_SEC)), 3)


def window_fits(beats, duration, size=30.0):
    """Part des fenêtres de 30 s où une grille régulière locale colle (diagnostic)."""
    good = total = 0
    t = 0.0
    while t + size <= duration:
        fit = fit_grid([b for b in beats if t <= b <= t + size])
        total += 1
        good += 1 if fit is not None and fit >= 0.8 else 0
        t += size
    return round(good / total, 2) if total else 0.0


_MODEL = None


def beat_this_model():
    """Modèle beat_this (téléchargé au premier usage), ou None : repli librosa, journalisé."""
    global _MODEL
    if _MODEL is None:
        try:
            from beat_this.inference import File2Beats

            _MODEL = File2Beats(checkpoint_path="final0", device="cpu", dbn=False)
        except Exception as error:  # noqa: BLE001
            print(f"beat_this indisponible ({type(error).__name__}) : repli librosa", flush=True)
            _MODEL = False
    return _MODEL or None


def analyse(mp3, workdir):
    import librosa

    # Même signal que le worker d'analyse : mono, 22 050 Hz (début et fin du son, profil d'énergie).
    y, sr = librosa.load(str(mp3), sr=22050, mono=True)
    model = beat_this_model()
    if model is not None:
        wav = workdir / "track.wav"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(mp3), "-ac", "1", "-ar", "44100", str(wav)], check=True)
        beats, downbeats = model(str(wav))
        beats, downbeats = [float(b) for b in beats], [float(d) for d in downbeats]
        engine = "beat_this"
    else:
        beats, downbeats = detect_librosa(y, sr)
        engine = "librosa"
    sound_start, sound_end = sound_bounds(y, sr)
    head = local_beats(beats, downbeats, max(0.0, sound_start - 0.1), sound_start + HEAD_SEC)
    tail = local_beats(beats, downbeats, max(0.0, sound_end - TAIL_SEC), sound_end + 0.1)
    if not head and not tail:
        raise ValueError("trop peu de temps détectés")
    spans = [g for g in (head, tail) if g]
    bpm = round(60000.0 / (sum(sum(g["d"]) for g in spans) / sum(len(g["d"]) for g in spans)), 3)
    first = head or tail
    grid = {
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
    return {"beatGrid": grid, "fit": fit_grid(beats), "windows": window_fits(beats, len(y) / sr)}


def main():
    tracks = json.loads(Path(os.environ["TRACKS_FILE"]).read_text(encoding="utf-8"))
    results = []
    for track in tracks:
        started = time.time()
        result = {"id": track["id"]}
        try:
            if not is_blob_url(track.get("url", "")):
                raise ValueError("adresse audio refusée (Vercel Blob uniquement)")
            with tempfile.TemporaryDirectory() as tmp:
                mp3 = Path(tmp) / f"{track['id']}.mp3"
                urllib.request.urlretrieve(track["url"], mp3)
                result.update(analyse(mp3, Path(tmp)))
        except Exception as error:  # un morceau en échec n'arrête pas les autres
            result["error"] = type(error).__name__ + ": " + str(error)[:200]
        results.append(result)
        if result.get("error"):
            print(f"- {track['id']} : échec ({result['error'][:80]})", flush=True)
        else:
            g = result["beatGrid"]
            print(
                f"- {track['id']} : {g['engine']} {g['bpm']} BPM, régularité {(g['head'] or {}).get('regularity')}/{(g['tail'] or {}).get('regularity')},"
                f" bords {'oui' if g['edges'] else 'non'} ({round(time.time() - started, 1)} s)",
                flush=True,
            )
    Path(os.environ.get("RESULTS_FILE", "beatgrid-results.json")).write_text(json.dumps({"results": results}, ensure_ascii=False), encoding="utf-8")
    return 0 if any("beatGrid" in r for r in results) or not results else 1


if __name__ == "__main__":
    sys.exit(main())
