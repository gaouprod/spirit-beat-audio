"""Prototype de remastering « gros studio électro » (workflow « Remastering (prototype) »).

Pour chaque morceau :
  1. ANALYSE  : volume (LUFS), crêtes vraies, écrêtage, dynamique, équilibre
                spectral comparé à une courbe électro de référence, dureté,
                sifflantes, artefacts aigus typiques de Suno (« scintillement »),
                coupure haute, stéréo (largeur, basses hors mono, corrélation).
  2. PLAN     : recette propre au morceau (JSON lisible, chaque réglage motivé).
  3. RENDU    : séparation en pistes (Demucs, facultatif) traitées chacune,
                puis bus master : désécrêtage, égalisation, mono des basses,
                élargissement, compression de colle, saturation douce,
                limiteur vers le volume cible.
  4. ÉCOUTE   : extraits avant / après / après à volume égal (sinon le plus
                fort paraît toujours meilleur) + rapport.

Rien n'est écrit en base ni sur le stockage : tout sort dans le dossier OUT_DIR.
Entrées : TRACKS_FILE (JSON [{id, title, url}] écrit par fetch_tracks.py) ou
TRACKS (JSON, usage local), OUT_DIR, STEMS (1/0), TARGET_LUFS.
"""
import json
import math
import os
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from pedalboard import (
    Compressor,
    Gain,
    HighpassFilter,
    HighShelfFilter,
    Limiter,
    LowShelfFilter,
    PeakFilter,
    Pedalboard,
    Reverb,
)
from scipy import signal

SR = 44100
# Courbe électro de référence : niveau relatif (dB) de la densité spectrale par
# bande, calé sur la bande médium. Pente d'environ -4,5 dB/octave au-dessus de
# 150 Hz, basses portées (+3 dB), typique d'un master club moderne.
BANDS = [
    ("sub", 25, 60),
    ("basse", 60, 150),
    ("bas_medium", 150, 400),
    ("medium", 400, 2000),
    ("presence", 2000, 5000),
    ("sifflantes", 5000, 9000),
    ("air", 9000, 14000),
    ("ultra", 14000, 19000),
]
TARGET_DB = {"sub": 13.0, "basse": 12.0, "bas_medium": 6.0, "medium": 0.0, "presence": -7.5, "sifflantes": -12.5, "air": -16.5, "ultra": -22.0}
BAND_EQ = {  # bande -> (type, fréquence, Q)
    "sub": ("lowshelf", 50, 0.7),
    "basse": ("peak", 100, 0.9),
    "bas_medium": ("peak", 280, 1.0),
    "presence": ("peak", 3200, 1.0),
    "sifflantes": ("peak", 6800, 1.2),
    "air": ("highshelf", 11000, 0.7),
}


# Seuil de scintillement (dB de saut moyen par trame). Valeur de départ, à caler
# sur les vrais morceaux : le rapport donne la mesure de chacun.
SHIMMER_THRESHOLD = 2.5


def db(x):
    return 20 * math.log10(max(float(x), 1e-12))


# ---------------------------------------------------------------- chargement
def is_blob_url(url):
    p = urllib.parse.urlparse(url or "")
    return p.scheme == "https" and bool(p.hostname) and p.hostname.endswith(".blob.vercel-storage.com")


def decode(path):
    """Décodage ffmpeg -> float32 stéréo 44,1 kHz."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "2", "-ar", str(SR), "-"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).copy()


def encode_mp3(audio, path, bitrate="320k"):
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        sf.write(tmp.name, np.clip(audio, -1, 1), SR, subtype="FLOAT")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp.name, "-b:a", bitrate, str(path)], check=True)


# ---------------------------------------------------------------- mesures
def true_peak_db(x):
    up = signal.resample_poly(x, 4, 1, axis=0)
    return db(np.max(np.abs(up)))


def lufs(x):
    return pyln.Meter(SR).integrated_loudness(x)


def short_term(x, win=3.0, hop=1.0):
    meter = pyln.Meter(SR, block_size=0.4)
    n, h = int(win * SR), int(hop * SR)
    out = []
    for start in range(0, max(len(x) - n, 1), h):
        seg = x[start:start + n]
        if len(seg) < n:
            break
        v = meter.integrated_loudness(seg)
        out.append(v if np.isfinite(v) else -70.0)
    return np.array(out) if out else np.array([-70.0])


def clipped_runs(x, thr=0.995, min_run=3):
    """Échantillons écrêtés : suites d'au moins `min_run` valeurs au plafond."""
    hits = np.abs(x) >= thr
    total = 0
    for ch in range(x.shape[1]):
        h = hits[:, ch].astype(np.int8)
        edges = np.diff(np.concatenate([[0], h, [0]]))
        starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
        total += int(np.sum((ends - starts)[(ends - starts) >= min_run]))
    return total / x.size


def band_levels(mono):
    f, p = signal.welch(mono, SR, nperseg=8192)
    levels = {}
    for name, lo, hi in BANDS:
        sel = (f >= lo) & (f < hi)
        levels[name] = 10 * math.log10(max(np.mean(p[sel]), 1e-20))
    ref = levels["medium"]
    rel = {k: round(v - ref, 1) for k, v in levels.items()}
    # Coupure haute : dernière fréquence à moins de 35 dB du niveau médium.
    pd = 10 * np.log10(np.maximum(p, 1e-20)) - ref
    above = f[(pd > -35 - 0) & (f > 8000)]
    cutoff = float(above.max()) if above.size else 8000.0
    return rel, round(cutoff)


def shimmer_index(mono):
    """Artefacts aigus de Suno : au-dessus de 12 kHz, l'énergie « papillonne »
    d'une trame à l'autre. On suit la part des aigus RELATIVEMENT au haut-médium
    (2-8 kHz) : une compression ou un limiteur, qui bougent les deux ensemble,
    ne faussent pas la mesure. Valeur moyenne des sauts, en dB par trame (~12 ms)."""
    f, _, z = signal.stft(mono, SR, nperseg=2048, noverlap=1536)
    p = np.abs(z) ** 2
    hf = p[(f >= 12000) & (f < 17000)].sum(axis=0) + 1e-12
    ref = p[(f >= 2000) & (f < 8000)].sum(axis=0) + 1e-12
    active = ref > np.percentile(ref, 20)  # ignore les silences
    ratio_db = 10 * np.log10(hf / ref)[active]
    return round(float(np.mean(np.abs(np.diff(ratio_db)))), 2) if ratio_db.size > 2 else 0.0


def stereo_stats(x):
    mid, side = (x[:, 0] + x[:, 1]) / 2, (x[:, 0] - x[:, 1]) / 2

    def band(sig, lo, hi):
        sos = signal.butter(4, [lo, hi], btype="band", fs=SR, output="sos")
        return float(np.sqrt(np.mean(signal.sosfilt(sos, sig) ** 2)) + 1e-12)

    corr = float(np.corrcoef(x[:, 0], x[:, 1])[0, 1]) if np.std(x[:, 1]) > 0 else 1.0
    return {
        "largeur": round(band(side, 300, 8000) / band(mid, 300, 8000), 3),
        "basses_hors_mono_db": round(db(band(side, 25, 120) / band(mid, 25, 120)), 1),
        "correlation": round(corr, 3),
    }


def sibilance_db(x):
    mono = x.mean(axis=1)
    sos_s = signal.butter(4, [5000, 9000], btype="band", fs=SR, output="sos")
    sos_v = signal.butter(4, [1000, 4000], btype="band", fs=SR, output="sos")
    s, v = signal.sosfilt(sos_s, mono), signal.sosfilt(sos_v, mono)
    hop = int(0.02 * SR)
    es = np.sqrt(np.convolve(s ** 2, np.ones(hop) / hop, "valid")[::hop])
    ev = np.sqrt(np.convolve(v ** 2, np.ones(hop) / hop, "valid")[::hop])
    loud = ev > np.percentile(ev, 60)
    if not np.any(loud):
        return 0.0
    # Pics de sifflantes (95e centile) par rapport à la voix.
    return round(db(np.percentile(es[loud], 95) / (np.median(ev[loud]) + 1e-9)), 1)


def analyze(x, vocals=None):
    mono = x.mean(axis=1)
    st = short_term(x)
    rel, cutoff = band_levels(mono)
    peak = db(np.max(np.abs(x)))
    rms = db(np.sqrt(np.mean(mono ** 2)))
    return {
        "lufs": round(lufs(x), 1),
        "crete_vraie_dbtp": round(true_peak_db(x), 2),
        "facteur_crete_db": round(peak - rms, 1),
        "plage_dynamique_lu": round(float(np.percentile(st, 95) - np.percentile(st, 10)), 1),
        "ecretage_pct": round(100 * clipped_runs(x), 4),
        "bandes_db": rel,
        "ecart_reference_db": {k: round(rel[k] - TARGET_DB[k], 1) for k in rel},
        "coupure_haute_hz": cutoff,
        "scintillement": shimmer_index(mono),
        "sifflantes_db": sibilance_db(vocals if vocals is not None else x),
        **stereo_stats(x),
    }


# ---------------------------------------------------------------- plan
def make_plan(m, target_lufs=-9.0, stems=False):
    """Recette propre au morceau. Chaque décision porte sa raison."""
    plan = {"cible_lufs": target_lufs, "crete_max_dbtp": -1.0, "eq": [], "pistes": {}, "raisons": []}
    why = plan["raisons"].append

    if m["ecretage_pct"] > 0.001:
        plan["desecretage"] = True
        why(f"Écrêtage numérique ({m['ecretage_pct']} % d'échantillons au plafond) : reconstruction des crêtes avant tout traitement.")

    # Égalisation vers la courbe de référence : on corrige 60 % de l'écart, ±4 dB max.
    for band, (kind, freq, q) in BAND_EQ.items():
        gap = m["ecart_reference_db"][band]
        if abs(gap) < 1.5:
            continue
        gain = float(np.clip(-0.6 * gap, -4.0, 4.0))
        # Ne jamais pousser les aigus d'un morceau qui scintille : on amplifierait l'artefact.
        if band in ("air", "sifflantes") and gain > 0 and m["scintillement"] > SHIMMER_THRESHOLD:
            why(f"Aigus ({band}) en retrait de {-gap} dB mais non relevés : ils portent les artefacts de Suno.")
            continue
        plan["eq"].append({"type": kind, "freq": freq, "q": q, "gain_db": round(gain, 1)})
        why(f"{band} {'+' if gap > 0 else ''}{gap} dB par rapport à la référence électro → {gain:+.1f} dB à {freq} Hz.")

    if m["scintillement"] > SHIMMER_THRESHOLD:
        strong = m["scintillement"] > 2 * SHIMMER_THRESHOLD
        plan["lissage_aigus"] = 0.85 if strong else 0.6
        cut = -2.0 if strong else -1.0
        plan["eq"].append({"type": "highshelf", "freq": 13500, "q": 0.7, "gain_db": cut})
        why(f"Scintillement aigu {m['scintillement']} (> {SHIMMER_THRESHOLD} = artefact de génération) → enveloppe au-dessus de 12 kHz lissée "
            f"(pics atténués jusqu'à {int(plan['lissage_aigus'] * 100)} %) et plateau {cut} dB au-dessus de 13,5 kHz.")

    if m["basses_hors_mono_db"] > -18:
        plan["mono_basses_hz"] = 120
        why(f"Basses en partie stéréo ({m['basses_hors_mono_db']} dB) → mono sous 120 Hz (impact club, compatibilité).")

    if m["largeur"] < 0.22:
        plan["elargissement_db"] = 2.5 if m["largeur"] < 0.12 else 1.5
        why(f"Image étroite (largeur {m['largeur']}) → canal latéral +{plan['elargissement_db']} dB entre 300 Hz et 8 kHz.")

    # Compression de colle : seulement s'il reste de la dynamique (Suno écrase souvent déjà tout).
    if m["facteur_crete_db"] > 11:
        plan["colle"] = {"ratio": 2.0, "seuil_db": -14.0, "attaque_ms": 30, "relache_ms": 150}
        why(f"Facteur de crête {m['facteur_crete_db']} dB : compression de colle 2:1.")
    else:
        why(f"Facteur de crête {m['facteur_crete_db']} dB : déjà compressé, pas de colle supplémentaire.")

    # La saturation crée des harmoniques aiguës : jamais sur un morceau déjà écrêté ou qui scintille.
    plan["saturation"] = 0.0 if (m["ecretage_pct"] > 0.001 or m["scintillement"] > SHIMMER_THRESHOLD) else 0.15
    if plan["saturation"]:
        why("Saturation douce légère (densité analogique) sur le bus.")

    if stems:
        voc = {"compression": {"ratio": 3.0, "seuil_db": -20.0}, "reverb": 0.07}
        if m["sifflantes_db"] > -6:
            voc["de_ess_db"] = -3.0 if m["sifflantes_db"] < -2 else -5.0
            why(f"Sifflantes fortes ({m['sifflantes_db']} dB) → atténuation {voc['de_ess_db']} dB vers 6,8 kHz sur la voix.")
        voc["plateau_aigu_db"] = -1.5
        why("Voix : compression 3:1, aigus adoucis (timbre métallique), petite réverbération pour la placer dans l'espace.")
        plan["pistes"] = {
            "vocals": voc,
            "drums": {"compression": {"ratio": 3.0, "seuil_db": -18.0, "attaque_ms": 15, "relache_ms": 120}, "gain_db": 0.8},
            "bass": {"mono": True, "compression": {"ratio": 2.5, "seuil_db": -16.0, "attaque_ms": 20, "relache_ms": 200}},
            "other": {"elargissement_db": 2.0},
        }
        why("Batterie : attaque lente 15 ms pour réveiller les transitoires ; basse mono et tenue ; nappes élargies.")

    plan["volume"] = (
        f"Volume {m['lufs']} LUFS → {target_lufs} LUFS (standard électro), crête vraie ≤ -1 dBTP."
    )
    return plan


def correction_moves(after, plan):
    moves = []
    for band, (kind, freq, q) in BAND_EQ.items():
        gap = after["ecart_reference_db"][band]
        if abs(gap) <= 2.0:
            continue
        gain = float(np.clip(-0.7 * gap, -3.0, 3.0))
        if band in ("air", "sifflantes") and gain > 0 and after["scintillement"] > SHIMMER_THRESHOLD:
            continue
        moves.append({"type": kind, "freq": freq, "q": q, "gain_db": round(gain, 1)})
        plan["raisons"].append(f"Après mastering, {band} {gap:+} dB par rapport à la référence → correction {gain:+.1f} dB à {freq} Hz.")
    return moves


# ---------------------------------------------------------------- traitements
def declip(x, thr=0.995):
    """Reconstruit les plateaux écrêtés par interpolation cubique (contexte ±8 échantillons)."""
    from scipy.interpolate import CubicSpline

    y = x.copy()
    for ch in range(y.shape[1]):
        s = y[:, ch]
        hit = np.abs(s) >= thr
        edges = np.diff(np.concatenate([[0], hit.astype(np.int8), [0]]))
        for a, b in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
            if b - a < 2 or a < 8 or b + 8 > len(s):
                continue
            idx = np.r_[a - 8:a, b:b + 8]
            cs = CubicSpline(idx, s[idx])
            s[a:b] = cs(np.arange(a, b))
    # Les crêtes reconstruites dépassent 0 dBFS : on redescend avant la suite.
    return y / max(1.0, np.max(np.abs(y)) / 0.95)


def ms(x):
    return (x[:, 0] + x[:, 1]) / 2, (x[:, 0] - x[:, 1]) / 2


def from_ms(mid, side):
    return np.stack([mid + side, mid - side], axis=1)


def mono_below(x, hz):
    mid, side = ms(x)
    sos = signal.butter(4, hz, btype="highpass", fs=SR, output="sos")
    return from_ms(mid, signal.sosfiltfilt(sos, side))


def widen(x, gain_db, lo=300, hi=8000):
    mid, side = ms(x)
    sos = signal.butter(2, [lo, hi], btype="band", fs=SR, output="sos")
    boost = signal.sosfiltfilt(sos, side) * (10 ** (gain_db / 20) - 1)
    return from_ms(mid, side + boost)


def eq_board(moves):
    fx = []
    for m in moves:
        g = m["gain_db"]
        if m["type"] == "lowshelf":
            fx.append(LowShelfFilter(cutoff_frequency_hz=m["freq"], gain_db=g, q=m["q"]))
        elif m["type"] == "highshelf":
            fx.append(HighShelfFilter(cutoff_frequency_hz=m["freq"], gain_db=g, q=m["q"]))
        else:
            fx.append(PeakFilter(cutoff_frequency_hz=m["freq"], gain_db=g, q=m["q"]))
    return fx


def comp(c):
    return Compressor(threshold_db=c["seuil_db"], ratio=c["ratio"], attack_ms=c.get("attaque_ms", 10), release_ms=c.get("relache_ms", 100))


def run(board, x):
    return board(x.T.astype(np.float32), SR).T


def process_stem(name, x, cfg):
    fx = []
    if "compression" in cfg:
        fx.append(comp(cfg["compression"]))
    if cfg.get("de_ess_db"):
        fx.append(PeakFilter(cutoff_frequency_hz=6800, gain_db=cfg["de_ess_db"], q=1.5))
    if cfg.get("plateau_aigu_db"):
        fx.append(HighShelfFilter(cutoff_frequency_hz=11000, gain_db=cfg["plateau_aigu_db"], q=0.7))
    if cfg.get("reverb"):
        fx.append(Reverb(room_size=0.35, damping=0.6, wet_level=cfg["reverb"], dry_level=1.0, width=1.0))
    if cfg.get("gain_db"):
        fx.append(Gain(gain_db=cfg["gain_db"]))
    y = run(Pedalboard(fx), x) if fx else x
    if cfg.get("mono"):
        y = mono_below(y, 20000)
    if cfg.get("elargissement_db"):
        y = widen(y, cfg["elargissement_db"])
    return y


def smooth_shimmer(x, depth):
    """Artefact de génération : au-dessus de 12 kHz, l'énergie papillonne de trame
    en trame SANS que le reste du spectre bouge. On suit la part des aigus
    relativement au haut-médium (2-8 kHz), on la lisse (médiane ~90 ms) et on
    n'atténue que les pics qui dépassent ce lissé. Un charley ou un clap, qui
    montent aigus ET haut-médium ensemble, ne sont pas touchés."""
    from scipy.ndimage import median_filter

    sos_hf = signal.butter(4, 12000, btype="highpass", fs=SR, output="sos")
    sos_ref = signal.butter(4, [2000, 8000], btype="band", fs=SR, output="sos")
    hf = signal.sosfiltfilt(sos_hf, x, axis=0)
    low = x - hf
    ref = signal.sosfiltfilt(sos_ref, x, axis=0)
    hop = 256
    n = len(x) // hop

    def env(sig):
        return np.sqrt(np.mean(sig[: n * hop].reshape(n, hop, 2) ** 2, axis=(1, 2))) + 1e-9

    ratio = env(hf) / env(ref)
    smooth = median_filter(ratio, size=15, mode="nearest")
    gain = np.clip(smooth / ratio, 1 - depth, 1.0)
    gain = np.convolve(gain, np.ones(4) / 4, mode="same")  # pas de saut audible
    g = np.interp(np.arange(len(x)), np.arange(n) * hop + hop / 2, gain)[:, None]
    return low + hf * g


def master(x, plan):
    y = declip(x) if plan.get("desecretage") else x
    y = run(Pedalboard([HighpassFilter(cutoff_frequency_hz=25), *eq_board(plan["eq"])]), y)
    if plan.get("mono_basses_hz"):
        y = mono_below(y, plan["mono_basses_hz"])
    if plan.get("elargissement_db"):
        y = widen(y, plan["elargissement_db"])
    if plan.get("colle"):
        y = run(Pedalboard([comp(plan["colle"])]), y)
    if plan.get("saturation"):
        k = 1 + 4 * plan["saturation"]
        y = np.tanh(k * y) / np.tanh(k)
    y = loudness_to(y, plan["cible_lufs"], plan["crete_max_dbtp"])
    if plan.get("lissage_aigus"):
        # En dernier : n'atténue que des aigus, donc ne peut pas dépasser la crête.
        y = smooth_shimmer(y, plan["lissage_aigus"])
    return y


def loudness_to(y, target, ceiling):
    """Gain + limiteur, ajustés jusqu'au volume cible sans dépasser la crête vraie."""
    # Le limiteur de pedalboard (JUCE) ramène son seuil à 0 dBFS : on le pilote
    # par le gain d'entrée, puis on redescend sous le plafond (marge inter-échantillons).
    gain = target - lufs(y)
    out = y
    for _ in range(10):
        out = run(Pedalboard([Gain(gain_db=gain), Limiter(threshold_db=0.0, release_ms=80), Gain(gain_db=ceiling - 0.6)]), y)
        tp = true_peak_db(out)
        if tp > ceiling:
            out = out * 10 ** ((ceiling - tp) / 20)
        err = target - lufs(out)
        if abs(err) < 0.3:
            break
        gain += err
    return out


def separate(path, workdir):
    subprocess.run(
        [sys.executable, "-m", "demucs", "-n", "htdemucs", "-o", str(workdir), str(path)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    base = Path(workdir) / "htdemucs" / Path(path).stem
    return {name: decode(base / f"{name}.wav") for name in ("vocals", "drums", "bass", "other")}


def loudest_excerpt(x, seconds=40):
    st = short_term(x, win=float(seconds), hop=2.0)
    start = int(np.argmax(st) * 2.0 * SR)
    return slice(start, start + int(seconds * SR))


# ---------------------------------------------------------------- principal
def remaster_track(track, out_dir, use_stems, target_lufs):
    slug = track["id"]
    tdir = out_dir / slug
    tdir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = tmp / f"{slug}.mp3"
        if track.get("path"):
            src = Path(track["path"])
        else:
            if not is_blob_url(track.get("url")):
                raise ValueError("adresse audio refusée (Vercel Blob uniquement)")
            urllib.request.urlretrieve(track["url"], src)
        x = decode(src)
        stems = separate(src, tmp / "sep") if use_stems else None
        before = analyze(x, stems["vocals"] if stems else None)
        plan = make_plan(before, target_lufs, stems=bool(stems))

        if stems:
            n = min(len(s) for s in stems.values())
            mix = sum(process_stem(k, v[:n], plan["pistes"][k]) for k, v in stems.items())
            mix = mix / max(1.0, np.max(np.abs(mix)) / 0.95)
        else:
            mix = x
        y = master(mix, plan)
        after = analyze(y)
        # Deuxième passe en boucle fermée : la compression et le limiteur déplacent
        # l'équilibre (mesuré : les basses perdaient 3 à 5 dB). On remesure et on
        # recorrige les bandes qui ont dérivé de plus de 2 dB, puis on refait le volume.
        fix = correction_moves(after, plan)
        if fix:
            plan["correction"] = fix
            y = run(Pedalboard(eq_board(fix)), y)
            y = loudness_to(y, plan["cible_lufs"], plan["crete_max_dbtp"])
            after = analyze(y)

        # Extraits sur le passage le plus fort de l'original ; la version « à volume
        # égal » ramène le remaster au volume de l'original pour une écoute honnête.
        cut = loudest_excerpt(x)
        matched = y * 10 ** ((before["lufs"] - after["lufs"]) / 20)
        encode_mp3(x[cut], tdir / "1-original.mp3")
        encode_mp3(matched[cut], tdir / "2-remaster-meme-volume.mp3")
        encode_mp3(y[cut], tdir / "3-remaster.mp3")
        encode_mp3(y, tdir / "remaster-complet.mp3")
        report = {"id": slug, "title": track.get("title"), "avant": before, "apres": after, "plan": plan}
        (tdir / "recette.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report


def markdown(reports):
    lines = ["# Remastering — prototype", "",
             "Écouter dans l'ordre : `1-original`, `2-remaster-meme-volume` (comparaison honnête), `3-remaster` (volume final).", ""]
    keys = [("lufs", "Volume (LUFS)"), ("crete_vraie_dbtp", "Crête vraie (dBTP)"), ("facteur_crete_db", "Facteur de crête (dB)"),
            ("plage_dynamique_lu", "Plage dynamique (LU)"), ("ecretage_pct", "Écrêtage (%)"), ("scintillement", "Scintillement aigu"),
            ("coupure_haute_hz", "Coupure haute (Hz)"), ("largeur", "Largeur stéréo"), ("basses_hors_mono_db", "Basses hors mono (dB)"),
            ("sifflantes_db", "Sifflantes (dB)")]
    for r in reports:
        lines += [f"## {r.get('title') or r['id']}", ""]
        if r.get("error"):
            lines += [f"Échec : {r['error']}", ""]
            continue
        lines += ["| Mesure | Avant | Après |", "|---|---|---|"]
        lines += [f"| {label} | {r['avant'][k]} | {r['apres'][k]} |" for k, label in keys]
        lines += ["", "**Écart à la référence électro (dB, avant → après)** : " + ", ".join(
            f"{b} {r['avant']['ecart_reference_db'][b]:+} → {r['apres']['ecart_reference_db'][b]:+}" for b, *_ in BANDS), ""]
        lines += ["**Recette appliquée :**", ""] + [f"- {w}" for w in r["plan"]["raisons"]] + [f"- {r['plan']['volume']}", ""]
    return "\n".join(lines)


def main():
    tracks_file = os.environ.get("TRACKS_FILE")
    tracks = json.loads(Path(tracks_file).read_text(encoding="utf-8") if tracks_file else os.environ["TRACKS"])
    out_dir = Path(os.environ.get("OUT_DIR", "remaster-out"))
    use_stems = os.environ.get("STEMS", "1") == "1"
    target = float(os.environ.get("TARGET_LUFS", "-9"))
    reports = []
    for t in tracks:
        print(f"- {t.get('title')}", flush=True)
        try:
            reports.append(remaster_track(t, out_dir, use_stems, target))
        except Exception as error:  # un morceau en échec n'arrête pas les autres
            reports.append({"id": t.get("id"), "title": t.get("title"), "error": str(error)[:300]})
            print(f"  échec : {error}", flush=True)
    (out_dir / "RAPPORT.md").write_text(markdown(reports), encoding="utf-8")
    # Le rapport va dans la Release privée de SPIRIT-BEAT, pas dans ce journal public.


if __name__ == "__main__":
    main()
