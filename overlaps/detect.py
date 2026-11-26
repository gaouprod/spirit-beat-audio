"""Détection des repères Overlap IN / Overlap OUT d'un morceau (LiveRégie).

LiveRégie enchaîne A → B par un fondu de durée `overlap_exit(A) + overlap_enter(B)`
qui fait tomber le dernier temps de A sur le premier temps de B. Il faut donc :

- Overlap IN  = instant du premier temps franc (premier kick) de B, depuis le début ;
- Overlap OUT = durée entre le « temps d'après » du dernier temps de groove de A
  (là où le groove continuerait, ou le coup final) et la fin du fichier.

Méthode (numpy + librosa pour le décodage et le spectre seulement) :
1. Attaques par bande : flux spectral (mel, dB) sous 150 Hz (kick, basse) et pleine bande.
2. Tempo global (autocorrélation, préférence 110-152 BPM), puis grilles LOCALES
   au début et à la fin : les morceaux Suno dérivent de 1 à 2 % entre le début
   et la fin, une grille unique décale la fin d'un demi-temps.
3. Force de chaque temps comparée au groove du morceau ; seuils en cascade pour
   les intros et outros calmes ; calage final sur l'attaque réelle (± 30 ms).

Banc de mesure (2026-11-26, 91 morceaux, juge indépendant = grille Beat This) :
repères à moins d'1/8 de temps d'un vrai temps — IN 89 %, OUT 86 %, contre
71 % / 62 % pour les repères posés en avril.
"""
from __future__ import annotations

import numpy as np

SR = 22050
HOP = 128
FR = SR / HOP  # trames par seconde
W = int(round(0.03 * FR))  # tolérance de calage autour d'un temps (± 30 ms)

# Bornes de sécurité : au-delà, le fondu LiveRégie serait déraisonnable.
MAX_ENTER_SEC = 8.0
MAX_EXIT_SEC = 8.0
MIN_EXIT_SEC = 0.2
VERSION = "overlaps-v1"


def features(y: np.ndarray) -> dict:
    import librosa

    spec = np.abs(librosa.stft(y, n_fft=1024, hop_length=HOP)).astype(np.float32) ** 2
    mel = librosa.feature.melspectrogram(S=spec, sr=SR, n_mels=64, fmax=11000)
    lm = librosa.power_to_db(mel)
    flux = np.maximum(0, np.diff(lm, axis=1, prepend=lm[:, :1]))
    mf = librosa.mel_frequencies(64, fmax=11000)
    return {
        "full": flux.mean(0),
        "low": flux[mf < 150].mean(0),
        "dur": len(y) / SR,
    }


def fit_grid(env, bpm_lo, bpm_hi, pref=None):
    """Période (trames) et phase d'une grille à tempo fixe sur `env`."""
    n = len(env)
    x = env - env.mean()
    ac = np.correlate(x, x, "full")[n - 1:]
    lags = np.arange(int(FR * 60 / bpm_hi), int(FR * 60 / bpm_lo) + 2)
    lags = lags[(lags > 0) & (lags < n // 2)]
    if not len(lags):
        return FR * 60 / ((bpm_lo + bpm_hi) / 2), 0.0
    sc = ac[lags].astype(float)
    if pref:
        b = 60 * FR / lags
        sc *= 1 + 0.35 * ((b >= pref[0]) & (b <= pref[1]))
    p0 = lags[int(np.argmax(sc))]
    best = (-1.0, float(p0), 0.0)
    pmin, pmax = 60 * FR / bpm_hi, 60 * FR / bpm_lo
    for p in np.arange(p0 - 1.5, p0 + 1.5, 0.02):
        if not pmin <= p <= pmax:
            continue
        phases = np.arange(0, p, 0.5)
        k = np.arange(0, int(n / p) + 1)
        idx = np.round(phases[:, None] + k[None, :] * p).astype(int)
        valid = idx < n
        scores = np.where(valid, env[np.minimum(idx, n - 1)], 0).sum(1) / np.maximum(valid.sum(1), 1)
        i = int(np.argmax(scores))
        if scores[i] > best[0]:
            best = (float(scores[i]), float(p), float(phases[i]))
    return best[1], best[2]


def grid_at(env, a, b, bpm, tol=0.04):
    a = max(0, int(a))
    b = min(len(env), int(b))
    p, ph = fit_grid(env[a:b], bpm * (1 - tol), bpm * (1 + tol))
    return p, a + ph


def closest(bt, b):
    """Temps de la grille `bt` le plus proche de `b` (ou `b` si la grille est vide)."""
    return bt[int(np.argmin(np.abs(bt - b)))] if len(bt) else b


def beats(g, a, b):
    p, ph = g
    k0, k1 = np.ceil((a - ph) / p), np.floor((b - ph) / p)
    return ph + np.arange(k0, k1 + 1) * p


def at(x, bt):
    return np.array([x[max(0, int(round(b)) - W):int(round(b)) + W + 1].max() if 0 <= round(b) < len(x) else 0.0 for b in bt])


def attack(x, b):
    i = min(max(int(round(b)), 0), len(x) - 1)
    a = max(0, i - W)
    seg = x[a:i + W + 1]
    return (a + int(np.argmax(seg))) / FR if seg.size else b / FR


def detect(f: dict) -> dict:
    full, low, dur = f["full"], f["low"], f["dur"]
    n = len(full)
    both = low + full
    env_in = np.convolve(low + full, np.ones(3) / 3, "same")
    env_out = np.convolve(3 * low + full, np.ones(3) / 3, "same")

    gp, _ = fit_grid(env_in[int(0.1 * n):int(0.9 * n)], 95, 175, (110, 152))
    bpm = 60 * FR / gp

    # ---- Overlap IN : premier temps avec un vrai kick (puis seuils plus doux)
    g = grid_at(env_in, 0, 45 * FR, bpm)
    bt = beats(g, 0, min(n - 1, 45 * FR))
    kl, kf = at(low, bt), at(full, bt)
    rl = np.percentile(kl[len(kl) // 2:], 60) if len(kl) else 0
    rf = np.percentile(kf[len(kf) // 2:], 60) if len(kf) else 0
    j = None
    for a_low, a_full in ((0.5, 0.1), (0.25, 0.05), (0.0, 0.3)):
        ok = np.flatnonzero((kl >= a_low * rl) & (kf >= a_full * rf) & (bt / FR <= MAX_ENTER_SEC))
        if len(ok):
            j = int(ok[0])
            break
    if j is None:
        enter, enter_rule = 0.0, "aucune attaque dans les 8 premières secondes"
    else:
        g2 = grid_at(env_in, bt[j] - 2 * g[0], bt[j] + 35 * FR, bpm)
        near = beats(g2, bt[j] - g2[0] * 0.6, bt[j] + g2[0] * 0.6)
        enter = attack(both, max(0.0, closest(near, bt[j])))
        enter_rule = "premier kick"

    # ---- Overlap OUT : temps qui suit le dernier temps de groove
    g = grid_at(env_out, n - 50 * FR, n, bpm)
    bt = beats(g, n - 50 * FR, n - 1)
    k = at(full, bt)
    r = np.percentile(k[: max(1, len(k) // 2)], 60) if len(k) else 0
    exit_, exit_rule = None, None
    for a_full in (0.6, 0.35, 0.2):
        ok = np.flatnonzero(k >= a_full * r)
        if not len(ok):
            continue
        j = int(ok[-1])
        g2 = grid_at(env_out, bt[j] - 35 * FR, bt[j] + g[0] * 0.6, bpm)
        near = beats(g2, bt[j] - g2[0] * 0.6, bt[j] + g2[0] * 0.6)
        last = closest(near, bt[j])
        nxt = last + g2[0]
        if nxt < n - W:
            i = int(round(nxt))
            seg = both[max(0, i - W):i + W + 1]
            t = attack(both, nxt) if seg.size and seg.max() >= 0.3 * r else nxt / FR
        else:  # le fichier s'arrête avant : on garde le dernier coup
            t = attack(both, last)
        cand = dur - t
        if cand <= MAX_EXIT_SEC:
            exit_, exit_rule = cand, f"temps après le dernier groove (seuil {a_full})"
            break
    if exit_ is None:
        exit_, exit_rule = MAX_EXIT_SEC, "outro longue : plafond"
    exit_ = max(MIN_EXIT_SEC, exit_)

    return {
        "overlap_enter_seconds": round(float(enter), 2),
        "overlap_exit_seconds": round(float(exit_), 2),
        "bpm": round(float(bpm), 2),
        "duration": round(float(dur), 2),
        "rules": {"enter": enter_rule, "exit": exit_rule},
        "version": VERSION,
    }


def detect_file(path: str) -> dict:
    import librosa

    y, _ = librosa.load(path, sr=SR, mono=True)
    return detect(features(y))
