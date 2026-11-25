"""Transcription des paroles chantées (workflow transcribe.yml de ce dépôt public).

Pour chaque morceau : téléchargement du MP3 (Vercel Blob uniquement), isolation
de la voix (Demucs htdemucs, deux pistes), puis transcription en français
(faster-whisper, CPU int8). Les paroles vont UNIQUEMENT dans RESULTS_FILE,
renvoyé à SPIRIT-BEAT : ce dépôt est public, ses journaux aussi, donc rien
n'y est imprimé d'autre que des compteurs.

Entrées : TRACKS_FILE (JSON [{id, title, url}] écrit par fetch_tracks.py),
SHARD / SHARDS (répartition entre jobs), MODEL (défaut large-v3-turbo).
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

from faster_whisper import WhisperModel

STANZA_GAP_SEC = 3.0
# Formules que Whisper invente sur la musique (génériques de vidéos) : jamais des paroles.
HALLUCINATIONS = ("sous-titrage", "sous-titres", "merci d'avoir regardé", "abonnez-vous", "amara.org", "radio-canada")
# Segments entiers inventés sur les passages instrumentaux (relevés sur 25 morceaux, 2026-11-25).
WHOLE_SEGMENT_HALLUCINATIONS = {"merci", "merci.", "musique", "...", "…"}
# Une phrase de Whisper = une ligne ; trop longue, elle est coupée aux virgules.
MAX_LINE_WORDS = 10


def is_blob_url(url):
    parsed = urllib.parse.urlparse(url)
    return parsed.scheme == "https" and parsed.hostname and parsed.hostname.endswith(".blob.vercel-storage.com")


def separate_vocals(mp3, workdir):
    subprocess.run(
        [sys.executable, "-m", "demucs", "--two-stems", "vocals", "-n", "htdemucs", "-o", str(workdir), str(mp3)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return workdir / "htdemucs" / mp3.stem / "vocals.wav"


def split_long(text):
    words = text.split()
    if len(words) <= MAX_LINE_WORDS:
        return [text]
    parts = [p.strip() for p in text.replace(";", ",").split(",") if p.strip()]
    return parts if len(parts) > 1 else [text]


def to_lyrics(segments):
    """Une ligne par phrase chantée, strophes aux longs silences."""
    lines, last_end = [], None
    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        if last_end is not None and seg["start"] - last_end > STANZA_GAP_SEC and lines and lines[-1] != "":
            lines.append("")
        for part in split_long(text):
            part = part.strip().rstrip(",")
            lines.append(part[:1].upper() + part[1:])
        last_end = seg["end"]
    return "\n".join(lines).strip()


def main():
    tracks = json.loads(Path(os.environ["TRACKS_FILE"]).read_text(encoding="utf-8"))
    shard, shards = int(os.environ.get("SHARD", 0)), int(os.environ.get("SHARDS", 1))
    mine = [t for i, t in enumerate(tracks) if i % shards == shard]
    print(f"Job {shard + 1}/{shards} : {len(mine)} morceaux", flush=True)
    model = WhisperModel(os.environ.get("MODEL", "large-v3-turbo"), device="cpu", compute_type="int8", cpu_threads=os.cpu_count() or 4)

    results = []
    for track in mine:
        started = time.time()
        result = {"id": track["id"], "title": track.get("title")}
        try:
            if not is_blob_url(track.get("url", "")):
                raise ValueError("adresse audio refusée (Vercel Blob uniquement)")
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                mp3 = tmp / f"{track['id']}.mp3"
                urllib.request.urlretrieve(track["url"], mp3)
                vocals = separate_vocals(mp3, tmp / "sep")
                # Pas de titre en amorce : Whisper le recopie au lieu d'écouter.
                # Les passages sans voix (souvent inventés) sont écartés.
                segments, info = model.transcribe(
                    str(vocals),
                    language="fr",
                    beam_size=5,
                    # Pas de détecteur de parole : il rate le chant ; la voix est déjà isolée.
                    vad_filter=False,
                    condition_on_previous_text=False,
                    compression_ratio_threshold=2.2,
                    no_speech_threshold=0.6,
                    word_timestamps=True,
                    hallucination_silence_threshold=2.0,
                )
                raw = list(segments)
                segs = [
                    {
                        "start": round(s.start, 1),
                        "end": round(s.end, 1),
                        "text": s.text.strip(),
                        "logprob": round(s.avg_logprob, 2),
                    }
                    for s in raw
                    if s.avg_logprob > -1.0 and s.no_speech_prob < 0.6
                    and not any(h in s.text.lower() for h in HALLUCINATIONS)
                    and s.text.strip().lower() not in WHOLE_SEGMENT_HALLUCINATIONS
                ]
                result["kept"] = f"{len(segs)}/{len(raw)}"
                result.update({"lyrics": to_lyrics(segs), "segments": segs, "duration": round(info.duration, 1)})
        except Exception as error:  # un morceau en échec n'arrête pas les autres
            result["error"] = type(error).__name__ + ": " + str(error)[:200]
        result["seconds"] = round(time.time() - started, 1)
        results.append(result)
        print(f"- {track['id']} : {len(result.get('segments', []))} segments, {result['seconds']} s", flush=True)

    # Récapitulatif compact en fin de journal (sans les segments) : une ligne par morceau.
    compacts = []
    for result in results:
        compact = {k: v for k, v in result.items() if k != "segments"}
        compact["confidence"] = round(sum(x["logprob"] for x in result.get("segments", [])) / max(len(result.get("segments", [])), 1), 2)
        compacts.append(compact)
        # Journal public : jamais les paroles.
        print(f"  {result['id']} : {'échec' if result.get('error') else 'ok'}, confiance {compact['confidence']}", flush=True)

    # Fichier renvoyé à SPIRIT-BEAT par le workflow (étape « Retour à SPIRIT-BEAT »).
    results_file = os.environ.get("RESULTS_FILE")
    if results_file and compacts:
        Path(results_file).write_text(json.dumps({"results": compacts}, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
