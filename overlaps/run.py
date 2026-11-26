"""Détection des repères Overlap IN/OUT (workflow overlaps.yml de ce dépôt public).

Entrée : TRACKS_FILE (JSON [{id, title, url}] écrit par fetch_tracks.py).
Sortie : RESULTS_FILE = {"results": [{id, overlap_enter_seconds, overlap_exit_seconds, bpm, ...}]},
renvoyé à SPIRIT-BEAT (POST /api/pipeline/overlaps), qui n'écrit que sur les
morceaux sans repères. Le journal public ne montre que des identifiants et des
nombres : ni adresse audio, ni titre, ni paroles.
"""
import json
import os
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect import detect_file  # noqa: E402


def is_blob_url(url):
    parsed = urllib.parse.urlparse(url)
    return parsed.scheme == "https" and parsed.hostname and parsed.hostname.endswith(".blob.vercel-storage.com")


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
                result.update(detect_file(str(mp3)))
        except Exception as error:  # un morceau en échec n'arrête pas les autres
            result["error"] = type(error).__name__ + ": " + str(error)[:200]
        results.append(result)
        if result.get("error"):
            print(f"- {track['id']} : échec ({result['error'][:80]})", flush=True)
        else:
            print(
                f"- {track['id']} : IN {result['overlap_enter_seconds']} s, OUT {result['overlap_exit_seconds']} s,"
                f" {result['bpm']} BPM ({round(time.time() - started, 1)} s)",
                flush=True,
            )
    Path(os.environ.get("RESULTS_FILE", "overlaps-results.json")).write_text(
        json.dumps({"results": results}, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
