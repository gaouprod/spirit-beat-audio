"""Récupère auprès de SPIRIT-BEAT les adresses audio des morceaux demandés.

Ce dépôt est PUBLIC : le workflow ne reçoit que des identifiants (IDS), jamais
une adresse audio. Les adresses sont demandées à SPIRIT-BEAT avec le secret
PIPELINE_SECRET, masquées dans le journal (::add-mask::) avant toute autre
sortie, puis écrites dans OUT_FILE pour les scripts de traitement.
"""
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path


def main():
    ids = [i for i in os.environ.get("IDS", "").replace(" ", "").split(",") if re.fullmatch(r"[A-Za-z0-9_-]{1,128}", i)]
    if not ids:
        sys.exit("Aucun identifiant de morceau valide.")
    secret = os.environ.get("PIPELINE_SECRET", "")
    if len(secret) < 32:
        sys.exit("Secret PIPELINE_SECRET absent ou trop court.")
    base = os.environ.get("SPIRIT_BEAT_URL", "https://spirit-beat.vercel.app").rstrip("/")
    request = urllib.request.Request(
        f"{base}/api/pipeline/tracks?ids={urllib.parse.quote(','.join(ids))}",
        headers={"Authorization": f"Bearer {secret}", "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        tracks = json.load(response)["tracks"]
    for t in tracks:
        print(f"::add-mask::{t['url']}", flush=True)
    missing = sorted(set(ids) - {t["id"] for t in tracks})
    Path(os.environ.get("OUT_FILE", "tracks.json")).write_text(json.dumps(tracks, ensure_ascii=False), encoding="utf-8")
    print(f"{len(tracks)} morceau(x) à traiter" + (f", introuvable(s) : {', '.join(missing)}" if missing else ""), flush=True)


if __name__ == "__main__":
    main()
