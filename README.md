# spirit-beat-audio

Traitements audio lourds de SPIRIT-BEAT (transcription des paroles,
remastering), exécutés sur GitHub Actions. Ce dépôt est **public** pour que
les minutes Actions soient gratuites : il ne contient **que des scripts**, ni
musique, ni paroles, ni code de l'application.

## Ce qui est public, et ce qui ne l'est jamais

| Public (journaux, entrées) | Jamais public |
|---|---|
| identifiants de morceaux, compteurs, durées, réussite/échec | adresses des fichiers audio (masquées), paroles, audio remasterisé, rapports |

- Les workflows ne se lancent qu'à la main ou par SPIRIT-BEAT (`workflow_dispatch`,
  réservé aux personnes ayant les droits d'écriture) : jamais sur une PR,
  donc les secrets ne sont jamais exposés à un fork.
- Les adresses audio sont demandées à SPIRIT-BEAT (`GET /api/pipeline/tracks`)
  avec `PIPELINE_SECRET` et masquées dans le journal.
- Les paroles repartent directement à SPIRIT-BEAT (`POST /api/pipeline/lyrics`).
- Le remastering est publié dans la Release **privée** du dépôt SPIRIT-BEAT.

## Workflows

- `transcribe.yml` — lancé automatiquement par SPIRIT-BEAT après chaque analyse.
- `remaster.yml` — prototype de remastering propre à chaque morceau.
- `overlaps.yml` — repères **Overlap IN / Overlap OUT** de LiveRégie (premier kick,
  temps qui suit le dernier groove), lancé par SPIRIT-BEAT après l'analyse de chaque
  morceau qui n'en a pas ; résultat renvoyé à `POST /api/pipeline/overlaps`, qui
  n'écrase jamais un repère déjà réglé. Méthode et banc de mesure : `overlaps/detect.py`.

## Réglages (Settings → Secrets and variables → Actions)

- Secret `PIPELINE_SECRET` : même valeur que la variable `PIPELINE_SECRET` de SPIRIT-BEAT sur Vercel.
- Secret `RELEASE_TOKEN` : jeton GitHub à droit « Contents : écriture » sur `gaouprod/SPIRIT-BEAT` seulement (remastering).
- Variable `SPIRIT_BEAT_URL` (facultative, défaut `https://spirit-beat.vercel.app`).
