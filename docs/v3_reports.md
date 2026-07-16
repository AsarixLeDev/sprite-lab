# Sprite Lab v3 reports

Every state-changing command writes a human-readable `index.html` and machine-readable `report.json` beside its run state. Use:

```powershell
python -m spritelab v3 report
python -m spritelab v3 report --open
```

Reports are static and offline: CSS is inline, no CDN is used, no network request is required, and the document remains useful without JavaScript. Browser opening uses the platform's default browser and is mockable in tests.

The project overview shows stage status, blockers, warnings, audit applicability, production authorization, source commit, and last run context. Stage cards show evidence paths and hashes plus every metric available from authoritative artifacts.

Dataset sections can include source disposition, license/provenance counts, extraction and suitability dispositions, label health/agreement, view composition, and leakage identities. Training sections can include campaign/seed progress, curves, checkpoint schedule, and resume identity. Evaluation sections can include metrics, galleries, diversity, palette metrics, memorization classes, review state, and promotion gates.

Missing information is rendered as **No data yet**. The report does not invent a curve, gallery, ETA, audit pass, or production authorization. `report.json` uses the same typed project state as terminal and JSON CLI output, so automation never parses human text.
