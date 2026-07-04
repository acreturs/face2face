# Leon-Studie: Landmark-Anzahl × Regularisierung (Sparse-Fit, nur Foto)

Erstellt 2026-07-04 (Claude). Alle Läufe: gleiches Foto
(`data/iphone/default/RGB/000000_RGB.png`, ~2300 px Gesicht), gleiche Pipeline
(`./build/face_recon --mode sparse --dataset iphone --sparse-reg <λ>`), nur
Landmark-Set und λ variiert. (Theorie-Kurzfassung: E_reg = λ‖α‖² ist der
Gesichts-Prior; λ muss mit Anzahl UND Größe der Daten-Residuen skalieren,
sonst Overfitting bzw. Kollaps aufs Mittelgesicht — Details im README §Energy.)

## Ergebnisse

Erzeugt mit `--set` (9 vs 25 Punkte) und `--sparse-reg <λ>`:

| Konfiguration | Displacement Ø / max (mm) | Landmark-RMS (px) | Bewertung |
|---|---|---|---|
| 9 LM, λ=100 (Default) | **0.94** / 3.12 | 8.85 | Form ≈ Mittelgesicht — gewollt: 2D kann Identität nicht liefern |
| 25 LM, λ=100 | **11.49** / 20.84 | 23.05 | **Depth-Overfitting**: Gesicht in Z eingedrückt (s. Profil-Plot) |
| 25 LM, λ=500 | 5.68 / 9.96 | 27.10 | Übergang — noch sichtbar verzogen |
| 25 LM, λ=2000 | **2.11** / 3.62 | 30.88 | Prior fängt es ein, Form wieder plausibel |

**Kernaussage:** Auf einem einzelnen Foto binden 2D-Landmarks nur die Pose,
nicht die Tiefe der Identität. Mehr Landmarks bei gleichem λ verkleinern zwar
den 2D-Fehler, aber nur, indem die 3D-Form in die unbeobachtete Tiefe verbogen
wird (Displacement ↓ RMS-Sockel steigt = Lehrbuch-Overfitting). Deshalb: iPhone
= 9 Punkte @ λ=100; die 25 Punkte lohnen erst mit einem Tiefen-Term (Biwi), der
die Identität misst. **RMS nur innerhalb desselben Sets vergleichen** — das
25er-Set hat Näherungspunkte (Lider/Kinn), sein Sockel liegt höher.

## Dateien

- `leon_<config>_wireframe.png` — Mesh-Overlay auf dem Foto; grün = Detektion,
  rot = projizierter Modell-Vertex
- `leon_<config>_render_overlay.png` — gerendertes Gesicht 70 % über dem Foto
- `profile_depth_overfitting.png` — **die Hauptfigur**: Profilansicht der vier
  Fits gegen das Mittelgesicht (grau); frontal sieht man die Verzerrung kaum,
  im Profil sofort
- `fitted_<config>.obj`, `mean_face.obj` — die Meshes für MeshLab/3D-Ansicht

Hinweis: Diese PNGs/OBJs sind gitignored und mit dem 9/25-Set + `--sparse-reg`
reproduzierbar (die früheren `*_contour_*`-Dateien gehören zum inzwischen
entfernten Jawline-Term und sind nur noch lokal, falls vorhanden).

## Vorschlag Bildunterschrift (englisch, für den Bericht)

> Sparse-only fitting to a single frontal photo. With 9 reliable landmarks and
> λ=100 the identity stays near the statistical mean (0.9 mm mean deviation) —
> intended, since depth-blind 2D landmarks cannot observe identity. Adding all
> 25 mappable landmarks at unchanged λ lets the optimiser trade 2D residual
> for out-of-plane shape distortion (11.5 mm, profile view) — classic
> overfitting of an under-constrained inverse problem. Raising the shape
> regulariser restores a plausible face (λ=2000: 2.1 mm) at slightly higher
> 2D residual. On RGB-D data (Biwi) the dense depth term supplies the missing
> constraint instead, and the same 25-landmark set is beneficial for pose.
