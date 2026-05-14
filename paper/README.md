# Ink2MIDI — IEEE Conference Paper

This directory contains a draft IEEE conference paper describing the
Ink2MIDI pipeline.

## Files

- `ink2midi_ieee.tex` — main LaTeX source, IEEEtran conference template.

## Building

### Option A — Overleaf (easiest)

1. Open <https://www.overleaf.com> and create a new blank project.
2. Upload `ink2midi_ieee.tex`.
3. Set the compiler to **pdfLaTeX** (Menu → Settings).
4. Click *Recompile*.

### Option B — Local pdflatex

```bash
cd paper
pdflatex ink2midi_ieee.tex
pdflatex ink2midi_ieee.tex   # second pass for refs
```

You need a TeX distribution (TeX Live on Linux, MacTeX on macOS, MiKTeX on
Windows) with the `IEEEtran` class installed. It ships with all major
distributions by default.

## Before you submit

The paper is written from honest project facts (README + source). A few
things still need your attention:

1. ~~**Authors / affiliation**~~ — done (Ojasvi Poonia, MSRIT, Bangalore).
2. ~~**Pipeline figure**~~ — done. The figure is now a native TikZ
   diagram drawn directly in LaTeX (no external image needed). It
   shows: Input image → YOLOv8 detection → $k$-NN graph construction
   → GAT relationship parsing → staff/pitch → rhythm/MIDI export →
   Output MIDI, with the learned vs. rule-based stages grouped in
   dashed bands. If you would prefer a hand-drawn or rendered figure
   instead, replace the `\begin{tikzpicture}...\end{tikzpicture}`
   block with `\includegraphics[width=0.95\textwidth]{figures/pipeline.pdf}`.
3. **GAT results subsection (Sec. V-B)** — currently marked
   *Reserved*. Run the GAT trainer on the writer-disjoint test split
   and fill in edge-level Precision / Recall / F1 / AUROC, plus an
   ablation against the GCN baseline.
4. **End-to-end MIDI evaluation (Sec. V-C)** — currently marked
   *Reserved*. Once you have generated MIDI on a held-out subset,
   compute note-level F1 (pitch-only and pitch+onset+duration) and
   add the table.
5. **Inference runtime** — optionally add a small table reporting
   per-image inference time on your hardware.
6. **References** — bibliography is hand-written (no `.bib` file
   needed). Double-check the exact venue / year for
   `tuggener2024deepscores2` against the version you actually cite.
7. **Plagiarism / similarity check** — run the manuscript through
   IEEE's similarity checker (or Turnitin via your institution)
   before submission. Wording is original to this draft, but the
   conventional sentences in *Related Work* (e.g. classical OMR
   pipeline description) are common phrasing across the field; tweak
   them to your own voice if the similarity score flags them.
8. **Ethics statement** — Section VI-C is a starting point. If your
   target venue requires a specific ethics block, paste it in.

## What is *not* fabricated in this draft

- Detection numbers in Tables I and II are taken **directly from your
  README**.
- The architecture description matches the `src/omr/` layout:
  YOLOv8s detector, 3-layer GAT (4 heads, 128-d hidden), $k$-NN graph
  with $k=8$, 5-d edge features, Hough-based staff analysis,
  `music21` MIDI export.
- The writer-disjoint split (writers 1–35 / 36–42 / 43–50) and
  hyper-parameters match `configs/`.
- No GAT or MIDI-level metrics were invented; both subsections are
  explicitly marked *Reserved* until you measure them.

## Originality and ethics

The draft is written from scratch for this project; it does not copy
text from any prior publication. All datasets are publicly licensed
for research and are credited. Limitations are listed honestly
(low stem recall, single-voice assumption, missing end-to-end MIDI
metrics) so that the paper can pass peer review on its real merits.
