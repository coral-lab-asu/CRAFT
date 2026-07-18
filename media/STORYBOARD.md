# CRAFT — Manim Storyboard

Two scenes in `craft_pipeline.py`, both in the **website palette** on a near-black
ground, 3Blue1Brown-style: one idea on screen at a time, smooth `Transform`/`Write`
morphs, a moving **query dot** that flows through the pipeline, tables as elegant
glyphs (not screen-filling grids), sparse text, slow easing.

Palette (from the site):
`BG #0e141d` · `INK #e7eef8` · `MUTED #93a2b8` · `BLUE #6ea8fe` · `GOLD #f5c542`
· `VIOLET #b090f5` · `TEAL #4bbcd0` · `CELL #1a2536`

Motion language:
- The **query** is a small gold dot. It is the through-line; it never disappears,
  it travels stage to stage.
- A **table** is a rounded card: a colored header bar + a violet corner cell +
  faint row lines. Never filled with real text in the video (that lives on the
  website); here they read as *tables* at a glance.
- Transitions are `rate_func=smooth` and deliberately slow; nothing snaps.

---

## Scene A — `CraftTeaser`  (target < 10 s)

The one-breath arc: **a question funnels through three ever-finer stages to a
short answer set.** No enrichment, no QA — just the cascade shape.

| # | Time | On screen | Motion |
|---|------|-----------|--------|
| A1 | 0.0–1.2s | Gold query dot appears center-left; a loose cloud of ~24 faint table glyphs on the right. | Dot fades in with a soft pulse; glyphs stagger-fade in. |
| A2 | 1.2–3.2s | Label "SPLADE" fades in above. The dot sends a wide ripple; most glyphs dim, ~8 stay lit (blue stroke). | Ripple expands (Circle grow+fade). Kept glyphs slide into a tidy arc. Caption: **wide net**. |
| A3 | 3.2–5.2s | Label morphs "SPLADE → Dense". The 8 collapse toward the dot; each briefly shows 2–3 row-lines highlighting (semantic rows), then 4 remain (gold stroke). | `Transform` the label; row-lines flicker; 4 survivors scale up slightly. Caption: **by meaning**. |
| A4 | 5.2–7.4s | Label morphs "Dense → Rerank". The 4 line up vertically; a violet sweep orders them; the top gains a gold ★. | Sweep = a violet line wiping top→bottom; ranks 1–3 fade in as small numerals. Caption: **precise top-k**. |
| A5 | 7.4–9.2s | Everything else dims; the funnel shape is drawn as one faint curve behind: wide→narrow, labeled `recall` (left) → `precision` (right). Title "CRAFT" writes in. | The three stage colors (blue/gold/violet) trace the funnel. "CRAFT" `Write`. Hold ~1s. |

End state: CRAFT wordmark + the funnel + the ★ top result. ~9s total.

---

## Scene B — `CraftExplainer`  (target ~30 s)

The full story, still minimal-per-beat. Order follows the paper: **enrich →
SPLADE (recall) → semantic row filtering (noise↓) → rerank (top-k) → QA payoff.**

### B0 · The problem  (0–3s)
- A single natural-language question appears (real: *"where does the brazos river
  start and stop"*), typed via `Write`.
- Beneath it, a large but faint field of table glyphs blooms outward — "hundreds
  of thousands." A small counter ticks up then abbreviates to `169K`.
- The gold **query dot** detaches from the question and drops toward the field.
- *Caption (small, bottom):* find the one table that answers it.

### B1 · Enrichment  (3–8s)  — *why recall improves*
- Pull ONE bare table glyph to center (others dim). It has an empty header bar.
- A small **LLM** badge (violet) touches it; a **title** ribbon + three faint
  description lines grow onto the card. The header bar fills blue.
- Simultaneously on the query side: the question sprouts one **sub-question**
  branch (a thin teal line to a second, shorter chip).
- *Caption:* LLM-written titles & sub-questions give the sparse model more to match.
- The enriched table and expanded query rejoin the field.

### B2 · Stage 1 · SPLADE  (8–13s)  — *recall, wide net*
- Label "STAGE 1 · SPLADE" (blue). The query dot emits a **wide lexical ripple**
  across the whole field.
- Most glyphs dim to ~10%; a band of ~8 keep a blue stroke and glide into a column.
- A bracket labels them `5,000` (shown as the surviving band).
- *Caption:* cast a wide, cheap net — keep recall high.

### B3 · Stage 2 · semantic row filtering  (13–20s)  — *noise down*
- Label morphs to "STAGE 2 · Dense". Zoom into ONE surviving table: its rows
  fan out as thin bars. The query dot scores each row; the **top rows light gold**,
  the rest fade — then the lit rows + the header snap together into a **mini-table**.
- Camera pulls back: the column of ~8 becomes ~4 (gold stroke), each now a compact
  mini-table.
- *Caption:* keep only the rows that matter — less noise, sharper matches.

### B4 · Stage 3 · rerank  (20–25s)  — *top-k*
- Label morphs to "STAGE 3 · Rerank". The 4 mini-tables enter a soft **violet
  embedding field** (a faint circle). Points arrange by distance to the query dot.
- A violet ordering sweep; the nearest 3 line up as a ranked list, #1 gets a ★
  (this is the real answer table: *Brazos River*).
- *Caption:* a stronger model — run only on the finalists — sets the final order.
- Small note: "any reranker / embedding model" (model-agnostic), faded.

### B5 · The payoff  (25–30s)  — *deeper recall + better QA*
- Two quiet payoffs animate side by side:
  1. **Deeper recall:** a tiny curve rises and the marker lands past `@5`
     (label "competitive recall at greater depth"). No numbers.
  2. **Fits more context:** three **mini-tables** slot neatly inside an "LLM
     context" bracket where two **full tables** overflowed it → a green ✓ answer.
- Everything condenses behind the **CRAFT** wordmark + the one-line:
  *cascade to cut noise, spend big models only on the top candidates.*
- Hold ~2s.

---

## Asset list
- Hand-built mobjects: query dot, table card (header+corner+rows), mini-table,
  ripple, embedding field, funnel curve, context bracket, ranked list, ★.
- SVG icons (in `media/icons/`, imported via `SVGMobject`): `search`, `llm`,
  `check`. Kept minimal and monochrome so they tint to the palette.
- Real example + table titles pulled from the NQ export (Brazos River query).

## Timing philosophy
Every morph uses slow `smooth`/`ease_in_out`; holds of 0.6–1.0s after each beat so
the eye can rest. Scene A compresses the same language into single quick beats to
stay under 10s; Scene B lets each beat breathe.
