"""
Manim animation of the CRAFT retrieval pipeline.

CRAFT: Training-Free Cascaded Retrieval for Tabular QA.
This video explains, scene by scene, what each stage does, why it is there, and
how the stages compose into one training-free retrieval pipeline. It is built to
teach the idea with motion and as little text as possible.

Render:
    pip install manim
    manim -qh media/craft_pipeline.py CraftPipeline      # 1080p
    manim -qk media/craft_pipeline.py CraftPipeline      # 4k
    manim -ql media/craft_pipeline.py CraftPipeline      # fast preview

The video is one Scene composed of ordered parts so the narrative flows:
    1. The problem      - a question + a huge pile of tables.
    2. Preprocessing    - enrich tables (title + description) and expand the query.
    3. Stage 1 (SPLADE) - sparse lexical filter: everything -> 5,000 (high recall).
    4. Stage 2 (Dense)  - mini-tables + sentence encoder: 5,000 -> 100 (semantic).
    5. Stage 3 (Neural) - embedding rerank: 100 -> top-k (high precision).
    6. Answer           - top-k mini-tables + question -> LLM -> answer.
    7. The arc          - "high recall to high precision", training-free.

Palette matches the paper figure and the project website.
"""

import math

import numpy as np
from manim import *

# ── Palette (from the CRAFT paper figure / website) ──────────────────────────
BLUE = "#2f7fe0"
NAVY = "#1e3a5f"
GOLD = "#eab308"
VIOLET = "#6d4bd6"
TEAL = "#2e9bb0"
PAGE = "#0e141d"      # dark ground, like the site's dark theme
INK = "#e7eef8"
MUTED = "#93a2b8"
CELL = "#dbeafe"


def mini_table(width=0.7, height=0.5, header=BLUE, corner=VIOLET, body=CELL):
    """A tiny table glyph: a header bar, a violet corner cell, a pale body.

    This is the paper's recurring table motif; we reuse it everywhere so the
    viewer reads "table" instantly without any label.
    """
    body_rect = Rectangle(width=width, height=height, stroke_width=1.2,
                          stroke_color=NAVY, fill_color=body, fill_opacity=1)
    header_rect = Rectangle(width=width, height=height * 0.28, stroke_width=1.2,
                            stroke_color=NAVY, fill_color=header, fill_opacity=1)
    header_rect.align_to(body_rect, UP)
    corner_cell = Rectangle(width=width * 0.22, height=height * 0.28, stroke_width=1.2,
                            stroke_color=NAVY, fill_color=corner, fill_opacity=1)
    corner_cell.align_to(header_rect, UL)
    return VGroup(body_rect, header_rect, corner_cell)


def stage_label(number, title, subtitle, color):
    """A compact stage caption: 'STAGE n', a title, and a one-line role."""
    tag = Text(f"STAGE {number}", font="sans-serif", weight=BOLD, color=color).scale(0.32)
    name = Text(title, font="sans-serif", weight=BOLD, color=INK).scale(0.5)
    role = Text(subtitle, font="sans-serif", color=MUTED).scale(0.3)
    group = VGroup(tag, name, role).arrange(DOWN, buff=0.1, aligned_edge=LEFT)
    return group


class CraftPipeline(Scene):
    def construct(self):
        self.camera.background_color = PAGE
        self.problem()
        self.preprocessing()
        self.stage_one()
        self.stage_two()
        self.stage_three()
        self.answer()
        self.closing_arc()

    # ── 1. The problem ───────────────────────────────────────────────────────
    def problem(self):
        """A natural-language question must find its one table among many.

        Establishes the task: open-domain table QA is a needle-in-a-haystack
        retrieval problem before it is a reading problem.
        """
        title = Text("CRAFT", font="sans-serif", weight=BOLD, color=BLUE).scale(1.6)
        sub = Text("Training-Free Cascaded Retrieval for Tabular QA",
                   font="sans-serif", color=INK).scale(0.5)
        VGroup(title, sub).arrange(DOWN, buff=0.3)
        self.play(FadeIn(title, shift=UP * 0.3), run_time=0.8)
        self.play(FadeIn(sub), run_time=0.6)
        self.wait(0.6)
        self.play(VGroup(title, sub).animate.scale(0.55).to_edge(UP), run_time=0.8)

        question = Text('"where does the brazos river start and stop"',
                        font="sans-serif", color=INK, slant=ITALIC).scale(0.55)
        question.next_to(VGroup(title, sub), DOWN, buff=0.5)
        self.play(Write(question), run_time=1.0)

        # A large field of tables: the corpus (169K / 419K in practice).
        pile = VGroup(*[mini_table() for _ in range(0, 84)])
        pile.arrange_in_grid(rows=6, cols=14, buff=0.12).scale(0.72)
        pile.next_to(question, DOWN, buff=0.5)
        self.play(LaggedStart(*[FadeIn(t, scale=0.6) for t in pile],
                              lag_ratio=0.01), run_time=1.6)
        corpus_note = Text("hundreds of thousands of tables",
                           font="sans-serif", color=MUTED).scale(0.34)
        corpus_note.next_to(pile, DOWN, buff=0.25)
        self.play(FadeIn(corpus_note), run_time=0.5)
        self.wait(0.8)

        self.q_text = question
        self.header = VGroup(title, sub)
        self.play(FadeOut(pile), FadeOut(corpus_note), run_time=0.7)

    # ── 2. Preprocessing ─────────────────────────────────────────────────────
    def preprocessing(self):
        """Enrich both sides before retrieval so lexical + semantic matching work.

        Why: tables often lack good titles and share little vocabulary with
        questions. CRAFT closes that gap up front — an LLM writes a title and
        description for every table and decomposes the query into sub-questions.
        """
        heading = Text("Preprocessing — enrich both sides",
                       font="sans-serif", weight=BOLD, color=INK).scale(0.5)
        heading.next_to(self.header, DOWN, buff=0.4)
        self.play(FadeIn(heading), run_time=0.5)

        # Left: a bare table gains a title + description.
        bare = mini_table(width=1.1, height=0.8)
        bare.move_to(LEFT * 3.2 + DOWN * 0.3)
        arrow1 = Arrow(LEFT * 1.9, LEFT * 0.6, buff=0.1, color=GOLD, stroke_width=4)
        arrow1.move_to(LEFT * 1.6 + DOWN * 0.3)
        enriched = VGroup(
            mini_table(width=1.1, height=0.8),
            Text("Title", font="sans-serif", weight=BOLD, color=GOLD).scale(0.3),
            Text("+ description", font="sans-serif", color=MUTED).scale(0.26),
        )
        enriched[1:].arrange(DOWN, buff=0.08)
        enriched.arrange(DOWN, buff=0.12)
        enriched.move_to(LEFT * 0.2 + DOWN * 0.3)
        gemini = Text("LLM", font="sans-serif", weight=BOLD, color=VIOLET).scale(0.3)
        gemini.next_to(arrow1, UP, buff=0.12)

        self.play(FadeIn(bare, shift=RIGHT * 0.2), run_time=0.5)
        self.play(GrowArrow(arrow1), FadeIn(gemini), run_time=0.5)
        self.play(FadeIn(enriched, shift=RIGHT * 0.2), run_time=0.6)

        # Right: the query expands into sub-questions.
        q_small = self.q_text.copy().scale(0.7).move_to(RIGHT * 2.2 + UP * 0.4)
        subq = VGroup(
            Text("→ source location?", font="sans-serif", color=TEAL).scale(0.34),
            Text("→ mouth location?", font="sans-serif", color=TEAL).scale(0.34),
        ).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        subq.next_to(q_small, DOWN, buff=0.25, aligned_edge=LEFT)
        self.play(TransformFromCopy(self.q_text, q_small), run_time=0.6)
        self.play(LaggedStart(*[FadeIn(s, shift=RIGHT * 0.2) for s in subq],
                              lag_ratio=0.3), run_time=0.8)
        self.wait(0.9)

        self.play(FadeOut(VGroup(heading, bare, arrow1, gemini, enriched, q_small, subq)),
                  run_time=0.6)

    # ── 3. Stage 1: SPLADE (sparse) ──────────────────────────────────────────
    def stage_one(self):
        """Sparse lexical retrieval — cast a wide, cheap net for high recall.

        Why first: SPLADE scores the whole corpus with a sparse, memory-light
        representation, so it can afford to look at every table. Its job is not
        precision but recall — keep the gold table in a much smaller shortlist.
        """
        label = stage_label("1", "SPLADE — sparse lexical",
                            "scan everything, keep the top 5,000  ·  high recall", BLUE)
        label.to_edge(LEFT).shift(UP * 2.2)
        self.play(FadeIn(label), run_time=0.5)

        many = VGroup(*[mini_table() for _ in range(0, 70)])
        many.arrange_in_grid(rows=5, cols=14, buff=0.12).scale(0.7)
        many.move_to(DOWN * 0.4)
        self.play(LaggedStart(*[FadeIn(t) for t in many], lag_ratio=0.008),
                  run_time=1.0)

        # A sparse "term match" sweep highlights a subset, the rest dim away.
        keep = many[:16]
        drop = many[16:]
        self.play(
            keep.animate.set_stroke(color=BLUE, width=2.5),
            run_time=0.6,
        )
        self.play(drop.animate.set_opacity(0.12), run_time=0.7)
        count = Text("5,000", font="sans-serif", weight=BOLD, color=BLUE).scale(0.5)
        count.next_to(many, DOWN, buff=0.3)
        self.play(FadeIn(count, shift=UP * 0.2), run_time=0.5)
        self.wait(0.8)

        # Collapse the survivors into a tidy shortlist to carry into Stage 2.
        shortlist = keep.copy()
        self.play(FadeOut(drop), FadeOut(count), FadeOut(label), run_time=0.5)
        self.stage1_out = shortlist
        self.play(shortlist.animate.arrange_in_grid(rows=2, cols=8, buff=0.14)
                  .scale(1.0).move_to(ORIGIN), run_time=0.8)

    # ── 4. Stage 2: mini-tables + dense encoder ──────────────────────────────
    def stage_two(self):
        """Build mini-tables and rerank them semantically — meaning over words.

        Why here: lexical overlap misses paraphrases and numeric/semantic cues.
        CRAFT compresses each candidate to a mini-table (its most query-relevant
        rows + headers) and scores it with a sentence encoder, narrowing 5,000
        to 100 by meaning, not just shared terms.
        """
        label = stage_label("2", "Dense reranking — mini-tables",
                            "compress to key rows, rank by meaning  ·  5,000 → 100", GOLD)
        label.to_edge(LEFT).shift(UP * 2.2)
        self.play(FadeIn(label), run_time=0.5)

        # Show one candidate shrinking to a mini-table (top rows + headers).
        focus = self.stage1_out[0].copy().set_opacity(1)
        self.play(self.stage1_out.animate.set_opacity(0.25), run_time=0.4)
        self.play(focus.animate.scale(2.4).move_to(LEFT * 3 + DOWN * 0.3), run_time=0.6)
        rows = VGroup(*[
            Rectangle(width=1.4, height=0.16, stroke_width=1, stroke_color=NAVY,
                      fill_color=CELL, fill_opacity=1)
            for _ in range(5)
        ]).arrange(DOWN, buff=0.05)
        rows.next_to(focus, RIGHT, buff=0.6)
        rows[0].set_fill(BLUE)  # header row
        mini_label = Text("mini-table", font="sans-serif", color=GOLD).scale(0.3)
        mini_label.next_to(rows, DOWN, buff=0.15)
        self.play(TransformFromCopy(focus, rows), FadeIn(mini_label), run_time=0.8)

        encoder = Text("Sentence Transformer", font="sans-serif", weight=BOLD,
                       color=TEAL).scale(0.34)
        encoder.move_to(RIGHT * 3 + UP * 0.2)
        q_dot = Dot(color=INK).scale(0.8).next_to(encoder, DOWN, buff=0.3)
        q_tag = Text("query", font="sans-serif", color=MUTED).scale(0.26).next_to(q_dot, RIGHT, buff=0.15)
        self.play(FadeIn(encoder), FadeIn(q_dot), FadeIn(q_tag), run_time=0.5)
        self.wait(0.6)

        self.play(FadeOut(VGroup(focus, rows, mini_label, encoder, q_dot, q_tag)),
                  run_time=0.5)
        keep = self.stage1_out[:8]
        drop = self.stage1_out[8:]
        self.play(keep.animate.set_opacity(1).set_stroke(color=GOLD, width=2.5),
                  drop.animate.set_opacity(0.08), run_time=0.7)
        count = Text("100", font="sans-serif", weight=BOLD, color=GOLD).scale(0.5)
        count.next_to(self.stage1_out, DOWN, buff=0.3)
        self.play(FadeIn(count), run_time=0.4)
        self.wait(0.7)

        self.stage2_out = keep.copy()
        self.play(FadeOut(drop), FadeOut(count), FadeOut(label), run_time=0.5)
        self.play(self.stage2_out.animate.arrange(RIGHT, buff=0.2).scale(1.1)
                  .move_to(ORIGIN), run_time=0.7)

    # ── 5. Stage 3: neural rerank (embeddings) ───────────────────────────────
    def stage_three(self):
        """A strong embedding model orders the finalists — high precision.

        Why last: it is the most expensive model, so CRAFT only runs it on the
        100 survivors. It re-embeds each mini-table and the query in a rich
        semantic space to get the final top-k ordering right.
        """
        label = stage_label("3", "Neural reranking — embeddings",
                            "precise final ordering  ·  100 → top-k", VIOLET)
        label.to_edge(LEFT).shift(UP * 2.2)
        self.play(FadeIn(label), run_time=0.5)

        space = Circle(radius=1.9, stroke_color=VIOLET, stroke_width=1.5,
                       fill_opacity=0.05, fill_color=VIOLET).move_to(DOWN * 0.3)
        space_label = Text("embedding space", font="sans-serif", color=MUTED).scale(0.28)
        space_label.next_to(space, DOWN, buff=0.15)
        self.play(Create(space), FadeIn(space_label), run_time=0.6)

        # Scatter the finalists as points; the query is the center of attention.
        import random
        random.seed(7)
        pts = VGroup()
        for _ in range(8):
            r = random.uniform(0.3, 1.7)
            a = random.uniform(0, TAU)
            pts.add(Dot([r * math.cos(a) + space.get_x(),
                         r * math.sin(a) + space.get_y(), 0],
                        color=BLUE).scale(0.7))
        q_dot = Dot(space.get_center(), color=GOLD).scale(1.1)
        self.play(LaggedStart(*[FadeIn(p) for p in pts], lag_ratio=0.1),
                  FadeIn(q_dot), run_time=0.8)

        # Rank by distance to the query: nearest few become the top-k.
        dists = sorted(pts, key=lambda p: np.linalg.norm(p.get_center() - q_dot.get_center()))
        topk = VGroup(*dists[:3])
        self.play(topk.animate.set_color(GOLD).scale(1.3),
                  *[Create(Line(q_dot.get_center(), p.get_center(),
                                stroke_width=1.5, color=GOLD)) for p in topk],
                  run_time=0.8)
        star = Star(n=5, outer_radius=0.18, color=GOLD, fill_opacity=1)
        star.move_to(dists[0].get_center())
        self.play(FadeIn(star, scale=0.5), run_time=0.4)
        count = Text("top-k", font="sans-serif", weight=BOLD, color=VIOLET).scale(0.5)
        count.next_to(space, RIGHT, buff=0.6)
        self.play(FadeIn(count), run_time=0.4)
        self.wait(0.9)

        self.play(FadeOut(VGroup(space, space_label, pts, q_dot, topk, star, count, label)),
                  FadeOut(self.stage2_out), run_time=0.6)

    # ── 6. Answer generation ─────────────────────────────────────────────────
    def answer(self):
        """Top-k mini-tables + the question feed an LLM to produce the answer.

        Why mini-tables: passing compact, most-relevant rows (not whole tables)
        keeps the context small so smaller LLMs can answer accurately.
        """
        heading = Text("Answer generation", font="sans-serif", weight=BOLD,
                       color=INK).scale(0.5).next_to(self.header, DOWN, buff=0.5)
        self.play(FadeIn(heading), run_time=0.5)

        topk = VGroup(*[mini_table(width=1.0, height=0.72) for _ in range(3)])
        topk.arrange(RIGHT, buff=0.3).move_to(LEFT * 3 + DOWN * 0.3)
        star = Star(n=5, outer_radius=0.13, color=GOLD, fill_opacity=1)
        star.move_to(topk[0].get_top())
        arrow = Arrow(LEFT * 1.1, RIGHT * 0.6, color=BLUE, stroke_width=4).move_to(DOWN * 0.3)
        llm = Text("LLM", font="sans-serif", weight=BOLD, color=VIOLET).scale(0.4)
        llm.next_to(arrow, UP, buff=0.15)
        ans = VGroup(
            Text("Answer", font="sans-serif", weight=BOLD, color=GOLD).scale(0.4),
            Text("✓", color=TEAL).scale(0.6),
        ).arrange(RIGHT, buff=0.2).move_to(RIGHT * 3 + DOWN * 0.3)

        self.play(FadeIn(topk), FadeIn(star), run_time=0.6)
        self.play(GrowArrow(arrow), FadeIn(llm), run_time=0.5)
        self.play(FadeIn(ans, shift=RIGHT * 0.3), run_time=0.6)
        self.wait(1.0)
        self.play(FadeOut(VGroup(heading, topk, star, arrow, llm, ans)), run_time=0.6)

    # ── 7. The arc ───────────────────────────────────────────────────────────
    def closing_arc(self):
        """One line that captures the design: recall first, precision last.

        Each stage uses a more expressive (and costlier) model than the last, but
        on a progressively smaller set — recall is protected early, precision is
        earned late, and no stage is fine-tuned.
        """
        bar_w = 8.0
        track = Line(LEFT * bar_w / 2, RIGHT * bar_w / 2, stroke_width=6,
                     color=MUTED).move_to(DOWN * 0.2)
        grad = track.copy().set_stroke(
            color=[BLUE, GOLD, VIOLET], width=6)
        left = Text("high recall", font="sans-serif", color=BLUE).scale(0.4)
        right = Text("high precision", font="sans-serif", color=VIOLET).scale(0.4)
        left.next_to(track, LEFT, buff=0.3)
        right.next_to(track, RIGHT, buff=0.3)

        s1 = Text("SPLADE", font="sans-serif", color=BLUE).scale(0.32).move_to(track.point_from_proportion(0.16) + UP * 0.4)
        s2 = Text("Dense", font="sans-serif", color=GOLD).scale(0.32).move_to(track.point_from_proportion(0.5) + UP * 0.4)
        s3 = Text("Neural", font="sans-serif", color=VIOLET).scale(0.32).move_to(track.point_from_proportion(0.84) + UP * 0.4)

        self.play(Create(track), FadeIn(left), FadeIn(right), run_time=0.7)
        self.play(Create(grad), LaggedStart(FadeIn(s1), FadeIn(s2), FadeIn(s3),
                                            lag_ratio=0.3), run_time=1.0)

        tagline = Text("Off-the-shelf models, cascaded. No fine-tuning.",
                       font="sans-serif", color=INK).scale(0.42)
        tagline.next_to(track, DOWN, buff=0.7)
        self.play(Write(tagline), run_time=1.0)
        self.wait(1.4)
        self.play(FadeOut(VGroup(track, grad, left, right, s1, s2, s3, tagline)),
                  FadeOut(self.header), run_time=0.8)
