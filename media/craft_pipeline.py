"""
Manim animations for the CRAFT retrieval pipeline (two scenes).

CRAFT: Training-Free Cascaded Retrieval for Tabular QA.

Two scenes, both in the project's website palette on a near-black ground, in a
3Blue1Brown register: one idea on screen at a time, slow smooth easing, a moving
gold "query" dot that flows through the whole pipeline, and tables drawn as
elegant glyphs rather than screen-filling grids.

    CraftTeaser     — the one-breath cascade arc            (< 10 s)
    CraftExplainer  — the full story, enrich → rank → QA    (~ 30 s)

Render (manim is NOT a runtime dependency of the package — install it only to
render, and keep all output on the big volume so it never fills the home disk):

    pip install manim
    export MEDIA_DIR=/mnt/data1/asing725/ACL/CRAFT/media/out
    manim -qh --media_dir "$MEDIA_DIR" media/craft_pipeline.py CraftTeaser
    manim -qh --media_dir "$MEDIA_DIR" media/craft_pipeline.py CraftExplainer

See media/STORYBOARD.md for the scene-by-scene plan this file implements.
"""

import math
import os
import random

import numpy as np
from manim import *

# Keep every render artifact on the large volume; the home disk is tight.
config.media_dir = os.environ.get(
    "MEDIA_DIR", os.path.join(os.path.dirname(__file__), "out")
)

# ── Palette (from the CRAFT paper figure / project website) ──────────────────
BG = "#0e141d"      # near-black ground, the site's dark theme
INK = "#e7eef8"     # primary text
MUTED = "#93a2b8"   # captions / secondary text
BLUE = "#6ea8fe"    # Stage 1 · SPLADE · recall
GOLD = "#f5c542"    # the query, and Stage 2 survivors
VIOLET = "#b090f5"  # LLM / Stage 3 embedding space
TEAL = "#4bbcd0"    # sub-questions / the answer check
CELL = "#1a2536"    # table body fill

ICONS = os.path.join(os.path.dirname(__file__), "icons")


# ── Reusable glyphs ──────────────────────────────────────────────────────────
def table_card(width=0.9, height=0.66, header=BLUE, corner=VIOLET,
               body=CELL, rows=3, stroke=1.4):
    """A table drawn as an elegant card, not a filled grid.

    A colored header bar, a violet corner cell, and a few faint row lines — it
    reads as "a table" at a glance without ever holding real text (that lives on
    the website). This is the recurring motif across both scenes.
    """
    body_rect = RoundedRectangle(
        width=width, height=height, corner_radius=0.06,
        stroke_width=stroke, stroke_color=MUTED, fill_color=body, fill_opacity=1,
    )
    header_rect = Rectangle(
        width=width, height=height * 0.26, stroke_width=0,
        fill_color=header, fill_opacity=1,
    ).align_to(body_rect, UP).align_to(body_rect, LEFT)
    corner_cell = Rectangle(
        width=width * 0.24, height=height * 0.26, stroke_width=0,
        fill_color=corner, fill_opacity=1,
    ).align_to(header_rect, UL)
    lines = VGroup()
    inner_top = header_rect.get_bottom()[1]
    inner_bottom = body_rect.get_bottom()[1] + height * 0.12
    for i in range(rows):
        y = inner_top - (i + 1) * (inner_top - inner_bottom) / (rows + 1)
        line = Line(
            [body_rect.get_left()[0] + width * 0.12, y, 0],
            [body_rect.get_right()[0] - width * 0.12, y, 0],
            stroke_width=1.0, stroke_color=MUTED,
        ).set_opacity(0.45)
        lines.add(line)
    return VGroup(body_rect, header_rect, corner_cell, lines)


def query_dot(color=GOLD, scale=1.0):
    """The through-line: a small gold dot with a faint halo."""
    dot = Dot(color=color, radius=0.09 * scale)
    halo = Dot(color=color, radius=0.18 * scale).set_opacity(0.18)
    return VGroup(halo, dot)


def load_icon(name, color=INK, height=0.5):
    """Load an SVG icon and tint it, or fall back to a labeled dot.

    Rendering is optional; if the icon file or the SVG loader is unavailable we
    degrade to a simple placeholder so the scene still constructs.
    """
    path = os.path.join(ICONS, f"{name}.svg")
    try:
        icon = SVGMobject(path)
        icon.set_stroke(color=color, width=2).set_fill(color, opacity=0)
        # a couple of icons use filled dots for eyes / accents
        for sub in icon.family_members_with_points():
            if sub.get_fill_opacity() > 0:
                sub.set_fill(color, opacity=1)
        icon.set_height(height)
        return icon
    except Exception:
        return VGroup(
            Circle(radius=height / 2, color=color, stroke_width=2),
            Text(name[:1].upper(), color=color).scale(height * 0.7),
        )


def caption(text, color=MUTED, scale=0.34):
    return Text(text, font="sans-serif", color=color).scale(scale)


def stage_label(number, title, color):
    tag = Text(f"STAGE {number}", font="sans-serif", weight=BOLD, color=color).scale(0.30)
    name = Text(title, font="sans-serif", weight=BOLD, color=INK).scale(0.42)
    return VGroup(tag, name).arrange(DOWN, buff=0.08, aligned_edge=LEFT)


# The real example threaded through both scenes (NQ, Brazos River query).
QUESTION = "where does the brazos river start and stop"


# ══════════════════════════════════════════════════════════════════════════════
# Scene A — CraftTeaser  (< 10 s)
# ══════════════════════════════════════════════════════════════════════════════
class CraftTeaser(Scene):
    """The one-breath arc: a question funnels through three ever-finer stages to
    a short answer set. No enrichment, no QA — just the cascade *shape*."""

    def construct(self):
        self.camera.background_color = BG
        random.seed(11)

        # A1 · a gold query dot, and a loose cloud of faint table glyphs.
        q = query_dot(scale=1.2).move_to(LEFT * 4.2)
        self.play(FadeIn(q, scale=0.6), run_time=0.7)

        cloud = VGroup(*[table_card(rows=2) for _ in range(24)])
        for card in cloud:
            card.scale(0.42).move_to([
                random.uniform(-0.5, 4.6), random.uniform(-2.6, 2.6), 0
            ]).set_opacity(0.55)
        self.play(
            LaggedStart(*[FadeIn(c, scale=0.7) for c in cloud], lag_ratio=0.03),
            run_time=1.2,
        )

        label = Text("SPLADE", font="sans-serif", weight=BOLD, color=BLUE).scale(0.5)
        label.to_edge(UP, buff=0.7)
        cap = caption("wide net").to_edge(DOWN, buff=0.7)

        # A2 · wide ripple; most dim, ~8 stay lit (blue) and slide into an arc.
        self.play(FadeIn(label), FadeIn(cap), run_time=0.5)
        ripple = Circle(radius=0.1, color=BLUE, stroke_width=3).move_to(q.get_center())
        kept = VGroup(*cloud[:8])
        dropped = VGroup(*cloud[8:])
        self.play(
            ripple.animate.scale(70).set_opacity(0),
            kept.animate.set_stroke(BLUE, width=2.4).set_opacity(1),
            dropped.animate.set_opacity(0.08),
            run_time=1.3, rate_func=smooth,
        )
        arc_pts = [
            [1.8 + 1.6 * math.cos(a), 1.6 * math.sin(a), 0]
            for a in np.linspace(-1.0, 1.0, 8)
        ]
        self.play(
            *[kept[i].animate.move_to(arc_pts[i]) for i in range(8)],
            FadeOut(dropped), run_time=0.9, rate_func=smooth,
        )

        # A3 · label morphs; survivors collapse, row-lines flicker, 4 remain (gold).
        label2 = Text("Dense", font="sans-serif", weight=BOLD, color=GOLD).scale(0.5).move_to(label)
        cap2 = caption("by meaning").move_to(cap)
        self.play(Transform(label, label2), Transform(cap, cap2), run_time=0.5)
        survivors = VGroup(*kept[:4])
        gone = VGroup(*kept[4:])
        for card in kept:
            card[3].set_stroke(GOLD, width=1.4)  # flicker the row-lines
        self.play(kept[0][3].animate.set_opacity(1), rate_func=there_and_back, run_time=0.4)
        col = [UP * 1.35, UP * 0.45, DOWN * 0.45, DOWN * 1.35]
        self.play(
            *[survivors[i].animate.move_to(RIGHT * 1.4 + col[i]).set_stroke(GOLD, width=2.4)
              for i in range(4)],
            FadeOut(gone), run_time=1.0, rate_func=smooth,
        )

        # A4 · label morphs; the 4 line up, a violet sweep orders them, top ★.
        label3 = Text("Rerank", font="sans-serif", weight=BOLD, color=VIOLET).scale(0.5).move_to(label)
        cap3 = caption("precise top-k").move_to(cap)
        self.play(Transform(label, label3), Transform(cap, cap3), run_time=0.5)
        sweep = Line(LEFT * 0.6, RIGHT * 0.6, stroke_width=3, color=VIOLET)
        sweep.set_opacity(0.8).move_to(RIGHT * 1.4 + UP * 1.9)
        self.play(sweep.animate.move_to(RIGHT * 1.4 + DOWN * 1.9), run_time=0.9, rate_func=smooth)
        ranks = VGroup(*[
            Text(str(i + 1), font="sans-serif", color=MUTED).scale(0.3)
            .next_to(survivors[i], LEFT, buff=0.2)
            for i in range(3)
        ])
        star = Star(n=5, outer_radius=0.16, color=GOLD, fill_opacity=1)
        star.next_to(survivors[0], LEFT, buff=0.2)
        self.play(FadeOut(sweep), LaggedStart(*[FadeIn(r) for r in ranks], lag_ratio=0.2),
                  run_time=0.6)
        self.play(FadeIn(star, scale=0.5), ranks[0].animate.set_opacity(0), run_time=0.4)

        # A5 · a faint funnel is traced behind: recall (wide) → precision (narrow).
        self.play(
            FadeOut(VGroup(q, survivors, ranks, star)),
            FadeOut(label), FadeOut(cap), run_time=0.6,
        )
        top = Line(LEFT * 4 + UP * 1.6, RIGHT * 4 + UP * 0.5, stroke_width=3)
        bot = Line(LEFT * 4 + DOWN * 1.6, RIGHT * 4 + DOWN * 0.5, stroke_width=3)
        for f in (top, bot):
            f.set_stroke([BLUE, GOLD, VIOLET])
        lft = caption("recall", BLUE).next_to(top, LEFT, buff=0.2).shift(DOWN * 1.6)
        rgt = caption("precision", VIOLET).next_to(top, RIGHT, buff=0.2).shift(DOWN * 0.6)
        wordmark = Text("CRAFT", font="sans-serif", weight=BOLD, color=INK).scale(1.3)
        self.play(Create(top), Create(bot), FadeIn(lft), FadeIn(rgt), run_time=1.0)
        self.play(Write(wordmark), run_time=0.9)
        self.wait(0.9)


# ══════════════════════════════════════════════════════════════════════════════
# Scene B — CraftExplainer  (~ 30 s)
# ══════════════════════════════════════════════════════════════════════════════
class CraftExplainer(Scene):
    """The full story, minimal per beat: enrich → SPLADE (recall) → semantic row
    filtering (noise↓) → rerank (top-k) → the QA payoff."""

    def construct(self):
        self.camera.background_color = BG
        random.seed(23)
        self.b0_problem()
        self.b1_enrichment()
        self.b2_splade()
        self.b3_row_filter()
        self.b4_rerank()
        self.b5_payoff()

    # B0 · the problem (0–3s)
    def b0_problem(self):
        question = Text(f'"{QUESTION}"', font="sans-serif", color=INK,
                        slant=ITALIC).scale(0.6).to_edge(UP, buff=1.0)
        self.play(Write(question), run_time=1.2)

        field = VGroup(*[table_card(rows=2) for _ in range(60)])
        for card in field:
            card.scale(0.3).move_to([
                random.uniform(-6, 6), random.uniform(-3, 1.4), 0
            ]).set_opacity(0.28)
        self.play(LaggedStart(*[FadeIn(c) for c in field], lag_ratio=0.006), run_time=1.2)
        counter = caption("169K tables", MUTED).next_to(question, DOWN, buff=0.4)
        self.play(FadeIn(counter), run_time=0.4)

        q = query_dot(scale=1.2).move_to(question.get_bottom() + DOWN * 0.15)
        self.play(FadeIn(q, scale=0.5), run_time=0.4)
        self.play(q.animate.move_to(DOWN * 1.2), run_time=0.7, rate_func=smooth)
        cap = caption("find the one table that answers it").to_edge(DOWN, buff=0.6)
        self.play(FadeIn(cap), run_time=0.4)
        self.wait(0.6)

        self.play(FadeOut(field), FadeOut(counter), FadeOut(cap),
                  question.animate.scale(0.7).to_edge(UP, buff=0.5), run_time=0.7)
        self.q = q
        self.question = question

    # B1 · enrichment (3–8s) — why recall improves
    def b1_enrichment(self):
        head = Text("Enrich both sides", font="sans-serif", weight=BOLD,
                    color=INK).scale(0.44).next_to(self.question, DOWN, buff=0.4)
        self.play(FadeIn(head), run_time=0.4)

        bare = table_card(width=1.5, height=1.05, rows=3).move_to(LEFT * 3.2 + DOWN * 0.6)
        self.play(self.q.animate.move_to(RIGHT * 3.0 + DOWN * 0.3), FadeIn(bare), run_time=0.6)

        llm = load_icon("llm", VIOLET, height=0.6).next_to(bare, UP, buff=0.35)
        self.play(FadeIn(llm, shift=DOWN * 0.15), run_time=0.5)

        # A title ribbon + faint description lines grow onto the card.
        ribbon = Rectangle(width=1.3, height=0.16, stroke_width=0,
                           fill_color=GOLD, fill_opacity=0.9).move_to(bare.get_top() + DOWN * 0.5)
        desc = VGroup(*[
            Line(LEFT * 0.5, RIGHT * 0.5, stroke_width=2, color=MUTED).set_opacity(0.5)
            for _ in range(3)
        ]).arrange(DOWN, buff=0.1).next_to(ribbon, DOWN, buff=0.12)
        self.play(GrowFromCenter(ribbon), run_time=0.4)
        self.play(LaggedStart(*[Create(l) for l in desc], lag_ratio=0.2), run_time=0.6)
        self.play(bare[1].animate.set_fill(BLUE, opacity=1), run_time=0.3)

        # The query sprouts one sub-question branch.
        branch = Line(self.q.get_center(), self.q.get_center() + RIGHT * 1.1 + DOWN * 0.5,
                      stroke_width=2, color=TEAL)
        subq = caption("→ source? → mouth?", TEAL, 0.3).next_to(branch.get_end(), RIGHT, buff=0.1)
        self.play(Create(branch), FadeIn(subq), run_time=0.6)

        cap = caption("LLM titles & sub-questions give the sparse model more to match")
        cap.to_edge(DOWN, buff=0.6)
        self.play(FadeIn(cap), run_time=0.5)
        self.wait(0.8)

        self.play(
            FadeOut(VGroup(head, bare, llm, ribbon, desc, branch, subq, cap)),
            self.q.animate.move_to(LEFT * 5).set_opacity(1), run_time=0.6,
        )

    # B2 · Stage 1 · SPLADE (8–13s) — recall, wide net
    def b2_splade(self):
        label = stage_label("1", "SPLADE", BLUE).next_to(self.question, DOWN, buff=0.35).to_edge(LEFT, buff=0.7)
        self.play(FadeIn(label), run_time=0.4)

        field = VGroup(*[table_card(rows=2) for _ in range(48)])
        for card in field:
            card.scale(0.34).move_to([
                random.uniform(-2.5, 5.5), random.uniform(-2.6, 1.6), 0
            ]).set_opacity(0.4)
        self.play(LaggedStart(*[FadeIn(c) for c in field], lag_ratio=0.008), run_time=0.9)

        ripple = Circle(radius=0.1, color=BLUE, stroke_width=3).move_to(self.q.get_center())
        kept = VGroup(*field[:8])
        dropped = VGroup(*field[8:])
        self.play(
            ripple.animate.scale(90).set_opacity(0),
            dropped.animate.set_opacity(0.06),
            kept.animate.set_stroke(BLUE, width=2.2).set_opacity(1),
            run_time=1.3, rate_func=smooth,
        )
        self.play(
            kept.animate.arrange(DOWN, buff=0.18).scale(1.1).move_to(RIGHT * 3.5),
            FadeOut(dropped), run_time=0.9, rate_func=smooth,
        )
        brace = Brace(kept, LEFT, color=MUTED)
        band = caption("5,000 kept", MUTED, 0.3).next_to(brace, LEFT, buff=0.15)
        cap = caption("cast a wide, cheap net — keep recall high").to_edge(DOWN, buff=0.6)
        self.play(FadeIn(brace), FadeIn(band), FadeIn(cap), run_time=0.5)
        self.wait(0.8)

        self.play(FadeOut(VGroup(label, brace, band, cap)), run_time=0.4)
        self.stage1 = kept

    # B3 · Stage 2 · semantic row filtering (13–20s) — noise down
    def b3_row_filter(self):
        label = stage_label("2", "Dense · mini-tables", GOLD).next_to(self.question, DOWN, buff=0.35).to_edge(LEFT, buff=0.7)
        self.play(FadeIn(label), run_time=0.4)

        # Zoom into ONE survivor; its rows fan out and are scored by the query.
        focus = self.stage1[0]
        self.play(self.stage1[1:].animate.set_opacity(0.2), run_time=0.4)
        big = focus.copy().set_opacity(1)
        self.play(big.animate.scale(2.6).move_to(LEFT * 3.2 + DOWN * 0.2), run_time=0.6)

        rows = VGroup(*[
            Rectangle(width=2.0, height=0.22, stroke_width=1, stroke_color=MUTED,
                      fill_color=CELL, fill_opacity=1)
            for _ in range(6)
        ]).arrange(DOWN, buff=0.06).next_to(big, RIGHT, buff=0.7)
        rows[0].set_fill(BLUE)  # header row
        self.play(TransformFromCopy(big, rows), run_time=0.7)

        # The query scores each row; top rows light gold, the rest fade.
        q_here = self.q.copy().next_to(rows, RIGHT, buff=0.5)
        self.play(FadeIn(q_here), run_time=0.3)
        lit = [1, 3]  # the rows that matter
        self.play(
            *[rows[i].animate.set_fill(GOLD, opacity=0.85) for i in lit],
            *[rows[i].animate.set_opacity(0.15) for i in (2, 4, 5)],
            run_time=0.8,
        )
        mini = VGroup(rows[0].copy(), rows[1].copy(), rows[3].copy())
        self.play(
            FadeOut(rows), FadeOut(q_here),
            mini.animate.arrange(DOWN, buff=0.04).move_to(rows.get_center()),
            run_time=0.7,
        )
        mini_cap = caption("mini-table", GOLD, 0.3).next_to(mini, DOWN, buff=0.15)
        self.play(FadeIn(mini_cap), run_time=0.3)
        self.wait(0.5)

        # Pull back: the column of ~8 becomes ~4 compact mini-tables (gold).
        self.play(FadeOut(VGroup(big, mini, mini_cap)), run_time=0.4)
        minis = VGroup(*[table_card(width=0.9, height=0.5, header=GOLD, rows=2)
                         for _ in range(4)])
        minis.arrange(DOWN, buff=0.2).move_to(RIGHT * 3.5)
        for m in minis:
            m.set_stroke(GOLD, width=2.2)
        self.play(FadeOut(self.stage1), LaggedStart(*[FadeIn(m, scale=0.8) for m in minis],
                                                    lag_ratio=0.12), run_time=0.8)
        cap = caption("keep only the rows that matter — less noise, sharper matches")
        cap.to_edge(DOWN, buff=0.6)
        self.play(FadeIn(cap), run_time=0.5)
        self.wait(0.8)

        self.play(FadeOut(VGroup(label, cap)), run_time=0.4)
        self.stage2 = minis

    # B4 · Stage 3 · rerank (20–25s) — top-k
    def b4_rerank(self):
        label = stage_label("3", "Rerank", VIOLET).next_to(self.question, DOWN, buff=0.35).to_edge(LEFT, buff=0.7)
        self.play(FadeIn(label), run_time=0.4)

        space = Circle(radius=2.0, stroke_color=VIOLET, stroke_width=1.5,
                       fill_opacity=0.05, fill_color=VIOLET).move_to(RIGHT * 1.0 + DOWN * 0.2)
        space_cap = caption("embedding space", MUTED, 0.28).next_to(space, DOWN, buff=0.15)
        self.play(Create(space), FadeIn(space_cap), run_time=0.6)

        q_center = query_dot(scale=1.3).move_to(space.get_center())
        pts = VGroup()
        for _ in range(4):
            r = random.uniform(0.5, 1.7)
            a = random.uniform(0, TAU)
            pts.add(Dot([r * math.cos(a) + space.get_x(),
                         r * math.sin(a) + space.get_y(), 0], color=GOLD).scale(0.9))
        self.play(
            ReplacementTransform(self.stage2, pts), FadeIn(q_center), run_time=0.9,
        )

        # Order by distance to the query; nearest 3 line up, #1 gets a ★.
        dists = sorted(pts, key=lambda p: np.linalg.norm(p.get_center() - q_center[1].get_center()))
        near = VGroup(*dists[:3])
        self.play(
            *[Create(Line(q_center.get_center(), p.get_center(), stroke_width=1.4,
                          color=VIOLET).set_opacity(0.6)) for p in near],
            near.animate.set_color(GOLD), run_time=0.8,
        )
        star = Star(n=5, outer_radius=0.16, color=GOLD, fill_opacity=1).move_to(dists[0].get_center())
        answer = caption("Brazos River", GOLD, 0.3).next_to(dists[0], UP, buff=0.2)
        self.play(FadeIn(star, scale=0.5), FadeIn(answer), run_time=0.5)

        note = caption("any reranker / embedding model", MUTED, 0.26).set_opacity(0.7)
        note.next_to(space, UP, buff=0.2)
        cap = caption("a stronger model — run only on the finalists — sets the final order")
        cap.to_edge(DOWN, buff=0.6)
        self.play(FadeIn(note), FadeIn(cap), run_time=0.5)
        self.wait(1.0)

        self.play(FadeOut(VGroup(label, space, space_cap, q_center, pts, star,
                                 answer, note, cap)), run_time=0.6)

    # B5 · the payoff (25–30s) — deeper recall + better QA
    def b5_payoff(self):
        # Left payoff: a recall curve rises and the marker lands past @5.
        axes = VGroup(
            Line(LEFT * 6 + DOWN * 1.5, LEFT * 6 + UP * 1.2, stroke_width=2, color=MUTED),
            Line(LEFT * 6 + DOWN * 1.5, LEFT * 1.8 + DOWN * 1.5, stroke_width=2, color=MUTED),
        )
        curve = VMobject(stroke_color=BLUE, stroke_width=4)
        curve.set_points_smoothly([
            LEFT * 6 + DOWN * 1.2, LEFT * 4.7 + DOWN * 0.2,
            LEFT * 3.4 + UP * 0.5, LEFT * 1.8 + UP * 0.85,
        ])
        depth = caption("recall at greater depth", MUTED, 0.3).next_to(axes, DOWN, buff=0.2).shift(RIGHT * 1.5)
        marker = Dot(LEFT * 1.8 + UP * 0.85, color=GOLD).scale(1.1)
        at5 = caption("@5+", GOLD, 0.28).next_to(marker, RIGHT, buff=0.1)
        self.play(Create(axes), run_time=0.4)
        self.play(Create(curve), run_time=1.0)
        self.play(FadeIn(marker, scale=0.5), FadeIn(at5), FadeIn(depth), run_time=0.4)

        # Right payoff: three mini-tables fit an LLM-context bracket; two full
        # tables overflowed it → a green check.
        bracket = VGroup(
            Line(UP * 1.2, DOWN * 1.2, stroke_width=2, color=MUTED),
            Line(UP * 1.2, UP * 1.2 + RIGHT * 0.2, stroke_width=2, color=MUTED),
            Line(DOWN * 1.2, DOWN * 1.2 + RIGHT * 0.2, stroke_width=2, color=MUTED),
        ).move_to(RIGHT * 2.6)
        ctx = caption("LLM context", MUTED, 0.28).next_to(bracket, UP, buff=0.15)
        minis = VGroup(*[table_card(width=1.1, height=0.42, header=GOLD, rows=1)
                         for _ in range(3)])
        minis.arrange(DOWN, buff=0.14).next_to(bracket, RIGHT, buff=0.25)
        for m in minis:
            m.set_stroke(GOLD, width=2)
        self.play(Create(bracket), FadeIn(ctx), run_time=0.4)
        self.play(LaggedStart(*[FadeIn(m, shift=LEFT * 0.2) for m in minis],
                              lag_ratio=0.15), run_time=0.7)
        check = load_icon("check", TEAL, height=0.5).next_to(minis, RIGHT, buff=0.35)
        self.play(FadeIn(check, scale=0.6), run_time=0.4)
        self.wait(0.8)

        # Condense to the wordmark + one line.
        self.play(FadeOut(VGroup(axes, curve, marker, at5, depth, bracket, ctx, minis, check),
                          shift=DOWN * 0.2), FadeOut(self.question), run_time=0.6)
        wordmark = Text("CRAFT", font="sans-serif", weight=BOLD, color=INK).scale(1.3).shift(UP * 0.4)
        line = caption("cascade to cut noise — spend big models only on the top candidates",
                       INK, 0.4).next_to(wordmark, DOWN, buff=0.4)
        self.play(Write(wordmark), run_time=0.8)
        self.play(FadeIn(line), run_time=0.6)
        self.wait(2.0)
