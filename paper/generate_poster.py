#!/usr/bin/env python3
"""
generate_poster.py — writes CRAFT_poster.drawio for ACL 2026

Canvas: 3456 × 4608 px  (36" × 48" @ 96 DPI)
3-column layout, ASU Maroon/Gold colour scheme.
Run:  python3 generate_poster.py
Then open CRAFT_poster.drawio in diagrams.net or draw.io desktop.
"""

import base64, os

IMG_DIR = os.path.join(os.path.dirname(__file__), 'static', 'images')
OUT_FILE = os.path.join(os.path.dirname(__file__), 'CRAFT_poster.drawio')

# ── Colours ───────────────────────────────────────────────────────────────────
MAROON = '#8C1D40'
GOLD   = '#FFC627'

# ── Styles ────────────────────────────────────────────────────────────────────
_base = 'rounded=1;arcSize=3;strokeColor=#CCCCCC;'
S_BG       = f'rounded=0;fillColor={MAROON};strokeColor=none;html=1;'
S_SEC_HDR  = (f'rounded=1;arcSize=5;fillColor={MAROON};strokeColor=none;'
              f'fontColor=#FFFFFF;fontStyle=1;fontSize=26;align=left;'
              f'spacingLeft=12;verticalAlign=middle;html=1;')
S_BODY     = (_base + 'fillColor=#FAFAFA;html=1;verticalAlign=top;align=left;'
              'spacingLeft=10;spacingTop=8;fontSize=21;whiteSpace=wrap;')
S_BODY_GOLD= (_base + f'fillColor=#FFF8E1;strokeColor={GOLD};strokeWidth=2;html=1;'
              'verticalAlign=top;align=left;spacingLeft=10;spacingTop=8;'
              'fontSize=21;whiteSpace=wrap;')
S_CALLOUT  = (f'rounded=1;arcSize=10;fillColor={GOLD};strokeColor={MAROON};strokeWidth=2;'
              f'fontStyle=1;fontSize=22;whiteSpace=wrap;html=1;'
              f'verticalAlign=middle;align=center;')
S_STAGE1   = (f'rounded=1;arcSize=15;fillColor=#FF6B35;strokeColor=none;'
              f'fontColor=#FFFFFF;fontStyle=1;fontSize=19;html=1;'
              f'align=center;verticalAlign=middle;')
S_STAGE2   = (f'rounded=1;arcSize=15;fillColor=#0D7CC5;strokeColor=none;'
              f'fontColor=#FFFFFF;fontStyle=1;fontSize=19;html=1;'
              f'align=center;verticalAlign=middle;')
S_STAGE3   = (f'rounded=1;arcSize=15;fillColor=#2E7D32;strokeColor=none;'
              f'fontColor=#FFFFFF;fontStyle=1;fontSize=19;html=1;'
              f'align=center;verticalAlign=middle;')
S_FOOTER_BG  = 'rounded=0;fillColor=#1a1a1a;strokeColor=none;html=1;'
S_FOOTER_TXT = ('text;html=1;align=left;verticalAlign=top;whiteSpace=wrap;'
                'strokeColor=none;fillColor=none;fontColor=#CCCCCC;fontSize=18;')
S_TEXT = ('text;html=1;align=left;verticalAlign=top;whiteSpace=wrap;'
          'strokeColor=none;fillColor=none;')

# ── Helpers ───────────────────────────────────────────────────────────────────

def b64(path: str) -> str:
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode()

def img_style(filename: str, mime: str) -> str:
    path = os.path.join(IMG_DIR, filename)
    if not os.path.exists(path):
        return None
    data = b64(path)
    return (f'shape=image;aspect=fixed;strokeColor=none;fillColor=none;'
            f'image=data:{mime};base64,{data};')

def xe(s: str) -> str:
    """XML-escape a string for use inside an XML attribute (double-quoted)."""
    return (s.replace('&', '&amp;')
             .replace('<', '&lt;')
             .replace('>', '&gt;')
             .replace('"', '&quot;'))

_cid = [1]

def _next_id() -> str:
    _cid[0] += 1
    return str(_cid[0])

_cells: list[str] = []

def cell(value: str, style: str, x: int, y: int, w: int, h: int) -> str:
    """Append a vertex mxCell and return its id."""
    cid = _next_id()
    _cells.append(
        f'    <mxCell id="{cid}" value="{xe(value)}" style="{style}" '
        f'vertex="1" parent="1">'
        f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" '
        f'as="geometry"/></mxCell>'
    )
    return cid

def img_cell(filename: str, mime: str, x: int, y: int, w: int, h: int) -> None:
    """Append a base64-embedded image cell (or a grey placeholder)."""
    s = img_style(filename, mime)
    if s is None:
        cell(f'[{filename}]', S_BODY, x, y, w, h)
        return
    cid = _next_id()
    _cells.append(
        f'    <mxCell id="{cid}" value="" style="{s}" vertex="1" parent="1">'
        f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" '
        f'as="geometry"/></mxCell>'
    )

def section(title: str, body_html: str,
            x: int, y: int, w: int, body_h: int,
            body_style: str = None) -> int:
    """Draw a section (header bar + body). Returns y below the section."""
    HDR_H = 44
    GAP   = 12
    cell(title, S_SEC_HDR, x, y, w, HDR_H)
    cell(body_html, body_style or S_BODY, x, y + HDR_H + GAP, w, body_h)
    return y + HDR_H + GAP + body_h + 20

# ── Canvas & column geometry ──────────────────────────────────────────────────
W, H   = 3456, 4608
C1X    = 40
C2X    = 1175
C3X    = 2310
CW     = 1105   # column width
FOOTER = 4448

# ── HEADER ────────────────────────────────────────────────────────────────────
cell('', S_BG, 0, 0, W, 280)

cell(
    '<font style="font-size:52px;font-weight:bold;color:#FFFFFF;">'
    'CRAFT: Training-Free Cascaded Retrieval for Tabular QA</font>',
    S_TEXT, 40, 10, 2900, 110
)
cell(
    f'<font style="font-size:26px;color:{GOLD};">'
    'Adarsh Singh<sup>1</sup>, Kushal Raj Bhandari<sup>2</sup>, '
    'Jianxi Gao<sup>3</sup>, Soham Dan<sup>2</sup>, '
    'Vivek Gupta<sup>1</sup></font>',
    S_TEXT, 40, 122, 2900, 52
)
cell(
    '<font style="font-size:20px;color:#FFFFFF;">'
    '<sup>1</sup>Arizona State University &amp;nbsp;&amp;nbsp;'
    '<sup>2</sup>Rensselaer Polytechnic Institute &amp;nbsp;&amp;nbsp;'
    '<sup>3</sup>Microsoft</font>',
    S_TEXT, 40, 178, 2900, 52
)
# ACL badge
cell(
    f'<b style="font-size:24px;color:{MAROON};">ACL 2026</b>'
    f'<br/><span style="font-size:19px;color:{MAROON};">Vienna, Austria</span>'
    f'<br/><span style="font-size:17px;color:#555;">arXiv: 2505.14984</span>',
    f'rounded=1;arcSize=10;fillColor={GOLD};strokeColor={MAROON};strokeWidth=3;'
    f'html=1;align=center;verticalAlign=middle;',
    2960, 80, 340, 160
)
# Logos
img_cell('asu_logo.png',   'image/png',  3326, 12,  120, 69)
img_cell('coral_logo.jpeg','image/jpeg', 3326, 90,  120, 122)

# ── COLUMN 1 ─────────────────────────────────────────────────────────────────

y1 = 300

# Introduction & Motivation
y1 = section(
    'Introduction &amp; Motivation',
    f'<ul style="font-size:20px;margin:4px 0 0 -16px;line-height:1.65">'
    '<li><b>Open-Domain Table QA (TQA)</b> requires finding the relevant table(s) '
    'from a large corpus (169K–419K tables) before answering a natural language query.</li>'
    '<li><b>Sparse retrievers</b> (BM25, SPLADE) are fast but miss semantic nuance '
    'between free-text queries and tabular structure.</li>'
    '<li><b>Dense retrievers</b> (DTR, BIBERT) capture semantics but need expensive '
    'fine-tuning on labeled data and are slow at scale.</li>'
    '<li><b>Key gap:</b> no prior <i>training-free</i> method combines sparse + dense '
    '+ LLM embeddings in a single cascaded pipeline for TQA.</li>'
    '</ul>',
    C1X, y1, CW, 480
)

# Problem Statement
y1 = section(
    'Problem Statement',
    f'<div style="font-size:21px;padding:4px;">'
    f'<b>Given:</b> A natural language query <i>q</i> and corpus <i>C</i> of N tables<br/>'
    f'<b>Find:</b> Top-k tables most likely containing the answer — '
    f'<b>without any task-specific training</b><br/><br/>'
    f'<span style="color:{MAROON};font-weight:bold;">Challenges:</span>'
    f'<ul style="margin:4px 0 0 -16px;">'
    f'<li>Lexical mismatch between NL queries &amp; tabular structure</li>'
    f'<li>Scale: 169K (NQ-Tables), 419K (OTT-QA) tables</li>'
    f'<li>Zero labeled data for target domain</li>'
    f'</ul></div>',
    C1X, y1, CW, 340
)

# CRAFT Architecture
cell('CRAFT Architecture', S_SEC_HDR, C1X, y1, CW, 44)
y1 += 44 + 8

img_w = CW
img_h = round(CW * 2153 / 2137)   # aspect-correct: ≈ 1114
img_cell('craft_overview.png', 'image/png', C1X, y1, img_w, img_h)
y1 += img_h + 6

cell(
    '<i style="font-size:18px;">Figure 1: CRAFT three-stage cascaded retrieval '
    'with LLM-based preprocessing.</i>',
    'text;html=1;align=center;verticalAlign=top;strokeColor=none;fillColor=none;fontSize=18;',
    C1X, y1, CW, 44
)
y1 += 52

# Stage badges row
bw = 350
cell('Stage 1: SPLADE<br/>169K → 5,000', S_STAGE1, C1X,           y1, bw,    60)
cell('Stage 2: all-mpnet<br/>5K → 100',  S_STAGE2, C1X + bw + 8,  y1, bw,    60)
cell('Stage 3: text-emb-3<br/>100 → k',  S_STAGE3, C1X + 2*(bw+8), y1, bw-5, 60)
y1 += 60 + 20

# Robustness
y1 = section(
    'Robustness Under Query Perturbation',
    f'<div style="font-size:20px;padding:4px;">'
    f'Stage 3 acts as a <b>noise-robust re-ranker</b>:<br/><br/>'
    f'<table style="border-collapse:collapse;width:100%;font-size:19px;">'
    f'<tr style="background:{MAROON};color:#FFF;">'
    f'<td style="padding:5px 10px;"><b>Condition</b></td>'
    f'<td style="padding:5px;text-align:center;width:120px;"><b>R@10</b></td></tr>'
    f'<tr style="background:#F5F5F5;">'
    f'<td style="padding:5px 10px;">Original queries (S1+S2+S3)</td>'
    f'<td style="padding:5px;text-align:center;">87.16</td></tr>'
    f'<tr><td style="padding:5px 10px;">Perturbed queries (S1+S2 only)</td>'
    f'<td style="padding:5px;text-align:center;">83.82</td></tr>'
    f'<tr style="background:#FFF8E1;font-weight:bold;">'
    f'<td style="padding:5px 10px;"><b>Perturbed + Stage 3 (CRAFT)</b></td>'
    f'<td style="padding:5px;text-align:center;"><b>87.16 ✓</b></td></tr>'
    f'</table><br/>'
    f'<b style="color:{MAROON};">Stage 3 fully recovers performance on noisy queries.</b>'
    f'</div>',
    C1X, y1, CW, 380
)

# Datasets
y1 = section(
    'Benchmark Datasets',
    f'<div style="font-size:20px;padding:4px;">'
    f'<b style="color:{MAROON};">NQ-Tables</b><br/>'
    f'&bull; 169,898 Wikipedia tables &bull; 966 test questions<br/>'
    f'&bull; Single-hop factual queries &bull; Source: Natural Questions<br/><br/>'
    f'<b style="color:{MAROON};">OTT-QA</b><br/>'
    f'&bull; 419,000 Wikipedia tables &bull; 2,214 dev questions<br/>'
    f'&bull; Multi-hop reasoning across tables &amp; passages<br/>'
    f'&bull; Evaluated in <b>zero-shot</b> transfer setting'
    f'</div>',
    C1X, y1, CW, 310
)

# Three key-metric callout boxes
if y1 + 20 + 155 < FOOTER:
    cy = y1 + 20
    bw2 = (CW - 2*12) // 3
    cell(f'<b style="font-size:26px;color:{MAROON};">86.83</b><br/>'
         '<span style="font-size:18px;">NQ-Tables R@10</span>',
         S_CALLOUT, C1X,            cy, bw2, 155)
    cell(f'<b style="font-size:26px;color:{MAROON};">89.88</b><br/>'
         '<span style="font-size:18px;">OTT-QA R@10</span>',
         S_CALLOUT, C1X + bw2+12,   cy, bw2, 155)
    cell(f'<b style="font-size:26px;color:{MAROON};">70%+</b><br/>'
         '<span style="font-size:18px;">Token Reduction</span>',
         S_CALLOUT, C1X + 2*(bw2+12), cy, bw2, 155)

# ── COLUMN 2 ─────────────────────────────────────────────────────────────────

y2 = 300

# Preprocessing Pipeline
y2 = section(
    'Preprocessing Pipeline',
    f'<div style="font-size:19px;padding:4px;line-height:1.7">'
    f'<b style="color:{MAROON};font-size:20px;">&#9312; Query Expansion</b><br/>'
    f'&nbsp;&nbsp;<span style="background:#E8F5E9;border:1px solid #2E7D32;'
    f'border-radius:3px;padding:1px 5px;">Query <i>q</i></span>'
    f' &#8594; <span style="background:#E8F5E9;border:1px solid #2E7D32;'
    f'border-radius:3px;padding:1px 5px;font-weight:bold;">Gemini 1.5-Flash</span>'
    f' &#8594; Sub-questions <i>q&#8321;, q&#8322;, ...</i><br/><br/>'
    f'<b style="color:{MAROON};font-size:20px;">&#9313; Table Enrichment</b><br/>'
    f'&nbsp;&nbsp;<span style="background:#E3F2FD;border:1px solid #1565C0;'
    f'border-radius:3px;padding:1px 5px;">Raw Table</span>'
    f' &#8594; <span style="background:#E8F5E9;border:1px solid #2E7D32;'
    f'border-radius:3px;padding:1px 5px;font-weight:bold;">Gemini 1.5-Flash</span>'
    f' &#8594; Title + Summary<br/><br/>'
    f'<b style="color:{MAROON};font-size:20px;">&#9314; Row Ranking</b><br/>'
    f'&nbsp;&nbsp;Rows ranked by semantic relevance to query using '
    f'<i>all-mpnet-base-v2</i><br/>'
    f'&nbsp;&nbsp;Top rows retained &#8594; <b>Mini-Table</b> '
    f'(reduces tokens by <b>70%+</b>)<br/><br/>'
    f'<b style="color:{MAROON};font-size:20px;">&#9315; Cascaded Retrieval</b><br/>'
    f'&nbsp;&nbsp;<span style="background:#FF6B35;color:#FFF;border-radius:3px;'
    f'padding:1px 6px;font-weight:bold;">SPLADE</span> (sparse)'
    f' &#8594; <span style="background:#0D7CC5;color:#FFF;border-radius:3px;'
    f'padding:1px 6px;font-weight:bold;">all-mpnet</span> (dense)'
    f' &#8594; <span style="background:#2E7D32;color:#FFF;border-radius:3px;'
    f'padding:1px 6px;font-weight:bold;">text-emb-3</span> (neural)<br/>'
    f'&nbsp;&nbsp;<span style="font-size:17px;color:#666;">'
    f'All tables &nbsp;&#8594;&nbsp; 5,000 &nbsp;&#8594;&nbsp; 100 &nbsp;&#8594;&nbsp; top-k</span>'
    f'</div>',
    C2X, y2, CW, 520
)

# NQ-Tables Results
y2 = section(
    'NQ-Tables Retrieval Results',
    f'<table style="border-collapse:collapse;width:100%;font-size:20px;">'
    f'<tr style="background:{MAROON};color:#FFF;">'
    f'<td style="padding:6px 10px;"><b>Model</b></td>'
    f'<td style="padding:6px;text-align:center;"><b>Type</b></td>'
    f'<td style="padding:6px;text-align:center;"><b>R@1</b></td>'
    f'<td style="padding:6px;text-align:center;"><b>R@10</b></td>'
    f'<td style="padding:6px;text-align:center;"><b>R@50</b></td>'
    f'</tr>'
    f'<tr style="background:#FFF;">'
    f'<td style="padding:5px 10px;">BM25</td>'
    f'<td style="padding:5px;text-align:center;">Sparse</td>'
    f'<td style="padding:5px;text-align:center;">47.84</td>'
    f'<td style="padding:5px;text-align:center;">72.90</td>'
    f'<td style="padding:5px;text-align:center;">89.19</td></tr>'
    f'<tr style="background:#F5F5F5;">'
    f'<td style="padding:5px 10px;">SPLADE</td>'
    f'<td style="padding:5px;text-align:center;">Sparse</td>'
    f'<td style="padding:5px;text-align:center;">62.11</td>'
    f'<td style="padding:5px;text-align:center;">83.33</td>'
    f'<td style="padding:5px;text-align:center;">94.20</td></tr>'
    f'<tr style="background:#FFF;">'
    f'<td style="padding:5px 10px;">BIBERT</td>'
    f'<td style="padding:5px;text-align:center;">Dense</td>'
    f'<td style="padding:5px;text-align:center;">59.40</td>'
    f'<td style="padding:5px;text-align:center;">82.25</td>'
    f'<td style="padding:5px;text-align:center;">93.50</td></tr>'
    f'<tr style="background:#F5F5F5;">'
    f'<td style="padding:5px 10px;">DTR</td>'
    f'<td style="padding:5px;text-align:center;">Dense</td>'
    f'<td style="padding:5px;text-align:center;">51.20</td>'
    f'<td style="padding:5px;text-align:center;">75.86</td>'
    f'<td style="padding:5px;text-align:center;">90.60</td></tr>'
    f'<tr style="background:#FFF;">'
    f'<td style="padding:5px 10px;">THYME</td>'
    f'<td style="padding:5px;text-align:center;">Hybrid</td>'
    f'<td style="padding:5px;text-align:center;">50.30</td>'
    f'<td style="padding:5px;text-align:center;">73.28</td>'
    f'<td style="padding:5px;text-align:center;">88.75</td></tr>'
    f'<tr style="background:#FFF8E1;">'
    f'<td style="padding:5px 10px;font-weight:bold;border-left:3px solid {GOLD};">'
    f'<b>CRAFT (ours)</b></td>'
    f'<td style="padding:5px;text-align:center;font-weight:bold;">Hybrid</td>'
    f'<td style="padding:5px;text-align:center;font-weight:bold;">49.84</td>'
    f'<td style="padding:5px;text-align:center;font-weight:bold;color:{MAROON};">86.83 &#9733;</td>'
    f'<td style="padding:5px;text-align:center;font-weight:bold;">97.17</td></tr>'
    f'</table>',
    C2X, y2, CW, 390
)

# Callout
cell(
    f'<b>CRAFT R@10: 86.83</b><br/>'
    f'<span style="font-size:19px;">+3.5 pp over SPLADE (83.33) — no training required</span>',
    S_CALLOUT, C2X, y2, CW, 80
)
y2 += 80 + 20

# NQ Context-Accuracy Plot
cell('Context Length vs. Accuracy (NQ-Tables)', S_SEC_HDR, C2X, y2, CW, 44)
y2 += 44 + 8
nq_plot_h = round(CW * 658 / 2485)
img_cell('Context_Accuracy_F1_nq_gemini (1).png', 'image/png', C2X, y2, CW, nq_plot_h)
y2 += nq_plot_h + 20

# Ablation Studies
y2 = section(
    'Ablation Studies',
    f'<div style="font-size:19px;padding:2px;">'
    f'<b style="color:{MAROON};">Stage Contribution (NQ-Tables)</b>'
    f'<table style="border-collapse:collapse;width:100%;margin-top:6px;font-size:19px;">'
    f'<tr style="background:{MAROON};color:#FFF;">'
    f'<td style="padding:4px 8px;"><b>Configuration</b></td>'
    f'<td style="padding:4px 8px;text-align:center;width:110px;"><b>R@10</b></td></tr>'
    f'<tr style="background:#F5F5F5;">'
    f'<td style="padding:4px 8px;">Stage 1: SPLADE</td>'
    f'<td style="padding:4px 8px;text-align:center;">83.33</td></tr>'
    f'<tr><td style="padding:4px 8px;">Stage 1 + Stage 2</td>'
    f'<td style="padding:4px 8px;text-align:center;">82.91</td></tr>'
    f'<tr style="background:#FFF8E1;font-weight:bold;">'
    f'<td style="padding:4px 8px;"><b>Full CRAFT (S1+S2+S3)</b></td>'
    f'<td style="padding:4px 8px;text-align:center;"><b>86.83</b></td></tr>'
    f'</table>'
    f'<br/><b style="color:{MAROON};">Preprocessing Ablation (NQ-Tables)</b>'
    f'<table style="border-collapse:collapse;width:100%;margin-top:6px;font-size:19px;">'
    f'<tr style="background:{MAROON};color:#FFF;">'
    f'<td style="padding:4px 8px;"><b>Configuration</b></td>'
    f'<td style="padding:4px 8px;text-align:center;width:110px;"><b>R@10</b></td></tr>'
    f'<tr style="background:#F5F5F5;">'
    f'<td style="padding:4px 8px;">No preprocessing (BM25 baseline)</td>'
    f'<td style="padding:4px 8px;text-align:center;">72.90</td></tr>'
    f'<tr><td style="padding:4px 8px;">+ Table summaries</td>'
    f'<td style="padding:4px 8px;text-align:center;">81.80</td></tr>'
    f'<tr style="background:#F5F5F5;">'
    f'<td style="padding:4px 8px;">+ Query expansion</td>'
    f'<td style="padding:4px 8px;text-align:center;">83.30</td></tr>'
    f'<tr style="background:#FFF8E1;font-weight:bold;">'
    f'<td style="padding:4px 8px;"><b>Full CRAFT preprocessing</b></td>'
    f'<td style="padding:4px 8px;text-align:center;"><b>86.83</b></td></tr>'
    f'</table></div>',
    C2X, y2, CW, 650
)

# End-to-End QA
y2 = section(
    'End-to-End QA Performance (NQ-Tables F1)',
    f'<div style="font-size:20px;padding:2px;">'
    f'Downstream QA F1 scores using top-n retrieved tables:<br/>'
    f'<table style="border-collapse:collapse;width:100%;margin-top:8px;font-size:19px;">'
    f'<tr style="background:{MAROON};color:#FFF;">'
    f'<td style="padding:5px 10px;"><b>Reader Model</b></td>'
    f'<td style="padding:5px;text-align:center;"><b>n=1</b></td>'
    f'<td style="padding:5px;text-align:center;"><b>n=3</b></td>'
    f'<td style="padding:5px;text-align:center;"><b>n=5</b></td></tr>'
    f'<tr style="background:#F5F5F5;">'
    f'<td style="padding:5px 10px;">Llama3-8B</td>'
    f'<td style="padding:5px;text-align:center;">39.13</td>'
    f'<td style="padding:5px;text-align:center;">39.50</td>'
    f'<td style="padding:5px;text-align:center;">40.55</td></tr>'
    f'<tr><td style="padding:5px 10px;">Mistral-7B</td>'
    f'<td style="padding:5px;text-align:center;">35.30</td>'
    f'<td style="padding:5px;text-align:center;">41.20</td>'
    f'<td style="padding:5px;text-align:center;">44.53</td></tr>'
    f'<tr style="background:#FFF8E1;font-weight:bold;">'
    f'<td style="padding:5px 10px;"><b>Qwen2.5-7B</b></td>'
    f'<td style="padding:5px;text-align:center;">37.01</td>'
    f'<td style="padding:5px;text-align:center;">43.85</td>'
    f'<td style="padding:5px;text-align:center;color:{MAROON};"><b>46.49 &#9733;</b></td></tr>'
    f'</table>'
    f'<span style="font-size:17px;color:#555;">Mini-tables enable larger context windows, '
    f'boosting E2E F1 as n increases.</span>'
    f'</div>',
    C2X, y2, CW, 360
)

# ── COLUMN 3 ─────────────────────────────────────────────────────────────────

y3 = 300

# OTT-QA Results
y3 = section(
    'Zero-Shot OTT-QA Results',
    f'<table style="border-collapse:collapse;width:100%;font-size:20px;">'
    f'<tr style="background:{MAROON};color:#FFF;">'
    f'<td style="padding:6px 10px;"><b>Model</b></td>'
    f'<td style="padding:6px;text-align:center;"><b>Type</b></td>'
    f'<td style="padding:6px;text-align:center;"><b>R@10</b></td>'
    f'<td style="padding:6px;text-align:center;"><b>R@50</b></td>'
    f'<td style="padding:6px;text-align:center;"><b>Zero-shot</b></td></tr>'
    f'<tr style="background:#FFF;">'
    f'<td style="padding:5px 10px;">BM25</td>'
    f'<td style="padding:5px;text-align:center;">Sparse</td>'
    f'<td style="padding:5px;text-align:center;">51.94</td>'
    f'<td style="padding:5px;text-align:center;">82.10</td>'
    f'<td style="padding:5px;text-align:center;">&#10003;</td></tr>'
    f'<tr style="background:#F5F5F5;">'
    f'<td style="padding:5px 10px;">SPLADE</td>'
    f'<td style="padding:5px;text-align:center;">Sparse</td>'
    f'<td style="padding:5px;text-align:center;">89.52</td>'
    f'<td style="padding:5px;text-align:center;">97.50</td>'
    f'<td style="padding:5px;text-align:center;">&#10003;</td></tr>'
    f'<tr style="background:#FFF;">'
    f'<td style="padding:5px 10px;">BIBERT</td>'
    f'<td style="padding:5px;text-align:center;">Dense</td>'
    f'<td style="padding:5px;text-align:center;">86.50</td>'
    f'<td style="padding:5px;text-align:center;">95.40</td>'
    f'<td style="padding:5px;text-align:center;">&#10007;</td></tr>'
    f'<tr style="background:#FFF8E1;">'
    f'<td style="padding:5px 10px;font-weight:bold;border-left:3px solid {GOLD};">'
    f'<b>CRAFT (ours)</b></td>'
    f'<td style="padding:5px;text-align:center;font-weight:bold;">Hybrid</td>'
    f'<td style="padding:5px;text-align:center;font-weight:bold;color:{MAROON};">89.88 &#9733;</td>'
    f'<td style="padding:5px;text-align:center;font-weight:bold;">96.07</td>'
    f'<td style="padding:5px;text-align:center;font-weight:bold;">&#10003;</td></tr>'
    f'</table>'
    f'<div style="margin-top:8px;font-size:18px;color:#555;">'
    f'Evaluated on 419K-table corpus, 2,214 dev queries. No fine-tuning on OTT-QA.</div>',
    C3X, y3, CW, 390
)

# OTT-QA callout
cell(
    f'<b>CRAFT R@10: 89.88</b><br/>'
    f'<span style="font-size:19px;">Best zero-shot model — beats supervised BIBERT</span>',
    S_CALLOUT, C3X, y3, CW, 80
)
y3 += 80 + 20

# OTT-QA Plot
cell('Context Length vs. Accuracy (OTT-QA)', S_SEC_HDR, C3X, y3, CW, 44)
y3 += 44 + 8
ottqa_h = round(CW * 762 / 2476)
img_cell('Context_Accuracy_F1_ottqa_large.png', 'image/png', C3X, y3, CW, ottqa_h)
y3 += ottqa_h + 20

# Token Efficiency
cell('Token Efficiency via Mini-Tables', S_SEC_HDR, C3X, y3, CW, 44)
y3 += 44 + 8
tok_w = 720
tok_h = round(tok_w * 799 / 808)
tok_x = C3X + (CW - tok_w) // 2
img_cell('token_consumption_table.png', 'image/png', tok_x, y3, tok_w, tok_h)
y3 += tok_h + 8
cell(
    f'<i style="font-size:18px;">Mini-table representation reduces token consumption '
    f'by <b>70%+</b> (e.g. Mistral: 3,314 → 477 tokens for n=1), enabling '
    f'larger context windows and higher E2E F1.</i>',
    'text;html=1;align=center;verticalAlign=top;strokeColor=none;fillColor=none;fontSize=18;',
    C3X, y3, CW, 68
)
y3 += 76 + 20

# Conclusion
y3 = section(
    'Conclusion',
    f'<ul style="font-size:20px;margin:4px 0 0 -16px;line-height:1.7">'
    f'<li>CRAFT is the first <b>training-free, three-stage cascaded</b> retrieval '
    f'pipeline for Open-Domain TQA.</li>'
    f'<li>Achieves <b style="color:{MAROON};">SOTA R@10 = 86.83 on NQ-Tables</b>, '
    f'surpassing all supervised baselines with no fine-tuning.</li>'
    f'<li><b>Zero-shot generalisation</b> to OTT-QA with R@10 = 89.88, '
    f'competitive with supervised models.</li>'
    f'<li>Mini-tables reduce token usage by <b>70%+</b> and boost downstream '
    f'QA F1 through larger context windows.</li>'
    f'<li>Stage 3 acts as a <b>noise-robust re-ranker</b>, recovering full '
    f'performance under query perturbation.</li>'
    f'</ul>',
    C3X, y3, CW, 520
)

# Future Work
y3 = section(
    'Future Work',
    f'<ul style="font-size:20px;margin:4px 0 0 -16px;line-height:1.7">'
    f'<li>Extend to <b>multi-hop TQA</b> requiring joins across multiple tables</li>'
    f'<li>Apply CRAFT to <b>heterogeneous corpora</b> (tables + passages)</li>'
    f'<li>Explore <b>lightweight stage-2 models</b> for lower latency deployment</li>'
    f'<li>Investigate table-aware <b>chunking strategies</b> for very wide tables</li>'
    f'</ul>',
    C3X, y3, CW, 330
)

# Key Contributions
y3 = section(
    'Key Contributions',
    f'<ol style="font-size:20px;margin:4px 0 0 -18px;line-height:1.75">'
    f'<li><b>Training-free cascaded pipeline</b> combining sparse + dense + neural '
    f'retrieval signals.</li>'
    f'<li><b>LLM preprocessing</b> via Gemini 1.5-Flash: query expansion + '
    f'table title/summary generation.</li>'
    f'<li><b>Mini-table representation</b> using relevance-ranked rows for '
    f'70%+ token reduction.</li>'
    f'<li>Extensive evaluation on <b>NQ-Tables &amp; OTT-QA</b> with ablations '
    f'confirming each component\'s contribution.</li>'
    f'</ol>',
    C3X, y3, CW, 420,
    body_style=S_BODY_GOLD
)

# ── FOOTER ────────────────────────────────────────────────────────────────────
cell('', S_FOOTER_BG, 0, FOOTER, W, 160)
cell(
    '[1] Herzig et al. (2021). <i>Open Domain QA over Tables via Dense Retrieval.</i> NAACL. &nbsp;|&nbsp; '
    '[2] Formal et al. (2021). <i>SPLADE: Sparse Lexical and Expansion Model.</i> SIGIR. &nbsp;|&nbsp; '
    '[3] Nguyen et al. (2022). <i>DTR: Dense Table Retrieval for Open Domain QA.</i> EMNLP. &nbsp;|&nbsp; '
    '[4] Chen et al. (2021). <i>OTT-QA: Open Table-and-Text QA.</i> ICLR.',
    S_FOOTER_TXT, 40, FOOTER + 14, 3000, 120
)
cell(
    f'<b style="color:{GOLD};font-size:22px;">ACL 2026 · Vienna</b><br/>'
    f'<span style="color:#AAAAAA;font-size:18px;">arXiv: 2505.14984</span>',
    'text;html=1;align=right;verticalAlign=top;strokeColor=none;fillColor=none;',
    3100, FOOTER + 18, 316, 90
)

# ── Assemble XML ──────────────────────────────────────────────────────────────
header = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<mxGraphModel dx="1422" dy="762" grid="0" gridSize="10" guides="1" '
    'tooltips="1" connect="1" arrows="1" fold="1" page="1" '
    f'pageScale="1" pageWidth="{W}" pageHeight="{H}" math="0" shadow="0">\n'
    '  <root>\n'
    '    <mxCell id="0"/>\n'
    '    <mxCell id="1" parent="0"/>\n'
)
xml = header + '\n'.join(_cells) + '\n  </root>\n</mxGraphModel>\n'

with open(OUT_FILE, 'w', encoding='utf-8') as f:
    f.write(xml)

size_kb = os.path.getsize(OUT_FILE) // 1024
print(f'Written: {OUT_FILE}  ({size_kb:,} KB, {len(_cells)} cells)')
