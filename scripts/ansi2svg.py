#!/usr/bin/env python3
"""Render ANSI-coloured terminal output (256-colour, bold, dim) to an SVG
"terminal window", for the README.   usage: moa doctor --color always | ansi2svg.py out.svg [title]"""
import re, sys, html

out = sys.argv[1]
title = sys.argv[2] if len(sys.argv) > 2 else "moa"
text = sys.stdin.read().rstrip("\n").replace("\t", "    ")


def x256(n):
    n = int(n)
    base = ["#1e1e1e", "#cd3131", "#0dbc79", "#e5e510", "#2472c8", "#bc3fbc", "#11a8cd", "#e5e5e5",
            "#666666", "#f14c4c", "#23d18b", "#f5f543", "#3b8eea", "#d670d6", "#29b8db", "#ffffff"]
    if n < 16:
        return base[n]
    if n < 232:
        n -= 16
        r, g, b = n // 36, (n // 6) % 6, n % 6
        c = [0, 95, 135, 175, 215, 255]
        return f"#{c[r]:02x}{c[g]:02x}{c[b]:02x}"
    v = 8 + (n - 232) * 10
    return f"#{v:02x}{v:02x}{v:02x}"


FG = "#d4d4d4"
lines = text.split("\n")
CW, LH, PAD, TOP = 8.43, 16.4, 18, 38
rows = []
for line in lines:
    spans, col, bold, dim, pos = [], FG, False, False, 0
    for m in re.finditer(r"\x1b\[([0-9;]*)m|([^\x1b]+)", line):
        if m.group(1) is not None:
            codes = [c for c in m.group(1).split(";") if c] or ["0"]
            i = 0
            while i < len(codes):
                c = codes[i]
                if c == "0":
                    col, bold, dim = FG, False, False
                elif c == "1":
                    bold = True
                elif c == "2":
                    dim = True
                elif c == "38" and i + 2 < len(codes) and codes[i + 1] == "5":
                    col = x256(codes[i + 2]); i += 2
                i += 1
        else:
            spans.append((pos, m.group(2), col, bold, dim)); pos += len(m.group(2))
    rows.append(spans)
width = int(max((len(re.sub(r"\x1b\[[0-9;]*m", "", l)) for l in lines), default=40) * CW + 2 * PAD)
height = int(len(lines) * LH + TOP + PAD)
o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
     '<style>text{font-family:"JetBrains Mono","DejaVu Sans Mono",Menlo,Consolas,monospace;font-size:14px;white-space:pre;dominant-baseline:alphabetic}</style>',
     f'<rect width="{width}" height="{height}" rx="10" fill="#14161a"/>',
     f'<rect width="{width}" height="28" rx="10" fill="#22252b"/><rect y="18" width="{width}" height="10" fill="#22252b"/>',
     '<circle cx="18" cy="14" r="6" fill="#ff5f56"/><circle cx="38" cy="14" r="6" fill="#ffbd2e"/><circle cx="58" cy="14" r="6" fill="#27c93f"/>',
     f'<text x="{width/2}" y="19" fill="#8b9099" text-anchor="middle" style="font-size:12px">{html.escape(title)}</text>']
for r, spans in enumerate(rows):
    y = TOP + (r + 1) * LH - 3.6
    for pos, s, col, bold, dim in spans:
        if not s.strip():
            continue
        lead = len(s) - len(s.lstrip(" "))
        attrs = f'fill="{col}"' + (' font-weight="700"' if bold else "") + (' opacity="0.6"' if dim else "")
        o.append(f'<text x="{PAD + (pos + lead) * CW:.1f}" y="{y}" {attrs} textLength="{len(s.strip(" ")) * CW:.1f}" lengthAdjust="spacingAndGlyphs">{html.escape(s.strip(" "))}</text>')
o.append("</svg>")
open(out, "w").write("\n".join(o))
print(out, width, "x", height)
