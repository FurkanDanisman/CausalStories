#!/usr/bin/env python3
"""Proposed smaller true graphs for the three benchmark worlds (structure only, no probabilities yet)."""
import subprocess
STYLE = '''graph [rankdir=LR, nodesep=0.35, ranksep=0.7, pad=0.3, fontname="Helvetica", labelloc=t, fontsize=18];
node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, margin="0.12,0.06", height=0.3];
edge [fontname="Helvetica", fontsize=9, arrowsize=0.7];
'''
# the colours of the results pages (results/run_6057822 :root tokens)
EVENT = 'fillcolor="#ffffff", color="#5c6763"'                              # .node: card fill, muted stroke
PART = 'fillcolor="#eaf0fa", color="#2d5fa8"'                               # .node.participant: part-bg, agent
ISO = 'fillcolor="#ffffff", color="#a86a08", style="rounded,filled,dashed"'  # .node.isolated: warn, dashed
ENABLES = 'color="#1f7a4d"'                                                 # .edge.enables
BLOCKS = 'color="#b4462f", style=dashed'                                    # .edge.blocks: block, dashed


def dot(name, title, cause, steps, outcome, blockers, distract, parts, en, bl, carry):
    L = [f'digraph G {{ {STYLE} label="{title}";']
    for n in cause + steps + [outcome] + blockers: L.append(f'"{n}" [{EVENT}];')
    for n in distract: L.append(f'"{n}" [{ISO}];')
    for n in parts: L.append(f'"{n}" [{PART}];')
    for a, b in en + carry: L.append(f'"{a}" -> "{b}" [{ENABLES}];')
    for a, b in bl: L.append(f'"{a}" -> "{b}" [{BLOCKS}];')
    L.append("}")
    for fmt in ("png", "pdf"):
        subprocess.run(["dot", f"-T{fmt}", "-Gdpi=160", "-o", f"{name}.{fmt}"], input="\n".join(L), text=True, check=True)



dot("1_homelessness", "1. Homelessness",
    ["house fire", "divorce"], ["job loss", "can't pay rent", "eviction"], "homelessness",
    ["savings", "providing a place to stay"], ["getting a pet", "taking up running"],
    ["EMPLOYER", "FAMILY MEMBER"],
    [("job loss", "can't pay rent"), ("divorce", "can't pay rent"), ("can't pay rent", "eviction"), ("eviction", "homelessness"), ("house fire", "homelessness")],
    [("savings", "can't pay rent"), ("savings", "homelessness"), ("providing a place to stay", "homelessness")],
    [("EMPLOYER", "job loss"), ("FAMILY MEMBER", "providing a place to stay")])
dot("2_career_success", "2. Career success",
    ["degree"], ["receiving guidance", "gaining skills", "getting noticed", "strong performance"], "promotion",
    ["facing discrimination", "layoffs"], ["moving apartment", "buying a car"],
    ["MENTOR", "COMPANY"],
    [("degree", "gaining skills"), ("receiving guidance", "gaining skills"), ("receiving guidance", "getting noticed"),
     ("gaining skills", "strong performance"), ("strong performance", "promotion"), ("getting noticed", "promotion")],
    [("facing discrimination", "promotion"), ("layoffs", "promotion")],
    [("MENTOR", "receiving guidance"), ("COMPANY", "layoffs")])
dot("3_mental_health", "3. Mental health (outcome: depression)",
    ["job loss", "losing a loved one"], ["loneliness", "chronic stress"], "depression",
    ["social support", "therapy"], ["new phone", "learning to cook"],
    ["EMPLOYER", "CLOSE FRIENDS", "THERAPIST"],
    [("job loss", "chronic stress"), ("losing a loved one", "loneliness"), ("loneliness", "chronic stress"),
     ("chronic stress", "depression")],
    [("social support", "loneliness"), ("therapy", "depression")],
    [("EMPLOYER", "job loss"), ("THERAPIST", "therapy"), ("CLOSE FRIENDS", "social support")])
print("ok")
