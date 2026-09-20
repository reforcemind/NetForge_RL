"""Sticky-note SVGs + Excalidraw sources (FlowEdge visual language)."""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent

INK = '#2B2118'
TITLE = '#1B4B6B'
CANVAS = '#FFF6DC'
EDGE = '#E8D7A8'
SHADOW = '#E8C98A'
YEL = '#FFD166'
TEAL = '#2EC4B6'
PINK = '#FF8FAB'
ORANGE = '#FF9F43'
PURPLE = '#7B6CF6'
GREEN = '#6BCB77'
CREAM = '#FFF1C2'
BLUE = '#8EC5FF'
RED = '#FF6B6B'
FONT = 'Trebuchet MS, Segoe UI, sans-serif'


def header(title: str, w: int, h: int, mid: str | None = None) -> str:
    extra = ''
    if mid:
        extra = (
            f'<text x="{w // 2}" y="58" text-anchor="middle" font-size="16" '
            f'font-family="{FONT}" font-weight="800" fill="{TITLE}">{mid}</text>'
        )
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img">
<title>{title}</title>
<defs>
<marker id="{_slug(title)}-arr" markerWidth="10" markerHeight="10" refX="8" refY="3.2" orient="auto"><path d="M0,0 L10,3.2 L0,6.4 Z" fill="{INK}"/></marker>
</defs>
<rect width="100%" height="100%" rx="22" fill="{CANVAS}" stroke="{EDGE}" stroke-width="3"/>
<text x="28" y="38" font-size="22" font-family="{FONT}" font-weight="800" fill="{TITLE}">{title}</text>
{extra}'''


def _slug(title: str) -> str:
    return ''.join(ch if ch.isalnum() else '-' for ch in title.lower()).strip('-')[:24]


def close() -> str:
    return '</svg>\n'


def card(
    x: float, y: float, w: float, h: float, fill: str, rot: float, inner: str
) -> str:
    cx = x + w / 2
    cy = y + h / 2
    return (
        f'<g transform="rotate({rot:.2f} {cx:.0f} {cy:.0f})">'
        f'<rect x="{x + 3:.0f}" y="{y + 5:.0f}" width="{w:.0f}" height="{h:.0f}" rx="16" fill="{SHADOW}" opacity="0.45"/>'
        f'<rect x="{x:.0f}" y="{y:.0f}" width="{w:.0f}" height="{h:.0f}" rx="16" fill="{fill}" stroke="{INK}" stroke-width="2.6"/>'
        f'{inner}</g>\n'
    )


def xml_escape(text: str) -> str:
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def t(
    x: float,
    y: float,
    size: int,
    weight: str,
    fill: str,
    text: str,
    anchor: str = 'middle',
) -> str:
    return (
        f'<text x="{x:.0f}" y="{y:.0f}" text-anchor="{anchor}" font-size="{size}" '
        f'font-family="{FONT}" font-weight="{weight}" fill="{fill}">{xml_escape(text)}</text>'
    )


def chip(
    x: float, y: float, w: float, h: float, fill: str, rot: float, label: str
) -> str:
    cx = x + w / 2
    cy = y + h / 2
    return (
        f'<g transform="rotate({rot:.2f} {cx:.0f} {cy:.0f})">'
        f'<rect x="{x:.0f}" y="{y:.0f}" width="{w:.0f}" height="{h:.0f}" rx="9" fill="{fill}" stroke="{INK}" stroke-width="2.2"/>'
        f'{t(cx, y + 24, 14, "800", INK, label)}</g>\n'
    )


def arrow(
    x1: float, y1: float, x2: float, y2: float, title: str, qy: float | None = None
) -> str:
    mid = (x1 + x2) / 2
    q = y1 if qy is None else qy
    return (
        f'<path d="M {x1:.0f} {y1:.0f} Q {mid:.0f} {q:.0f} {x2:.0f} {y2:.0f}" fill="none" '
        f'stroke="{TITLE}" stroke-width="2.8" marker-end="url(#{_slug(title)}-arr)"/>\n'
    )


def write(name: str, body: str) -> None:
    path = OUT / name
    path.write_text(body, encoding='utf-8')
    print('wrote', path.name, path.stat().st_size)


# ---------------------------------------------------------------------------
# Excalidraw
# ---------------------------------------------------------------------------

_EID = 0


def eid(prefix: str = 'e') -> str:
    global _EID
    _EID += 1
    return f'{prefix}{_EID:04d}'


def e_rect(x, y, w, h, bg, angle=0.0, seed=1):
    return {
        'id': eid('r'),
        'type': 'rectangle',
        'x': x,
        'y': y,
        'width': w,
        'height': h,
        'angle': angle,
        'strokeColor': INK,
        'backgroundColor': bg,
        'fillStyle': 'solid',
        'strokeWidth': 2,
        'strokeStyle': 'solid',
        'roughness': 1,
        'opacity': 100,
        'groupIds': [],
        'frameId': None,
        'roundness': {'type': 3},
        'seed': seed,
        'version': 1,
        'versionNonce': seed + 7,
        'isDeleted': False,
        'boundElements': None,
        'updated': 1,
        'link': None,
        'locked': False,
    }


def e_text(x, y, w, h, text, color=INK, size=20, align='center', seed=2):
    return {
        'id': eid('t'),
        'type': 'text',
        'x': x,
        'y': y,
        'width': w,
        'height': h,
        'angle': 0,
        'strokeColor': color,
        'backgroundColor': 'transparent',
        'fillStyle': 'solid',
        'strokeWidth': 1,
        'strokeStyle': 'solid',
        'roughness': 0,
        'opacity': 100,
        'groupIds': [],
        'frameId': None,
        'roundness': None,
        'seed': seed,
        'version': 1,
        'versionNonce': seed + 3,
        'isDeleted': False,
        'boundElements': None,
        'updated': 1,
        'link': None,
        'locked': False,
        'text': text,
        'fontSize': size,
        'fontFamily': 2,
        'textAlign': align,
        'verticalAlign': 'middle',
        'containerId': None,
        'originalText': text,
        'autoResize': True,
        'lineHeight': 1.25,
    }


def e_arrow(x, y, w, h, seed=3):
    return {
        'id': eid('a'),
        'type': 'arrow',
        'x': x,
        'y': y,
        'width': w,
        'height': h,
        'angle': 0,
        'strokeColor': TITLE,
        'backgroundColor': 'transparent',
        'fillStyle': 'solid',
        'strokeWidth': 2,
        'strokeStyle': 'solid',
        'roughness': 1,
        'opacity': 100,
        'groupIds': [],
        'frameId': None,
        'roundness': {'type': 2},
        'seed': seed,
        'version': 1,
        'versionNonce': seed + 9,
        'isDeleted': False,
        'boundElements': None,
        'updated': 1,
        'link': None,
        'locked': False,
        'startBinding': None,
        'endBinding': None,
        'lastCommittedPoint': None,
        'startArrowhead': None,
        'endArrowhead': 'arrow',
        'points': [[0, 0], [w, h]],
    }


def scene(elements: list, w: int = 1180, h: int = 400) -> dict:
    return {
        'type': 'excalidraw',
        'version': 2,
        'source': 'https://excalidraw.com',
        'elements': elements,
        'appState': {
            'gridSize': None,
            'viewBackgroundColor': CANVAS,
            'currentItemFontFamily': 2,
            'width': w,
            'height': h,
        },
        'files': {},
    }


def write_ex(name: str, elements: list, w: int = 1180, h: int = 400) -> None:
    path = OUT / name
    path.write_text(json.dumps(scene(elements, w, h), indent=2), encoding='utf-8')
    print('wrote', path.name, path.stat().st_size)


# ---------------------------------------------------------------------------
# Diagrams
# ---------------------------------------------------------------------------


def mark_svg() -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" width="120" height="120" role="img">
<title>NetForge</title>
<rect width="120" height="120" rx="22" fill="{CANVAS}" stroke="{EDGE}" stroke-width="4"/>
<circle cx="38" cy="46" r="11" fill="{YEL}" stroke="{INK}" stroke-width="2.4"/>
<circle cx="82" cy="42" r="11" fill="{TEAL}" stroke="{INK}" stroke-width="2.4"/>
<circle cx="60" cy="78" r="11" fill="{PINK}" stroke="{INK}" stroke-width="2.4"/>
<line x1="47" y1="50" x2="73" y2="46" stroke="{INK}" stroke-width="2.4"/>
<line x1="44" y1="55" x2="54" y2="70" stroke="{INK}" stroke-width="2.4"/>
<line x1="75" y1="52" x2="66" y2="70" stroke="{INK}" stroke-width="2.4"/>
</svg>
'''


def purpose():
    title = 'the match'
    parts = [
        header(title, 1180, 400, 'Red attacks  -  Blue defends  -  SIEM in the middle')
    ]
    parts.append(
        card(
            36,
            78,
            350,
            220,
            RED,
            -0.08,
            t(211, 156, 24, '800', TITLE, 'Red')
            + t(211, 188, 16, '600', INK, 'discover, exploit')
            + t(211, 208, 16, '600', INK, 'escalate, exfil')
            + t(211, 228, 16, '600', INK, 'fog of war')
            + t(211, 266, 14, '600', TITLE, 'starts blind'),
        )
    )
    parts.append(
        card(
            414,
            78,
            350,
            220,
            TEAL,
            0.24,
            t(589, 156, 24, '800', TITLE, 'Network')
            + t(589, 188, 16, '600', INK, 'hosts, subnets')
            + t(589, 208, 16, '600', INK, 'firewalls, decoys')
            + t(589, 228, 16, '600', INK, 'durative actions')
            + t(589, 266, 14, '600', TITLE, 'the gym world'),
        )
    )
    parts.append(
        card(
            792,
            78,
            350,
            220,
            BLUE,
            0.10,
            t(967, 156, 24, '800', TITLE, 'Blue')
            + t(967, 188, 16, '600', INK, 'isolate, restore')
            + t(967, 208, 16, '600', INK, 'decoys, EDR')
            + t(967, 228, 16, '600', INK, 'reads SIEM only')
            + t(967, 266, 14, '600', TITLE, 'no oracle'),
        )
    )
    parts.append(chip(70, 312, 148, 36, YEL, 0.9, 'ransomware'))
    parts.append(chip(470, 312, 174, 36, ORANGE, -1.4, 'APT / cloud / OT'))
    parts.append(chip(848, 312, 174, 36, PINK, -1.4, '1 Red + 3 Blue'))
    parts.append(
        t(590, 378, 15, '800', TITLE, 'PettingZoo parallel  ->  or Gymnasium Blue-v4')
    )
    parts.append(close())
    write('purpose.svg', ''.join(parts))
    write_ex(
        'purpose.excalidraw',
        [
            e_text(28, 12, 280, 32, 'the match', TITLE, 28, 'left', 11),
            e_text(
                240,
                44,
                700,
                24,
                'Red attacks - Blue defends - SIEM in the middle',
                TITLE,
                18,
                'center',
                12,
            ),
            e_rect(36, 78, 350, 220, RED, -0.01, 21),
            e_text(36, 150, 350, 40, 'Red', TITLE, 28, 'center', 22),
            e_text(
                36,
                190,
                350,
                80,
                'discover, exploit\nescalate, exfil\nstarts blind',
                INK,
                18,
                'center',
                23,
            ),
            e_rect(414, 78, 350, 220, TEAL, 0.004, 31),
            e_text(414, 150, 350, 40, 'Network', TITLE, 28, 'center', 32),
            e_text(
                414,
                190,
                350,
                80,
                'hosts, subnets\nfirewalls, decoys\ndurative actions',
                INK,
                18,
                'center',
                33,
            ),
            e_rect(792, 78, 350, 220, BLUE, 0.002, 41),
            e_text(792, 150, 350, 40, 'Blue', TITLE, 28, 'center', 42),
            e_text(
                792,
                190,
                350,
                80,
                'isolate, restore\nreads SIEM only\nno oracle',
                INK,
                18,
                'center',
                43,
            ),
        ],
        1180,
        400,
    )


def workflow():
    title = 'one tick'
    parts = [header(title, 980, 270)]
    boxes = [
        (24, 78, 155, 100, YEL, -0.13, 'reset', 'seeded net'),
        (210, 78, 165, 100, TEAL, -0.07, 'observe', 'SIEM + mask'),
        (406, 78, 185, 100, PURPLE, -0.15, 'act', 'Red / Blue'),
        (622, 78, 155, 100, ORANGE, -0.25, 'resolve', 'conflicts'),
        (808, 78, 148, 100, GREEN, -0.17, 'log', 'Sysmon-like'),
    ]
    for x, y, w, h, fill, rot, a, b in boxes:
        cx = x + w / 2
        parts.append(
            card(
                x,
                y,
                w,
                h,
                fill,
                rot,
                t(cx, 118, 16, '800', INK, a) + t(cx, 140, 14, '600', INK, b),
            )
        )
    parts.append(arrow(179, 128, 210, 128, title, 133))
    parts.append(arrow(375, 128, 406, 128, title, 126))
    parts.append(arrow(591, 128, 622, 128, title, 128))
    parts.append(arrow(777, 128, 808, 128, title, 129))
    parts.append(
        chip(
            220,
            200,
            527,
            36,
            BLUE,
            -1.5,
            'exploits and isolates take time  -  logs can be late',
        )
    )
    parts.append(close())
    write('workflow.svg', ''.join(parts))
    els = [e_text(28, 12, 280, 32, 'one tick', TITLE, 28, 'left', 61)]
    xs = [24, 210, 406, 622, 808]
    fills = [YEL, TEAL, PURPLE, ORANGE, GREEN]
    labels = [
        ('reset', 'seeded net'),
        ('observe', 'SIEM + mask'),
        ('act', 'Red / Blue'),
        ('resolve', 'conflicts'),
        ('log', 'Sysmon-like'),
    ]
    ws = [155, 165, 185, 155, 148]
    for i, (x, fill, (a, b), w) in enumerate(zip(xs, fills, labels, ws)):
        els += [
            e_rect(x, 78, w, 100, fill, 0, 70 + i),
            e_text(x, 100, w, 28, a, INK, 20, 'center', 80 + i),
            e_text(x, 132, w, 24, b, INK, 16, 'center', 90 + i),
        ]
        if i:
            els.append(
                e_arrow(
                    xs[i - 1] + ws[i - 1], 128, x - (xs[i - 1] + ws[i - 1]), 0, 100 + i
                )
            )
    els += [
        e_rect(220, 200, 527, 36, BLUE, -0.026, 110),
        e_text(
            220,
            202,
            527,
            32,
            'exploits and isolates take time - logs can be late',
            INK,
            16,
            'center',
            111,
        ),
    ]
    write_ex('workflow.excalidraw', els, 980, 270)


def why_this():
    title = 'what Blue actually sees'
    rows = [
        (
            YEL,
            'true compromise map',
            'simulator state',
            TEAL,
            'SIEM belief graph',
            'alerts + inventory',
        ),
        (
            PINK,
            'instant telemetry',
            'every exploit is obvious',
            GREEN,
            'delayed / noisy logs',
            'log_latency, drop, decoys',
        ),
        (
            ORANGE,
            'one giant isolate',
            'kill the network to win',
            PURPLE,
            'SLA vs security',
            'keep hosts online',
        ),
        (
            RED,
            'scripted campaign only',
            'one attacker forever',
            TEAL,
            'Red population',
            'random / heuristic / kill-chain',
        ),
        (
            PINK,
            'discrete instant acts',
            'done in one step',
            GREEN,
            'durative actions',
            'ticks until complete',
        ),
    ]
    parts = [header(title, 980, 570)]
    y = 72
    for i, (lc, lt, ls, rc, rt, rs) in enumerate(rows):
        parts.append(
            card(
                36,
                y,
                380,
                70,
                lc,
                0.04 if i % 2 == 0 else -0.12,
                t(226, y + 25, 16, '800', INK, lt) + t(226, y + 47, 14, '600', INK, ls),
            )
        )
        parts.append(arrow(428, y + 35, 534, y + 35, title, y + 32))
        parts.append(
            card(
                544,
                y,
                400,
                70,
                rc,
                -0.08 if i % 2 == 0 else 0.06,
                t(744, y + 25, 16, '800', INK, rt) + t(744, y + 47, 14, '600', INK, rs),
            )
        )
        y += 78
    parts.append(chip(36, 472, 180, 36, CREAM, -1.4, 'SOC view'))
    parts.append(chip(248, 472, 200, 36, CREAM, -0.8, 'not god-mode'))
    parts.append(chip(532, 472, 226, 36, YEL, -1.2, 'keep the lights on'))
    parts.append(
        t(
            490,
            540,
            14,
            '800',
            TITLE,
            'Blue trains on logs   -   Red trains on recon   -   the rest is hidden',
        )
    )
    parts.append(close())
    write('why-this.svg', ''.join(parts))
    els = [e_text(28, 12, 520, 32, 'what Blue actually sees', TITLE, 26, 'left', 200)]
    y = 72
    for i, (lc, lt, ls, rc, rt, rs) in enumerate(rows):
        els += [
            e_rect(36, y, 380, 70, lc, 0, 210 + i),
            e_text(36, y + 8, 380, 28, lt, INK, 18, 'center', 220 + i),
            e_text(36, y + 36, 380, 24, ls, INK, 14, 'center', 230 + i),
            e_arrow(416, y + 35, 128, 0, 240 + i),
            e_rect(544, y, 400, 70, rc, 0, 250 + i),
            e_text(544, y + 8, 400, 28, rt, INK, 18, 'center', 260 + i),
            e_text(544, y + 36, 400, 24, rs, INK, 14, 'center', 270 + i),
        ]
        y += 78
    write_ex('why-this.excalidraw', els, 980, 570)


def status():
    title = 'in the gym'
    parts = [header(title, 980, 520)]

    def pill(
        x: float, y: float, w: float, fill: str, mark: str, name: str, rot: float
    ) -> None:
        parts.append(card(x, y, w, 56, fill, rot, ''))
        parts.append(
            f'<circle cx="{x + 18:.0f}" cy="{y + 28:.0f}" r="11" fill="{fill}" stroke="{INK}" stroke-width="2"/>'
        )
        parts.append(t(x + 18, y + 33, 13, '800', INK, mark))
        parts.append(t(x + w / 2 + 10, y + 32, 13, '800', TITLE, name))

    parts.append(t(36, 68, 16, '800', TITLE, 'APIs', 'start'))
    pill(36, 80, 220, GREEN, '+', 'PettingZoo', 0.11)
    pill(268, 80, 250, GREEN, '+', 'Gymnasium Blue-v4', 0.15)
    pill(530, 80, 200, GREEN, '+', 'AEC wrapper', 0.08)

    parts.append(t(36, 160, 16, '800', TITLE, 'World', 'start'))
    pill(36, 172, 200, GREEN, '+', 'SIEM logs', 0.25)
    pill(248, 172, 210, GREEN, '+', 'belief graphs', 0.06)
    pill(470, 172, 200, GREEN, '+', 'decoys / OT', -0.05)
    pill(682, 172, 190, TEAL, '+', 'HTML replay', -0.20)

    parts.append(t(36, 252, 16, '800', TITLE, 'Agents', 'start'))
    pill(36, 264, 200, GREEN, '+', 'heuristics', 0.14)
    pill(248, 264, 220, GREEN, '+', 'kill-chain Red', 0.02)
    pill(480, 264, 210, TEAL, '+', 'scripted Blue', -0.11)

    parts.append(t(36, 344, 16, '800', TITLE, 'Your code', 'start'))
    pill(36, 356, 160, CREAM, 'x', 'MAPPO', 0.11)
    pill(208, 356, 150, CREAM, 'x', 'QMIX', -0.21)
    pill(370, 356, 190, CREAM, 'x', 'GNN trainer', -0.18)
    pill(572, 356, 200, CREAM, 'x', 'LLM agent', -0.08)

    parts.append(
        t(
            36,
            448,
            14,
            '800',
            TITLE,
            'entry:  gym.make / pettingzoo.make   -   netforge run / evaluate',
            'start',
        )
    )
    parts.append(
        t(
            36,
            478,
            14,
            '800',
            TITLE,
            '+ ships in this repo    x train elsewhere, run here',
            'start',
        )
    )
    parts.append(close())
    write('status.svg', ''.join(parts))
    els = [e_text(28, 12, 280, 32, 'in the gym', TITLE, 26, 'left', 300)]
    rows_e = [
        (
            80,
            'APIs',
            [
                (36, 220, 'PettingZoo', GREEN),
                (268, 250, 'Gymnasium Blue-v4', GREEN),
                (530, 200, 'AEC wrapper', GREEN),
            ],
        ),
        (
            172,
            'World',
            [
                (36, 200, 'SIEM logs', GREEN),
                (248, 210, 'belief graphs', GREEN),
                (470, 200, 'decoys / OT', GREEN),
                (682, 190, 'HTML replay', TEAL),
            ],
        ),
        (
            264,
            'Agents',
            [
                (36, 200, 'heuristics', GREEN),
                (248, 220, 'kill-chain Red', GREEN),
                (480, 210, 'scripted Blue', TEAL),
            ],
        ),
        (
            356,
            'Your code',
            [
                (36, 160, 'MAPPO', CREAM),
                (208, 150, 'QMIX', CREAM),
                (370, 190, 'GNN trainer', CREAM),
                (572, 200, 'LLM agent', CREAM),
            ],
        ),
    ]
    for y, lab, items in rows_e:
        els.append(e_text(36, y - 28, 200, 24, lab, TITLE, 18, 'left', y))
        for x, w, name, fill in items:
            els += [
                e_rect(x, y, w, 56, fill, 0, y + x),
                e_text(x, y + 12, w, 32, name, TITLE, 16, 'center', y + x + 1),
            ]
    write_ex('status.excalidraw', els, 980, 520)


def arena():
    title = 'eval splits'
    parts = [header(title, 1180, 340, 'train  /  dev  /  hidden   -  same metrics')]
    cols = [
        (36, YEL, 'train', 'public', 'procedural packs', 'learn here'),
        (414, TEAL, 'dev', 'public', 'held-out offset', 'pick a checkpoint'),
        (792, PINK, 'hidden', 'organizer', 'reserved + OOD', 'final check'),
    ]
    for x, fill, name, who, how, use in cols:
        cx = x + 175
        parts.append(
            card(
                x,
                78,
                350,
                180,
                fill,
                0.12 if fill == YEL else (-0.1 if fill == TEAL else 0.06),
                t(cx, 130, 28, '800', TITLE, name)
                + t(cx, 168, 16, '600', INK, who)
                + t(cx, 192, 16, '600', INK, how)
                + t(cx, 228, 14, '600', TITLE, use),
            )
        )
    parts.append(
        chip(
            200,
            274,
            780,
            36,
            BLUE,
            -0.7,
            'mission, SLA, false positives, kinetic  -  not return alone',
        )
    )
    parts.append(close())
    write('arena.svg', ''.join(parts))
    els = [
        e_text(28, 12, 280, 32, 'eval splits', TITLE, 28, 'left', 400),
        e_text(
            280,
            44,
            620,
            24,
            'train / dev / hidden - same metrics',
            TITLE,
            18,
            'center',
            401,
        ),
    ]
    for x, fill, name, who, how, use in cols:
        els += [
            e_rect(x, 78, 350, 180, fill, 0, 410 + x),
            e_text(x, 110, 350, 40, name, TITLE, 28, 'center', 411 + x),
            e_text(x, 160, 350, 70, f'{who}\n{how}\n{use}', INK, 18, 'center', 412 + x),
        ]
    write_ex('arena.excalidraw', els, 1180, 340)


def belief():
    title = 'SIEM vs full state'
    parts = [
        header(title, 980, 300, 'Blue trains on logs   -   full map is diagnostics')
    ]
    parts.append(
        card(
            36,
            78,
            420,
            160,
            YEL,
            -0.12,
            t(246, 130, 22, '800', TITLE, 'SIEM belief')
            + t(246, 162, 16, '600', INK, 'what a SOC would see')
            + t(246, 186, 16, '600', INK, 'partial, delayed, noisy')
            + t(246, 214, 14, '600', TITLE, 'default for Blue'),
        )
    )
    parts.append(arrow(468, 158, 524, 158, title, 150))
    parts.append(
        card(
            534,
            78,
            410,
            160,
            PINK,
            0.14,
            t(739, 130, 22, '800', TITLE, 'oracle graph')
            + t(739, 162, 16, '600', INK, 'true compromise')
            + t(739, 186, 16, '600', INK, 'debug / replay')
            + t(739, 214, 14, '600', TITLE, 'not a Blue obs'),
        )
    )
    parts.append(
        chip(220, 250, 540, 36, TEAL, -1.1, 'netforge questions belief-vs-oracle')
    )
    parts.append(close())
    write('belief.svg', ''.join(parts))
    write_ex(
        'belief.excalidraw',
        [
            e_text(28, 12, 400, 32, 'SIEM vs full state', TITLE, 26, 'left', 500),
            e_rect(36, 78, 420, 160, YEL, -0.002, 501),
            e_text(
                36,
                120,
                420,
                80,
                'SIEM belief\nwhat a SOC would see',
                INK,
                20,
                'center',
                502,
            ),
            e_arrow(456, 158, 78, 0, 503),
            e_rect(534, 78, 410, 160, PINK, 0.002, 504),
            e_text(
                534,
                120,
                410,
                80,
                'oracle graph\ntrue compromise, debug only',
                INK,
                20,
                'center',
                505,
            ),
        ],
        980,
        300,
    )


def main() -> None:
    (OUT / 'mark.svg').write_text(mark_svg(), encoding='utf-8')
    print('wrote mark.svg')
    purpose()
    workflow()
    why_this()
    status()
    arena()
    belief()


if __name__ == '__main__':
    main()
