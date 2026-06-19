import io

import matplotlib

matplotlib.use('Agg', force=True)

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from netforge_rl.render.snapshot import Snapshot


_LAYOUT_CACHE = {}


def _layout_for(snap):
    key = hash((snap.labels, snap.subnets, snap.edges))
    cached = _LAYOUT_CACHE.get(key)
    if cached is not None:
        return cached
    g = nx.Graph()
    g.add_nodes_from(range(snap.n_nodes))
    g.add_edges_from(snap.edges)
    layout = nx.spring_layout(g, seed=0)
    _LAYOUT_CACHE[key] = layout
    return layout


def render_rgb(snap: Snapshot, *, figsize=(6.0, 6.0), dpi=100, show_labels=False):
    """Render a snapshot to a uint8 H×W×3 array."""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    g = nx.Graph()
    g.add_nodes_from(range(snap.n_nodes))
    g.add_edges_from(snap.edges)
    pos = _layout_for(snap)

    nx.draw_networkx_edges(g, pos, ax=ax, alpha=0.4, width=0.8)
    nx.draw_networkx_nodes(
        g,
        pos,
        ax=ax,
        node_color=snap.colors,
        node_size=120,
        edgecolors='black',
        linewidths=0.5,
    )
    if show_labels:
        nx.draw_networkx_labels(
            g,
            pos,
            labels={i: s for i, s in enumerate(snap.labels)},
            font_size=6,
            ax=ax,
        )

    ax.set_title(f'NetForge — tick {snap.tick}', fontsize=10)
    ax.axis('off')
    fig.tight_layout(pad=0.2)

    buf = io.BytesIO()
    fig.savefig(buf, format='raw', dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    w, h = int(figsize[0] * dpi), int(figsize[1] * dpi)
    img = np.frombuffer(buf.getvalue(), dtype=np.uint8).reshape(h, w, 4)
    return img[..., :3].copy()
