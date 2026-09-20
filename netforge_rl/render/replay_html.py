from __future__ import annotations

import json
from pathlib import Path

from netforge_rl.core.graph_obs import build_graph_observation, build_oracle_graph
from netforge_rl.environment.constants import PADDING_SUBNET


class ReplayLog:
    """Network + SIEM + actions + rewards for the interactive viewer."""

    def __init__(self, scenario: str = '', seed: int = 0):
        self.scenario = scenario
        self.seed = seed
        self.ticks: list[dict] = []

    def capture(
        self, env, rewards: dict, infos: dict, actions: dict | None = None
    ) -> None:
        gs = (
            env.unwrapped.global_state
            if hasattr(env, 'unwrapped')
            else env.global_state
        )
        hosts = []
        for ip, host in sorted(gs.all_hosts.items()):
            if host.subnet_cidr == PADDING_SUBNET:
                continue
            hosts.append(
                {
                    'ip': ip,
                    'hostname': host.hostname,
                    'subnet': gs.get_subnet_name(host.subnet_cidr),
                    'status': host.status,
                    'compromised': host.compromised_by != 'None',
                    'isolated': host.status == 'isolated',
                    'decoy': host.decoy not in (None, 'inactive'),
                }
            )
        siem = []
        for entry in gs.siem_log_buffer[-12:]:
            log = entry[0] if isinstance(entry, tuple) else entry
            siem.append(str(log)[:240])
        sample = next((v for k, v in infos.items() if 'blue' in k), {})
        self.ticks.append(
            {
                'tick': int(
                    getattr(
                        env.unwrapped if hasattr(env, 'unwrapped') else env,
                        'current_tick',
                        0,
                    )
                ),
                'hosts': hosts,
                'siem': siem,
                'rewards': {k: float(v) for k, v in rewards.items()},
                'actions': {
                    k: [int(x) for x in v]
                    if hasattr(v, '__iter__') and not isinstance(v, (str, bytes))
                    else str(v)
                    for k, v in (actions or {}).items()
                },
                'metrics': {
                    'sla': float(sample.get('SLA_Uptime_Percentage', 1.0)),
                    'compromised': float(sample.get('compromised_hosts', 0.0)),
                    'false_positives': float(sample.get('false_positives_total', 0.0)),
                    'blue_reward': float(
                        sum(v for k, v in rewards.items() if 'blue' in k)
                    ),
                    'red_reward': float(
                        sum(v for k, v in rewards.items() if 'red' in k)
                    ),
                },
                'belief_n': int(
                    build_graph_observation(gs, 'blue_dmz')['node_mask'].sum()
                ),
                'oracle_n': int(build_oracle_graph(gs)['node_mask'].sum()),
            }
        )

    def to_dict(self) -> dict:
        return {
            'scenario': self.scenario,
            'seed': self.seed,
            'ticks': self.ticks,
        }

    def save_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict()))
        return path


def write_replay_html(payload: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(payload).replace('</', '<\\/')
    path.write_text(_HTML.replace('__PAYLOAD__', blob), encoding='utf-8')
    return path


_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>NetForge Replay</title>
<style>
:root { --bg:#07131a; --panel:#0d1f29; --line:#1c3b48; --teal:#2a9d8f; --red:#e76f51; --gold:#e9c46a; --muted:#8aa4b0; --text:#e7f1f4; }
* { box-sizing:border-box; }
body { margin:0; font-family: ui-sans-serif, system-ui, sans-serif; background:var(--bg); color:var(--text); }
header { padding:16px 24px; border-bottom:1px solid var(--line); display:flex; justify-content:space-between; align-items:center; }
h1 { font-size:18px; margin:0; letter-spacing:.04em; }
h1 span { color:var(--teal); }
.meta { color:var(--muted); font-size:13px; }
main { display:grid; grid-template-columns: 1.3fr 1fr; gap:12px; padding:12px 24px 24px; }
.card { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:12px; min-height:280px; }
svg { width:100%; height:360px; }
.node { stroke:#042029; stroke-width:1.5; }
.legend { display:flex; gap:12px; font-size:12px; color:var(--muted); margin-top:8px; }
.dot { width:8px; height:8px; border-radius:50%; display:inline-block; margin-right:4px; }
.siem { font-family: ui-monospace, monospace; font-size:11px; max-height:360px; overflow:auto; }
.siem div { padding:4px 0; border-bottom:1px solid var(--line); color:#b7d4dc; }
.controls { grid-column:1 / -1; display:flex; gap:12px; align-items:center; }
input[type=range] { flex:1; }
.kpis { display:flex; gap:16px; }
.kpi { font-size:12px; color:var(--muted); }
.kpi b { display:block; color:var(--text); font-size:18px; }
canvas { width:100%; height:90px; }
</style>
</head>
<body>
<header>
  <h1>NETFORGE <span>ARENA</span> · replay</h1>
  <div class="meta" id="meta"></div>
</header>
<main>
  <div class="card">
    <div class="meta">belief graph · hosts from SIEM + inventory (oracle is diagnostic-only)</div>
    <svg id="g" viewBox="0 0 640 360"></svg>
    <div class="legend">
      <span><i class="dot" style="background:#2a9d8f"></i>clean</span>
      <span><i class="dot" style="background:#e76f51"></i>compromised</span>
      <span><i class="dot" style="background:#6c757d"></i>isolated</span>
      <span><i class="dot" style="background:#e9c46a"></i>decoy</span>
    </div>
  </div>
  <div class="card">
    <div class="meta">SIEM feed</div>
    <div class="siem" id="siem"></div>
  </div>
  <div class="controls card">
    <button id="play">Play</button>
    <input id="scrub" type="range" min="0" max="0" value="0"/>
    <div class="kpis">
      <div class="kpi">tick <b id="t">0</b></div>
      <div class="kpi">SLA <b id="sla">1.00</b></div>
      <div class="kpi">owned <b id="own">0</b></div>
      <div class="kpi">FP <b id="fp">0</b></div>
    </div>
  </div>
  <div class="card" style="grid-column:1 / -1">
    <div class="meta">Blue reward timeline</div>
    <canvas id="chart" width="1200" height="90"></canvas>
  </div>
</main>
<script>
const DATA = __PAYLOAD__;
const svg = document.getElementById('g');
const siem = document.getElementById('siem');
const scrub = document.getElementById('scrub');
const meta = document.getElementById('meta');
meta.textContent = DATA.scenario + ' · seed ' + DATA.seed + ' · ' + DATA.ticks.length + ' ticks';
scrub.max = Math.max(0, DATA.ticks.length - 1);
let i = 0, playing = false;
function color(h){
  if (h.isolated) return '#6c757d';
  if (h.decoy) return '#e9c46a';
  if (h.compromised) return '#e76f51';
  return '#2a9d8f';
}
function layout(hosts){
  const groups = {};
  hosts.forEach(h => { (groups[h.subnet] = groups[h.subnet] || []).push(h); });
  const keys = Object.keys(groups);
  const pos = {};
  keys.forEach((k, gi) => {
    const cx = 90 + gi * 150, cy = 180;
    groups[k].forEach((h, j) => {
      const a = (j / Math.max(groups[k].length, 1)) * Math.PI * 2;
      pos[h.ip] = [cx + Math.cos(a)*55, cy + Math.sin(a)*70];
    });
  });
  return pos;
}
function render(idx){
  const tick = DATA.ticks[idx] || {hosts:[], siem:[], metrics:{}};
  const pos = layout(tick.hosts || []);
  let html = '';
  (tick.hosts||[]).forEach(h => {
    const p = pos[h.ip]; if (!p) return;
    html += `<circle class="node" cx="${p[0]}" cy="${p[1]}" r="9" fill="${color(h)}"><title>${h.hostname}</title></circle>`;
  });
  svg.innerHTML = html;
  siem.innerHTML = (tick.siem||[]).map(l => `<div>${l.replace(/[<>]/g,'')}</div>`).join('') || '<div>(no alerts)</div>';
  document.getElementById('t').textContent = tick.tick ?? idx;
  document.getElementById('sla').textContent = (tick.metrics?.sla ?? 1).toFixed(2);
  document.getElementById('own').textContent = tick.metrics?.compromised ?? 0;
  document.getElementById('fp').textContent = tick.metrics?.false_positives ?? 0;
  scrub.value = idx;
  drawChart(idx);
}
function drawChart(idx){
  const c = document.getElementById('chart');
  const ctx = c.getContext('2d');
  ctx.clearRect(0,0,c.width,c.height);
  const ys = DATA.ticks.map(t => t.metrics?.blue_reward || 0);
  let acc = 0; const cum = ys.map(y => (acc += y));
  if (!cum.length) return;
  const min = Math.min(...cum, 0), max = Math.max(...cum, 1);
  ctx.strokeStyle = '#2a9d8f'; ctx.lineWidth = 2; ctx.beginPath();
  cum.forEach((y,i) => {
    const x = i / Math.max(cum.length-1,1) * c.width;
    const yy = c.height - (y-min)/(max-min+1e-6) * (c.height-8) - 4;
    i ? ctx.lineTo(x,yy) : ctx.moveTo(x,yy);
  });
  ctx.stroke();
  ctx.fillStyle = '#e9c46a';
  const x = idx / Math.max(cum.length-1,1) * c.width;
  ctx.fillRect(x-1,0,2,c.height);
}
document.getElementById('play').onclick = () => {
  playing = !playing;
  document.getElementById('play').textContent = playing ? 'Pause' : 'Play';
};
scrub.oninput = e => { i = +e.target.value; render(i); };
setInterval(() => { if (!playing) return; i = (i+1) % DATA.ticks.length; render(i); }, 220);
render(0);
</script>
</body>
</html>
"""
