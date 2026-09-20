# Self-play / Elo

One ladder: Blue score = mean SLA, Red = `1 − SLA`. `benchmarks/self_play.py`.

```bash
python -m benchmarks.self_play --seeds 5 --max-ticks 150
```

```python
from benchmarks.self_play import population_tournament

result = population_tournament(
    scenarios=['ransomware', 'apt_espionage', 'cloud_hybrid'],
    seeds=list(range(5)),
    max_ticks=150,
)
```

Default pools: `random`, `heuristic`, `killchain-red`. Pass `red_pool` / `blue_pool`
(`name → factory`) to add trained agents.

```
 1. heuristic-blue    1041.7
 2. random-blue       1022.1
 3. killchain-red      980.4
```
