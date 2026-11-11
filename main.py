"""Forecast Roblox's Up-and-Coming top 10 over a 72-hour window (make.fun roblox-up-and-coming-forecast-01).

Reads history.ndjson and task.json from $MAKEFUN_INPUT_DIR (else /input) and writes forecast.json to
$MAKEFUN_OUTPUT_DIR (else /output). Python 3.13, standard library only, deterministic.

Model: each game on the last chart has a future log-rank drawn from a Laplace distribution centred on its current
log-rank (shifted by its recent log-rank trend) whose width grows with the horizon and the game's volatility. A
place's distribution at one horizon is the probability mass each game puts on that place, normalised over games,
after reserving a share for games not on the last chart (newcomers), which grows with the horizon. The forecast
is the average of those distributions over the 72 hourly slots of the window.
"""
import datetime as dt
import json
import math
import os

PARAMS = {
    's0': 0.25,       # Laplace scale of log-rank at horizon 0
    's1': 0.12,       # scale growth per sqrt(hour)
    'mom': 0.5,       # weight on recent log-rank slope (per hour) as drift
    'mom_cap': 24.0,  # hours of drift at most
    'eps': 0.03,      # broad component: share spread evenly over places
    'nu_max': 0.6,    # newcomer share ceiling
    'nu_tau': 90.0,   # newcomer share time scale (hours)
    'nu_pow': 2.0,    # newcomer share shape
    'nu_place': 0.08,  # extra newcomer share per place below 1 (relative)
}
if os.environ.get('UAC_PARAMS'):
    PARAMS.update(json.loads(os.environ['UAC_PARAMS']))

IN_DIR = os.environ.get('MAKEFUN_INPUT_DIR') or '/input'
OUT_DIR = os.environ.get('MAKEFUN_OUTPUT_DIR') or '/output'


def ts(s):
    return dt.datetime.fromisoformat(s.replace('Z', '+00:00')).timestamp() / 3600.0


def lap_cdf(x, mu, b):
    z = (x - mu) / b
    if z < 0:
        return 0.5 * math.exp(z)
    return 1.0 - 0.5 * math.exp(-z)


def load():
    with open(os.path.join(IN_DIR, 'task.json')) as f:
        task = json.load(f)
    reads = []
    with open(os.path.join(IN_DIR, 'history.ndjson')) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get('ok') is False or not r.get('up_and_coming'):
                continue
            reads.append(r)
    reads.sort(key=lambda r: r['at'])
    return task, reads


def chart(read):
    return [r['universe_id'] for r in sorted(read['up_and_coming'], key=lambda r: r['rank'])]


def slopes(reads, t_last, span=24.0):
    """Least-squares slope of log-rank per hour over the reads in the last `span` hours, per game."""
    pts = {}
    for r in reads:
        t = ts(r['at'])
        if t < t_last - span:
            continue
        for i, g in enumerate(chart(r)):
            pts.setdefault(g, []).append((t - t_last, math.log(i + 1)))
    out = {}
    for g in sorted(pts):
        p = pts[g]
        if len(p) < 2:
            continue
        mt = sum(x for x, _ in p) / len(p)
        my = sum(y for _, y in p) / len(p)
        sxx = sum((x - mt) ** 2 for x, _ in p)
        if sxx <= 1e-9:
            continue
        out[g] = sum((x - mt) * (y - my) for x, y in p) / sxx
    return out


def forecast(task, reads):
    P = PARAMS
    places = int(task.get('places', 10))
    w = task['window']
    t0 = ts(w['from'])
    nslots = int(w.get('slots', 72))
    slot_h = float(w.get('slot_minutes', 60)) / 60.0
    last = reads[-1]
    t_last = ts(last['at'])
    games = chart(last)
    n = len(games)
    slope = slopes(reads, t_last)
    edges = [math.log(k + 0.5) for k in range(1, places + 1)]
    acc = [[0.0] * n for _ in range(places)]
    for s in range(nslots):
        d = max(0.0, t0 + (s + 0.5) * slot_h - t_last)
        scale = P['s0'] + P['s1'] * math.sqrt(d)
        nu_base = P['nu_max'] * (1.0 - math.exp(-((d / P['nu_tau']) ** P['nu_pow'])))
        mass = [[0.0] * n for _ in range(places)]
        for i, g in enumerate(games):
            mu = math.log(i + 1) + P['mom'] * slope.get(g, 0.0) * min(d, P['mom_cap'])
            prev = 0.0
            for k in range(places):
                c = lap_cdf(edges[k], mu, scale)
                mass[k][i] = (1 - P['eps']) * (c - prev) + P['eps'] / n
                prev = c
        for k in range(places):
            tot = sum(mass[k])
            nu = min(0.95, nu_base * (1.0 + P['nu_place'] * k))
            for i in range(n):
                acc[k][i] += (1.0 - nu) * mass[k][i] / tot / nslots
    return games, acc


def write(task, games, acc):
    places = len(acc)
    rows = []
    for k in range(places):
        order = sorted(range(len(games)), key=lambda i: (-acc[k][i], games[i]))[:100]
        total = 0
        for i in order:
            q = int(acc[k][i] * 1e6)   # floor to 6 decimals
            if q < 1 or total + q > 999990:
                continue
            total += q
            rows.append({'universe_id': games[i], 'place': k + 1, 'p': q / 1e6})
    # top 10: assign places greedily by probability, best first
    top, used = [], set()
    for k in range(places):
        best = max((i for i in range(len(games)) if games[i] not in used), key=lambda i: (acc[k][i], -i))
        top.append(games[best])
        used.add(games[best])
    out = {'probabilities': rows, 'top10': top}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, task.get('output', 'forecast.json')), 'w') as f:
        f.write(json.dumps(out, separators=(',', ':')) + '\n')


def main():
    task, reads = load()
    games, acc = forecast(task, reads)
    write(task, games, acc)


if __name__ == '__main__':
    main()
