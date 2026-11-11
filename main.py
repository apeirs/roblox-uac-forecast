"""Forecast Roblox's Up-and-Coming top 10 over a 72-hour window (make.fun roblox-up-and-coming-forecast-01).

Reads history.ndjson and task.json from $MAKEFUN_INPUT_DIR (else /input) and writes forecast.json to
$MAKEFUN_OUTPUT_DIR (else /output). Python 3.13, standard library only, deterministic.

Model: a conditional logit. For a place k at a horizon of d hours after the last read, the game there is one of the
games on the last chart, or a game not on it ("newcomer"). Each game's utility combines its log-rank distance to
k (a kernel that widens with d), its recent rank gain, how long it has been on the chart, and its player count
against the chart's median, each with a horizon interaction; the newcomer option has a utility that grows with d
and with k. The coefficients were fitted by maximum likelihood on pairs of archive reads 1 to 80 hours apart.
A place's forecast is the average of the softmax over the window's hourly slots; the newcomer share is left
unnamed, so the scorer spreads it over games that join the chart.
"""
import datetime as dt
import json
import math
import os

# Coefficients of the utility (see feature_vector); fitted offline.
BETA = [0.6555, -1.3452, 6.1498, -0.397, -0.4099, 0.0863, 0.7286, 0.0532, -0.406, 0.297, 0.0211,
        -6.4977, 0.6002, 1.6573]
PARAMS = {
    'gain_lag': 12.0,    # hours back for the rank-gain feature
    'age_cap': 7.0,      # days; age since first seen is capped here (and set to it for games in the first read)
    'shrink': 0.0,       # mix with uniform-over-chart, per place
    'h_scale': 72.0,     # hours that make h = 1
}
if os.environ.get('UAC_PARAMS'):
    _p = json.loads(os.environ['UAC_PARAMS'])
    BETA = _p.pop('beta', BETA)
    PARAMS.update(_p)

IN_DIR = os.environ.get('MAKEFUN_INPUT_DIR') or '/input'
OUT_DIR = os.environ.get('MAKEFUN_OUTPUT_DIR') or '/output'
NFEAT = 14


def ts(s):
    return dt.datetime.fromisoformat(s.replace('Z', '+00:00')).timestamp() / 3600.0


def load(in_dir=None):
    in_dir = in_dir or IN_DIR
    with open(os.path.join(in_dir, 'task.json')) as f:
        task = json.load(f)
    reads = []
    with open(os.path.join(in_dir, 'history.ndjson')) as f:
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


class History:
    """Charts, times and per-game first sighting, with origin features for any read."""

    def __init__(self, reads):
        self.t = [ts(r['at']) for r in reads]
        self.charts = [chart(r) for r in reads]
        self.playing = [{x['universe_id']: (x.get('playing') or 0) for x in r['up_and_coming']} for r in reads]
        self.first = {}
        for i, c in enumerate(self.charts):
            for g in c:
                self.first.setdefault(g, self.t[i])

    def origin(self, i):
        """Per-game raw features at read i: list of (game, logrank, gain, age_days, plmed)."""
        t_i = self.t[i]
        c = self.charts[i]
        lag = PARAMS['gain_lag']
        prev = None
        best = None
        for j in range(i - 1, -1, -1):
            dtj = t_i - self.t[j]
            if dtj < lag * 0.5:
                continue
            if dtj > lag * 2.5:
                break
            if best is None or abs(dtj - lag) < best:
                best, prev = abs(dtj - lag), j
        prank = {}
        if prev is not None:
            prank = {g: k + 1 for k, g in enumerate(self.charts[prev])}
            absent = len(self.charts[prev]) + 10
        pl = sorted(self.playing[i].values())
        med = pl[len(pl) // 2] if pl else 0
        cap = PARAMS['age_cap']
        out = []
        for k, g in enumerate(c):
            lr = math.log(k + 1)
            gain = (math.log(prank.get(g, absent)) - lr) if prev is not None else 0.0
            if self.first[g] <= self.t[0]:
                age = cap
            else:
                age = min(cap, (t_i - self.first[g]) / 24.0)
            plm = max(-3.0, min(3.0, math.log((1 + self.playing[i].get(g, 0)) / (1 + med))))
            out.append((g, lr, gain, age, plm))
        return out


def feature_vector(o, k, d):
    """Features of origin game o = (g, logrank, gain, age, plmed) for place k at horizon d hours."""
    h = d / PARAMS['h_scale']
    lk = math.log(k)
    dist = o[1] - lk
    ad = abs(dist)
    return (-ad, -ad * h, -ad / (1.0 + d) ** 0.5, dist, dist * h, o[2], o[2] * h, o[3], o[3] * h, o[4], o[4] * h,
            0.0, 0.0, 0.0)


def new_vector(k, d):
    return (0.0,) * 11 + (1.0, math.log(1.0 + d), math.log(k))


def probs(orig, k, d, beta):
    """Softmax over the origin games and the newcomer option: (list of p per game, p_new)."""
    us = [sum(b * x for b, x in zip(beta, feature_vector(o, k, d))) for o in orig]
    un = sum(b * x for b, x in zip(beta, new_vector(k, d)))
    m = max(max(us), un)
    es = [math.exp(u - m) for u in us]
    en = math.exp(un - m)
    z = sum(es) + en
    return [e / z for e in es], en / z


def forecast(task, reads):
    places = int(task.get('places', 10))
    w = task['window']
    t0 = ts(w['from'])
    nslots = int(w.get('slots', 72))
    slot_h = float(w.get('slot_minutes', 60)) / 60.0
    hist = History(reads)
    i = len(reads) - 1
    orig = hist.origin(i)
    games = [o[0] for o in orig]
    n = len(games)
    acc = [[0.0] * n for _ in range(places)]
    sh = PARAMS['shrink']
    for s in range(nslots):
        d = max(0.5, t0 + (s + 0.5) * slot_h - hist.t[i])
        for k in range(places):
            p, pn = probs(orig, k + 1, d, BETA)
            for j in range(n):
                acc[k][j] += ((1 - sh) * p[j] + sh * (1 - pn) / n) / nslots
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
