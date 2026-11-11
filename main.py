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
BETA = [-0.0948, -0.8319, 8.0059, -0.4529, -0.2704, 0.1477, 0.0988, -0.1008, -0.213, 0.3214, 0.0742,
        -10.5314, 2.2247, 1.2403]
PARAMS = {
    'gain_lag': 12.0,    # hours back for the rank-gain feature
    'age_cap': 7.0,      # days; age since first seen is capped here (and set to it for games in the first read)
    'shrink': 0.0,       # mix with uniform-over-chart, per place
    'h_scale': 72.0,     # hours that make h = 1
    'refit_lambda': 100.0,  # ridge strength toward BETA for the run-time refit (0 = no refit)
    'refit_pairs': 300,   # read pairs used by the refit at most (evenly thinned)
    'refit_iters': 2,     # Newton steps
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


def solve(A, b):
    """Gaussian elimination with partial pivoting: x with A x = b."""
    n = len(b)
    M = [A[r][:] + [b[r]] for r in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        if M[c][c] == 0.0:
            continue
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                if f != 0.0:
                    for cc in range(c, n + 1):
                        M[r][cc] -= f * M[c][cc]
    return [M[r][n] / M[r][r] if M[r][r] != 0.0 else 0.0 for r in range(n)]


def refit(hist, prior):
    """Newton steps on the history's own read pairs (1 to 80 h apart), with a ridge centred on the prior."""
    lam = PARAMS['refit_lambda']
    pairs = []
    for i in range(len(hist.t)):
        for j in range(i + 1, len(hist.t)):
            d = hist.t[j] - hist.t[i]
            if 1.0 <= d <= 80.0:
                pairs.append((i, j, d))
    # thin evenly within 8-hour lag bins, so hourly reads' many short-lag pairs don't swamp the long lags
    cap = int(PARAMS['refit_pairs']) // 10
    bins = {}
    for p in pairs:
        bins.setdefault(min(9, int(p[2] // 8)), []).append(p)
    pairs = []
    for b in sorted(bins):
        ps = bins[b]
        if len(ps) > cap:
            step = len(ps) / cap
            ps = [ps[int(a * step)] for a in range(cap)]
        pairs.extend(ps)
    if not pairs:
        return prior
    origins = {}
    samples = []
    for i, j, d in pairs:
        if i not in origins:
            orig = hist.origin(i)
            origins[i] = (orig, {o[0]: a for a, o in enumerate(orig)})
        orig, idx = origins[i]
        for k in range(1, 11):
            X = [feature_vector(o, k, d) for o in orig] + [new_vector(k, d)]
            samples.append((X, idx.get(hist.charts[j][k - 1], len(orig))))
    beta = list(prior)
    F = NFEAT
    for _ in range(int(PARAMS['refit_iters'])):
        g = [-lam * (b - p) for b, p in zip(beta, prior)]
        H = [[lam if a == c else 0.0 for c in range(F)] for a in range(F)]
        for X, ch in samples:
            us = [sum(b * x for b, x in zip(beta, xv)) for xv in X]
            mx = max(us)
            es = [math.exp(u - mx) for u in us]
            z = sum(es)
            ex = [0.0] * F
            exx = [[0.0] * F for _ in range(F)]
            for e, xv in zip(es, X):
                p = e / z
                if p < 1e-9:
                    continue
                for a in range(F):
                    xa = xv[a] * p
                    if xa == 0.0:
                        continue
                    ex[a] += xa
                    row = exx[a]
                    for c in range(a, F):
                        row[c] += xa * xv[c]
            xc = X[ch]
            for a in range(F):
                g[a] += xc[a] - ex[a]
                for c in range(a, F):
                    v = exx[a][c] - ex[a] * ex[c]
                    H[a][c] += v
                    if c != a:
                        H[c][a] += v
        step = solve(H, g)
        big = max(abs(s) for s in step)
        if not math.isfinite(big):
            return list(prior)
        if big > 1.0:   # damp: the refit should stay a correction to the prior
            step = [s / big for s in step]
        beta = [b + s for b, s in zip(beta, step)]
    if not all(math.isfinite(b) for b in beta):
        return list(prior)
    return beta


def forecast(task, reads):
    places = int(task.get('places', 10))
    w = task['window']
    t0 = ts(w['from'])
    nslots = int(w.get('slots', 72))
    slot_h = float(w.get('slot_minutes', 60)) / 60.0
    hist = History(reads)
    beta = refit(hist, BETA) if PARAMS['refit_lambda'] > 0 else BETA
    i = len(reads) - 1
    orig = hist.origin(i)
    games = [o[0] for o in orig]
    n = len(games)
    acc = [[0.0] * n for _ in range(places)]
    sh = PARAMS['shrink']
    for s in range(nslots):
        d = max(0.5, t0 + (s + 0.5) * slot_h - hist.t[i])
        for k in range(places):
            p, pn = probs(orig, k + 1, d, beta)
            for j in range(n):
                acc[k][j] += ((1 - sh) * p[j] + sh * (1 - pn) / n) / nslots
    return games, acc


def write(task, games, acc, reads):
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
        cand = [i for i in range(len(games)) if games[i] not in used]
        if not cand:
            break
        best = max(cand, key=lambda i: (acc[k][i], -i))
        top.append(games[best])
        used.add(games[best])
    for r in reversed(reads):   # a chart shorter than 10: fill from earlier charts
        for g in chart(r):
            if len(top) < 10 and g not in used:
                top.append(g)
                used.add(g)
    out ={'probabilities': rows, 'top10': top}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, task.get('output', 'forecast.json')), 'w') as f:
        f.write(json.dumps(out, separators=(',', ':')) + '\n')


def main():
    task, reads = load()
    games, acc = forecast(task, reads)
    write(task, games, acc, reads)


if __name__ == '__main__':
    main()
