// gta6-trophy-rarity-forecast-01: a forecaster for every GTA VI trophy's earn rate
// on Exophase at a case's target, as five quantiles per trophy.
//
// The model, per trophy, from the case's own history only:
//
//   1. COHORT DELAY (the main model). Players arrive over time — the
//      tracked count N(t) only grows — and a player who bought the game at
//      time s earns a trophy by time t with probability R * (1 - exp(-(t-s)
//      / tau)): R is the share who ever earn it, tau the typical delay.
//      So the earners at t are the cohorts summed with that delay, and
//      the rate is earners over players tracked. R and tau are fitted per
//      trophy (tau by a grid search, R by least squares), which makes the
//      forecast SATURATE at R instead of running away: a rate cannot pass
//      the share of players who will ever earn the trophy. This is the
//      shape a log-time line misses — it keeps rising past the asymptote.
//
//   2. EARNER / DENOMINATOR SPLIT and 3. LOG-TIME RATE FITS are the
//      cross-checks: the rate is also modelled as earners over players
//      tracked, each count fitted in log time, and as a log-time line
//      (plain, and with a 1/(1+h) bend that lets the curve flatten).
//      They are NOT averaged into the median — for a saturating rate
//      they are all wrong in the same direction, rising past the
//      asymptote — but how far they sit from the cohort median measures
//      how contested the extrapolation is, and that widens the band.
//      The four models are compared on a LONG holdout (fitted on the
//      first half of the reads, scored on the second): a short holdout
//      cannot tell a still-rising line from a curve about to saturate;
//      a long one can.
//
//   4. STORY CHAIN. Chapter trophies form a chain: nobody holds Chapter 2
//      without Chapter 1, the final mission without Chapter 4, or the
//      platinum without the final mission. The chain is read from the
//      descriptions ("Complete Chapter N", "Complete the final mission",
//      the platinum tier) and the medians are forced non-increasing along
//      it (isotonic regression).
//
//   5. HONEST SPREAD. The quantiles come from the best model's holdout
//      error plus a term that grows with the horizon and a term for the
//      biggest refresh jump the history shows, with a floor so the 90%
//      band is never as narrow as a persistence band. A history that never
//      moved at all (a page Exophase has not refreshed yet) gets a wide
//      band and a median shifted down, because the next refresh dilutes
//      the rates.
//
// Everything is deterministic: no randomness, no clock, no network. The
// same case file always writes the same bytes.
//
// Node 22, standard library only. Reads every <id>.in in the input
// directory ($MAKEFUN_INPUT_DIR else /input) and writes <id>.out per case
// to the output directory ($MAKEFUN_OUTPUT_DIR else /output).

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** Release anchor for log time: the game's release date, midnight UTC. */
const RELEASE_MS = Date.parse('2026-11-19T00:00:00Z');
/** Delay grid (hours) for the cohort model. */
const TAU_GRID = [0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];

interface Trophy {
  id: number;
  name: string;
  description: string;
  tier: string;
  points: number;
  secret: boolean;
}
interface Read {
  slot: string;
  at: string;
  players_tracked: number;
  trophies_earned: number;
  rates: Array<number | null>;
}
interface Case {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  game: { name: string; platform: string; release_date: string; source: string };
  trophies: Trophy[];
  history: Read[];
  notes?: string;
}

/** A percent clamped to the range the scorer's logit is defined on. */
const clampPct = (x: number): number => Math.min(99.99, Math.max(0.01, x));
/** logit of a percent, as the scorer defines it. */
const logit = (pct: number): number => {
  const p = clampPct(pct) / 100;
  return Math.log(p / (1 - p));
};
/** Inverse logit, back to a percent. */
const logistic = (l: number): number => 100 / (1 + Math.exp(-l));
/** x rounded to 4 decimals, as the starter rounds. */
const round4 = (x: number): number => Math.round(x * 10000) / 10000;

/**
 * Weighted least squares for y = sum_k coef[k] * feat[k](row). Solves the
 * normal equations by Gaussian elimination (at most 3 features here, so
 * the solve is exact and fast). Returns null when the system is singular.
 */
function wls(
  feats: Array<Array<number>>,
  ys: number[],
  ws: number[],
): Array<number> | null {
  const m = feats[0].length;
  const a: Array<Array<number>> = [];
  for (let i = 0; i < m; i++) {
    a.push(new Array<number>(m + 1).fill(0));
    for (let j = 0; j < m; j++) {
      let s = 0;
      for (let r = 0; r < ys.length; r++) s += ws[r] * feats[r][i] * feats[r][j];
      a[i][j] = s;
    }
    let s = 0;
    for (let r = 0; r < ys.length; r++) s += ws[r] * feats[r][i] * ys[r];
    a[i][m] = s;
  }
  for (let col = 0; col < m; col++) {
    let piv = col;
    for (let r = col + 1; r < m; r++) if (Math.abs(a[r][col]) > Math.abs(a[piv][col])) piv = r;
    if (Math.abs(a[piv][col]) < 1e-12) return null;
    const tmp = a[col];
    a[col] = a[piv];
    a[piv] = tmp;
    for (let r = 0; r < m; r++) {
      if (r === col) continue;
      const f = a[r][col] / a[col][col];
      for (let k = col; k <= m; k++) a[r][k] -= f * a[col][k];
    }
  }
  const coef: Array<number> = [];
  for (let i = 0; i < m; i++) coef.push(a[i][m] / a[i][i]);
  return coef;
}

/** Predict y at a feature row from fitted coefficients. */
const predict = (coef: Array<number>, feat: Array<number>): number => {
  let s = 0;
  for (let k = 0; k < coef.length; k++) s += coef[k] * feat[k];
  return s;
};

/**
 * Isotonic regression, non-increasing, by the pool-adjacent-violators
 * algorithm. Returns the closest non-increasing sequence to `vals`.
 */
function isotonicDecreasing(vals: number[]): number[] {
  const out: Array<{ sum: number; n: number }> = [];
  for (const v of vals) {
    out.push({ sum: v, n: 1 });
    while (out.length >= 2) {
      const b = out[out.length - 1];
      const a = out[out.length - 2];
      if (a.sum / a.n >= b.sum / b.n) break;
      out.pop();
      out.pop();
      out.push({ sum: a.sum + b.sum, n: a.n + b.n });
    }
  }
  const res: number[] = [];
  for (const block of out) {
    for (let i = 0; i < block.n; i++) res.push(block.sum / block.n);
  }
  return res;
}

/** One read reduced to the numbers the fits need. */
interface Point {
  x: number; // ln(1 + hours since release)
  h: number; // hours since release
  t: number; // ms since the epoch, for recency weights
  n: number; // players tracked
  rate: number | null; // this trophy's rate, percent
}

/**
 * The cohort-delay model. Cohorts are the players who arrived between
 * consecutive reads (dN >= 0, arrival at the later read's time). For a
 * delay tau, the "matured mass" at read j is G_j = N_j - M_j, where M_j
 * is the not-yet-matured mass, updated by the recurrence
 * M_j = M_{j-1} * exp(-(t_j - t_{j-1}) / tau) + dN_j. Earners are
 * E_j = R * G_j, so R is fitted by one-parameter least squares.
 * Returns, for a tau, the fitted R and the earners it predicts at each
 * read, plus the pieces needed to forecast past the last read.
 */
function cohortFit(
  nSeries: Array<{ t: number; n: number }>,
  earners: Array<number | null>,
  tau: number,
  idx: Array<number>,
  weights: Array<number>,
): { r: number; g: Array<number> } | null {
  // Matured mass G at every read (computed over the full series, since a
  // cohort's maturation only depends on arrivals, not on the trophy).
  // The first read's players are the launch cohort, already playing when
  // the recording starts: they count as matured (M_0 = 0), so G_0 is the
  // whole tracked count and R is the rate the first read shows. Starting
  // them unmatured would force G_0 = 0 and make R meaningless on a short
  // history.
  const g: Array<number> = [];
  let m = 0;
  let prevN = 0;
  let prevT = nSeries[0].t;
  for (let j = 0; j < nSeries.length; j++) {
    const dN = Math.max(nSeries[j].n - prevN, 0);
    if (j === 0) m = 0;
    else m = m * Math.exp(-(nSeries[j].t - prevT) / (tau * 3600 * 1000)) + dN;
    g.push(nSeries[j].n - m);
    prevN = nSeries[j].n;
    prevT = nSeries[j].t;
  }
  let num = 0;
  let den = 0;
  for (let k = 0; k < idx.length; k++) {
    const j = idx[k];
    const e = earners[j];
    if (e === null || !Number.isFinite(e) || g[j] <= 0) continue;
    num += weights[k] * g[j] * e;
    den += weights[k] * g[j] * g[j];
  }
  if (den <= 0) return null;
  return { r: num / den, g };
}

/** Forecast one trophy: its median logit and its spread in logit units. */
function forecastTrophy(
  pts: Point[],
  nSeries: Array<{ t: number; n: number }>,
  xTarget: number,
  hTarget: number,
  tTarget: number,
  horizonDays: number,
  nFitAll: Array<number> | null,
  maxJumpLnN: number,
  flat: boolean,
  lastRate: number,
): { medianLogit: number; spread: number } {
  const usable = pts.filter((p) => p.rate !== null && p.n > 0);
  if (usable.length < 2) {
    const spread = 0.5 * Math.sqrt(horizonDays) + 0.35;
    return { medianLogit: logit(lastRate), spread: Math.min(2, Math.max(0.35, spread)) };
  }
  const tLast = usable[usable.length - 1].t;
  // Recency weights: 12 h half-life, so the current regime dominates.
  const w = (p: Point): number => Math.pow(0.5, (tLast - p.t) / (12 * 3600 * 1000));
  const ws = usable.map(w);

  // Long holdout: fit on the first half of the reads, score on the second.
  // A short holdout cannot tell a still-rising line from a curve about to
  // saturate; a long one can.
  const nHold = usable.length >= 4 ? Math.floor(usable.length / 2) : 0;
  const nFit = usable.length - nHold;
  const fitIdx = usable.slice(0, nFit);
  const holdIdx = usable.slice(nFit);

  const fitLnE = (rows: Point[]): Array<number> | null =>
    wls(
      rows.map((p) => [1, p.x]),
      rows.map((p) => Math.log(Math.max(((p.rate as number) / 100) * p.n, 0.5))),
      rows.map(w),
    );
  const fitLnN = (rows: Point[]): Array<number> | null =>
    wls(
      rows.map((p) => [1, p.x]),
      rows.map((p) => Math.log(Math.max(p.n, 1))),
      rows.map((p) => Math.pow(0.5, (tLast - p.t) / (24 * 3600 * 1000))),
    );
  const fitRate = (rows: Point[], bend: boolean): Array<number> | null =>
    wls(
      rows.map((p) => (bend ? [1, p.x, 1 / (1 + p.h)] : [1, p.x])),
      rows.map((p) => logit(p.rate as number)),
      rows.map(w),
    );

  // Earners per read, aligned with the full N series (null where missing).
  const earnersFull: Array<number | null> = nSeries.map((_, j) => {
    const p = pts[j];
    if (!p || p.rate === null || p.n <= 0) return null;
    return ((p.rate as number) / 100) * p.n;
  });

  // ---- Holdout errors (logit scale) for each model. ----
  let maeA = NaN;
  let maeB1 = NaN;
  let maeB2 = NaN;
  let maeC = NaN;
  // The cohort model's delay tau is picked on the FULL history, not on
  // the holdout: a slow delay with a high asymptote and a fast delay with
  // a low one both fit a short recent window, but only the right one fits
  // the early rise too, so the early reads are what separate them.
  let bestTau = NaN;
  {
    let best = { sse: Infinity, tau: NaN };
    for (const tau of TAU_GRID) {
      const fit = cohortFit(
        nSeries,
        earnersFull,
        tau,
        usable.map((p) => pts.indexOf(p)),
        ws,
      );
      if (!fit || fit.r <= 0) continue;
      let sse = 0;
      let wsum = 0;
      for (let k = 0; k < usable.length; k++) {
        const p = usable[k];
        const j = pts.indexOf(p);
        if (j < 0 || fit.g[j] <= 0) continue;
        const rPred = (fit.r * fit.g[j] / p.n) * 100;
        const d = logit(p.rate as number) - logit(rPred);
        sse += ws[k] * d * d;
        wsum += ws[k];
      }
      if (wsum > 0 && sse / wsum < best.sse) best = { sse: sse / wsum, tau };
    }
    bestTau = best.tau;
  }
  if (nHold > 0 && nFit >= 2) {
    const cE = fitLnE(fitIdx);
    const cN = fitLnN(fitIdx);
    const cB1 = fitRate(fitIdx, false);
    const cB2 = fitRate(fitIdx, true);
    // Cohort model holdout error, at the tau the full history picked.
    if (Number.isFinite(bestTau)) {
      const fit = cohortFit(
        nSeries,
        earnersFull,
        bestTau,
        fitIdx.map((p) => pts.indexOf(p)),
        fitIdx.map(w),
      );
      if (fit && fit.r > 0) {
        let s = 0;
        let cnt = 0;
        for (const p of holdIdx) {
          const j = pts.indexOf(p);
          if (j < 0 || fit.g[j] <= 0) continue;
          const rPred = (fit.r * fit.g[j] / p.n) * 100;
          s += Math.abs(logit(p.rate as number) - logit(rPred));
          cnt++;
        }
        if (cnt > 0) maeC = s / cnt;
      }
    }
    let sA = 0;
    let sB1 = 0;
    let sB2 = 0;
    let cntA = 0;
    let cntB = 0;
    for (const p of holdIdx) {
      const y = logit(p.rate as number);
      if (cE && cN) {
        const ePred = Math.exp(predict(cE, [1, p.x]));
        const nPred = Math.max(Math.exp(predict(cN, [1, p.x])), 1);
        sA += Math.abs(y - logit((ePred / nPred) * 100));
        cntA++;
      }
      if (cB1) {
        sB1 += Math.abs(y - predict(cB1, [1, p.x]));
        cntB++;
      }
      if (cB2) {
        sB2 += Math.abs(y - predict(cB2, [1, p.x, 1 / (1 + p.h)]));
      }
    }
    maeA = cntA > 0 ? sA / cntA : NaN;
    maeB1 = cntB > 0 ? sB1 / cntB : NaN;
    maeB2 = cntB > 0 ? sB2 / cntB : NaN;
  }

  // ---- Full fits on all usable reads. ----
  const cE = fitLnE(usable);
  const cB1 = fitRate(usable, false);
  const cB2 = fitRate(usable, true);
  // Cohort model on all reads, at the tau the full history picked.
  const tauC = bestTau;
  let logitC = NaN;
  if (Number.isFinite(tauC)) {
    const fit = cohortFit(
      nSeries,
      earnersFull,
      tauC,
      usable.map((p) => pts.indexOf(p)),
      ws,
    );
    if (fit && fit.r > 0 && nFitAll) {
      // Forecast earners at the target: old cohorts matured to T, plus
      // the players who arrive between the last read and T (uniform
      // arrival, so their mean matured share is 1 - tau/dT*(1-exp(-dT/tau))).
      const nT = Math.max(Math.exp(predict(nFitAll, [1, xTarget])), 1);
      const nLast = nSeries[nSeries.length - 1].n;
      const mLast = nLast - fit.g[nSeries.length - 1];
      const dT = Math.max((tTarget - tLast) / 3600000, 0);
      const oldPart = nLast - mLast * Math.exp(-dT / tauC);
      let newPart = 0;
      if (nT > nLast && dT > 0) {
        const fNew = 1 - (tauC / dT) * (1 - Math.exp(-dT / tauC));
        newPart = (nT - nLast) * Math.max(0, Math.min(1, fNew));
      }
      const eT = fit.r * (oldPart + newPart);
      logitC = logit((eT / nT) * 100);
    }
  }

  // Model A forecast: E(target) / N(target).
  let logitA = NaN;
  if (cE && nFitAll) {
    const eT = Math.exp(predict(cE, [1, xTarget]));
    const nT = Math.max(Math.exp(predict(nFitAll, [1, xTarget])), 1);
    logitA = logit((eT / nT) * 100);
  }
  const logitB1 = cB1 ? predict(cB1, [1, xTarget]) : NaN;
  const logitB2 = cB2 ? predict(cB2, [1, xTarget, 1 / (1 + hTarget)]) : NaN;

  // ---- The median: the cohort model is the structural one. ----
  // The log-time models are approximations that keep rising past the
  // asymptote a rate saturates at; they are not averaged into the median,
  // because they are all wrong in the same direction for a saturating
  // rate. They still matter: how far they sit from the cohort median
  // measures how uncertain the extrapolation is, and that widens the
  // band. If the cohort model fails, fall back to the log-time blend.
  const weight = (mae: number): number =>
    Number.isFinite(mae) ? 1 / (mae + 0.1) : 1 / 0.4;
  let medianLogit: number;
  let disagree = 0;
  if (Number.isFinite(logitC)) {
    medianLogit = logitC;
    const others = [logitA, logitB1, logitB2].filter((l) => Number.isFinite(l));
    if (others.length > 0) {
      let s = 0;
      for (const l of others) s += (l - logitC) * (l - logitC);
      disagree = Math.sqrt(s / others.length);
    }
  } else {
    const parts: Array<{ l: number; w: number }> = [];
    if (Number.isFinite(logitA)) parts.push({ l: logitA, w: weight(maeA) });
    if (Number.isFinite(logitB1)) parts.push({ l: logitB1, w: weight(maeB1) });
    if (Number.isFinite(logitB2)) parts.push({ l: logitB2, w: weight(maeB2) });
    if (parts.length === 0) {
      medianLogit = logit(lastRate);
    } else {
      let sw = 0;
      let sl = 0;
      for (const p of parts) {
        sw += p.w;
        sl += p.w * p.l;
      }
      medianLogit = sl / sw;
    }
  }
  if (!Number.isFinite(medianLogit)) medianLogit = logit(lastRate);
  medianLogit = Math.max(logit(0.01), Math.min(logit(99.99), medianLogit));

  // Spread: the cohort model's holdout error, plus a horizon term, a
  // refresh-jump term, and the models' disagreement about the median —
  // the band widens exactly where the extrapolation is contested.
  // Floored so it is never persistence-narrow.
  let sResid = NaN;
  const maes = [maeC, maeA, maeB1, maeB2].filter((m) => Number.isFinite(m));
  if (maes.length > 0) sResid = Math.min(...maes);
  if (!Number.isFinite(sResid)) {
    if (cB1) {
      let s = 0;
      for (const p of usable) {
        s += Math.abs(logit(p.rate as number) - predict(cB1, [1, p.x]));
      }
      sResid = s / usable.length;
    } else {
      sResid = 0.3;
    }
  }
  sResid = Math.max(0.15, sResid);
  const sHor = 0.35 * Math.sqrt(horizonDays);
  const sJump = Math.min(0.8, 0.5 * maxJumpLnN);
  const sDis = Math.min(1.2, 0.5 * disagree);
  let spread = Math.sqrt(sResid * sResid + sHor * sHor + sJump * sJump + sDis * sDis);
  spread = Math.min(2, Math.max(0.35, spread));

  if (flat) {
    // A page that never moved has not been refreshed yet: the next refresh
    // dilutes the rates, so widen the band and pull the median down.
    spread = Math.max(spread, 1.0);
    medianLogit -= 0.3;
    medianLogit = Math.max(logit(0.01), Math.min(logit(99.99), medianLogit));
  }
  return { medianLogit, spread };
}

/** The story chain, in order: Chapter 1..4, final mission, platinum. */
function storyChain(trophies: Trophy[]): Array<number> {
  const chain: Array<{ order: number; idx: number }> = [];
  trophies.forEach((t, i) => {
    const m = /chapter\s+(\d+)/i.exec(t.description);
    if (m) {
      chain.push({ order: parseInt(m[1], 10), idx: i });
    } else if (/final mission/i.test(t.description)) {
      chain.push({ order: 99, idx: i });
    } else if (t.tier === 'platinum') {
      chain.push({ order: 100, idx: i });
    }
  });
  chain.sort((a, b) => a.order - b.order);
  return chain.map((c) => c.idx);
}

/** Forecast one whole case. */
function forecastCase(c: Case): Array<{ id: number; q: number[] }> {
  const release = Date.parse(`${c.game.release_date}T00:00:00Z`);
  const anchor = Number.isFinite(release) ? release : RELEASE_MS;
  const horizonDays = c.horizon_minutes / 1440;
  const xTarget = Math.log(1 + (Date.parse(c.target) - anchor) / 3600000);
  const hTarget = Math.exp(xTarget) - 1;
  const tTarget = Date.parse(c.target);

  const reads = c.history;
  const tLast = reads.length > 0 ? Date.parse(reads[reads.length - 1].at) : 0;

  // The player count's own fit, on all reads (24 h recency half-life).
  const nSeries = reads
    .map((r) => ({ t: Date.parse(r.at), n: r.players_tracked }))
    .filter((p) => Number.isFinite(p.t) && p.n > 0);
  let nFitAll: Array<number> | null = null;
  if (nSeries.length >= 2) {
    nFitAll = wls(
      nSeries.map((p) => [1, Math.log(1 + (p.t - anchor) / 3600000)]),
      nSeries.map((p) => Math.log(p.n)),
      nSeries.map((p) => Math.pow(0.5, (tLast - p.t) / (24 * 3600 * 1000))),
    );
  }
  // Biggest single-read jump in the player count (refresh risk).
  let maxJumpLnN = 0;
  for (let j = 1; j < nSeries.length; j++) {
    const d = Math.abs(Math.log(nSeries[j].n) - Math.log(nSeries[j - 1].n));
    if (Number.isFinite(d) && d > maxJumpLnN) maxJumpLnN = d;
  }
  // Flat history: the player count never moved and no rate ever moved.
  let flat = nSeries.length > 0 && nSeries.every((p) => p.n === nSeries[0].n);
  if (flat) {
    outer: for (let i = 0; i < c.trophies.length; i++) {
      let first: number | null = null;
      for (const r of reads) {
        const v = r.rates[i];
        if (v === null || v === undefined) continue;
        if (first === null) first = v;
        else if (v !== first) {
          flat = false;
          break outer;
        }
      }
    }
  }

  // Per-trophy forecasts.
  const medians: Array<number> = [];
  const spreads: Array<number> = [];
  c.trophies.forEach((t, i) => {
    const pts: Point[] = reads.map((r) => ({
      x: Math.log(1 + (Date.parse(r.at) - anchor) / 3600000),
      h: (Date.parse(r.at) - anchor) / 3600000,
      t: Date.parse(r.at),
      n: r.players_tracked,
      rate: r.rates[i] === null || r.rates[i] === undefined ? null : r.rates[i],
    }));
    const last = [...reads].reverse().find((r) => r.rates[i] !== null && r.rates[i] !== undefined);
    const lastRate = last ? (last.rates[i] as number) : 0.5;
    const f = forecastTrophy(pts, nSeries, xTarget, hTarget, tTarget, horizonDays, nFitAll, maxJumpLnN, flat, lastRate);
    medians.push(f.medianLogit);
    spreads.push(f.spread);
  });

  // Enforce the story chain: medians non-increasing along it.
  const chain = storyChain(c.trophies);
  if (chain.length >= 2) {
    const adj = isotonicDecreasing(chain.map((i) => medians[i]));
    chain.forEach((trophyIdx, k) => {
      medians[trophyIdx] = adj[k];
    });
  }

  return c.trophies.map((t, i) => {
    const q = Z.map((z) => round4(clampPct(logistic(medians[i] + z * spreads[i]))));
    // Rounding can only tie, never invert, but enforce it outright.
    for (let k = 1; k < q.length; k++) q[k] = Math.max(q[k], q[k - 1]);
    return { id: t.id, q };
  });
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  if (!c.history || c.history.length === 0) throw new Error(`${file} has no history`);
  const trophies = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
