// roblox-up-and-coming-forecast-01 entry: an empirical forecaster for Roblox's Up-and-Coming chart.
//
// Method (fit entirely from the history the program is given at run time — nothing about a
// specific week is baked in):
//   The scored window is the 72 hours after the history ends, read hourly, so the quantity to
//   estimate, for each place 1..10, is the distribution of that place's occupant over horizons
//   of roughly 0 to 72 hours past the last read. The history itself measures exactly that, at
//   every pair of reads: how often the occupant of a place is still there after a given gap, and
//   when it changed, where the new occupant came from (a neighbouring place, the rest of the top
//   10, ranks 11-20, ranks 21-40, deep in the chart, or off the chart entirely).
//
//   Version 2 refines the estimator after backtesting on truncated histories (official scoring):
//   - Instead of four wide gap bins, persistence and the origin mix are estimated on a fine
//     horizon grid (every 3 h from 3 to 69) with a triangular kernel over pair gaps
//     (bandwidth 6 h), then averaged with equal weight over the grid — each grid point stands
//     for 3 of the window's 72 hours. This tracks the steep decay of persistence with horizon
//     (place 1 holds ~0.66 at 12-18 h but ~0.05 by 60 h) that wide bins blur.
//   - Per-place estimates are shrunk toward the pooled all-places estimate at the same horizon
//     (prior strength 30 kernel-weighted pairs), which stabilises the sparse long-horizon and
//     low-place cells without washing out the strong per-place differences.
//   - The incoming share for a place is spread over the matching games of the last read with
//     weights playing^0.25 — nearly flat. Backtests showed the earlier playing^1 weighting
//     concentrated far too much on the current high-player games: who takes a given place is
//     much closer to uniform over the origin pool than player counts suggest.
//   - The share measured to arrive from deep in the chart or off it is left unnamed, which the
//     scoring spreads over the chart, as a hedge no named game can give. A small extra hedge
//     scales all named mass down at long horizons (a smooth ramp from 1.00 up to 18 h to 0.88
//     from 54 h on), where surprise occupants are most common.
//   Deterministic: no clock (the window comes from task.json), no randomness.

import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';

export interface ChartGame {
  universe_id: number;
  rank: number;
  playing: number | null;
}

export interface Read {
  t: number; // epoch ms
  chart: ChartGame[]; // in chart order
  rankOf: Map<number, number>;
}

interface Task {
  history_until: string;
  window: { from: string; until: string };
  places: number;
}

/** Horizon grid (hours past the earlier read); each point covers 3 of the 72 window hours. */
const GRID: number[] = Array.from({ length: 23 }, (_, i) => 3 + 3 * i);
/** Triangular kernel bandwidth over pair gaps, in hours. */
const BW = 6;
/** Shrinkage prior strength, in kernel-weighted pairs, toward the pooled estimate. */
const PRIOR = 30;
/** Exponent on player counts when spreading a place's incoming share over its origin pool. */
const ALPHA = 0.25;
/** Cap on a place's total named probability; the rest stays unnamed as the hedge. */
const MAX_NAMED = 0.995;

/** Extra unnamed hedge at horizon h: 1 up to 18 h, easing to 0.88 from 54 h on. */
function hedge(h: number): number {
  const x = Math.min(1, Math.max(0, (h - 18) / 36));
  return 1 - 0.12 * (x * x * (3 - 2 * x));
}

type Origin = 'top10' | 'r11_20' | 'r21_40' | 'deep' | 'new';
const ORIGINS: Origin[] = ['top10', 'r11_20', 'r21_40', 'deep', 'new'];

function originOf(rankInEarlier: number | undefined): Origin {
  if (rankInEarlier === undefined) return 'new';
  if (rankInEarlier <= 10) return 'top10';
  if (rankInEarlier <= 20) return 'r11_20';
  if (rankInEarlier <= 40) return 'r21_40';
  return 'deep';
}

export function parseHistory(text: string): Read[] {
  const reads: Read[] = [];
  for (const line of text.split('\n')) {
    const s = line.trim();
    if (!s) continue;
    let d: any;
    try {
      d = JSON.parse(s);
    } catch {
      continue;
    }
    if (!d || !Array.isArray(d.up_and_coming)) continue;
    const chart: ChartGame[] = d.up_and_coming
      .filter((g: any) => typeof g?.universe_id === 'number')
      .map((g: any, i: number) => ({
        universe_id: g.universe_id as number,
        rank: typeof g.rank === 'number' ? g.rank : i + 1,
        playing: typeof g.playing === 'number' ? g.playing : null,
      }))
      .sort((a: ChartGame, b: ChartGame) => a.rank - b.rank);
    reads.push({ t: Date.parse(d.at), chart, rankOf: new Map(chart.map((g) => [g.universe_id, g.rank] as const)) });
  }
  reads.sort((a, b) => a.t - b.t);
  return reads;
}

interface Pair {
  a: Read;
  b: Read;
  dh: number; // gap in hours
}

export interface Forecast {
  probabilities: { universe_id: number; place: number; p: number }[];
  top10: number[];
}

export function buildForecast(reads: Read[], places: number): Forecast {
  const last = reads[reads.length - 1]!;
  const pairs: Pair[] = [];
  for (let i = 0; i < reads.length; i++) {
    for (let j = i + 1; j < reads.length; j++) {
      pairs.push({ a: reads[i]!, b: reads[j]!, dh: (reads[j]!.t - reads[i]!.t) / 3_600_000 });
    }
  }
  const weightOf = (g: ChartGame): number => Math.pow(Math.max(1, g.playing ?? 1), ALPHA);
  const pools = new Map<Origin, ChartGame[]>();
  for (const o of ORIGINS) pools.set(o, last.chart.filter((g) => originOf(g.rank) === o));
  // dist[place] : universe_id -> probability
  const dist = new Map<number, Map<number, number>>();
  const add = (place: number, id: number, p: number) => {
    if (p <= 0) return;
    const m = dist.get(place) ?? new Map<number, number>();
    m.set(id, (m.get(id) ?? 0) + p);
    dist.set(place, m);
  };
  const wH = 1 / GRID.length;
  for (const h of GRID) {
    // Pooled (all places) kernel-weighted counts at this horizon: the shrinkage prior.
    let poolSame = 0;
    let poolCh = 0;
    const poolMix: Record<Origin, number> = { top10: 0, r11_20: 0, r21_40: 0, deep: 0, new: 0 };
    // Per-place kernel-weighted counts at this horizon.
    const same = new Array<number>(places).fill(0);
    const ch = new Array<number>(places).fill(0);
    const mix: Record<Origin, number>[] = Array.from({ length: places }, () => ({
      top10: 0, r11_20: 0, r21_40: 0, deep: 0, new: 0,
    }));
    for (const { a, b, dh } of pairs) {
      const kw = Math.max(0, 1 - Math.abs(dh - h) / BW);
      if (kw === 0) continue;
      const lim = Math.min(places, a.chart.length, b.chart.length);
      for (let k = 0; k < lim; k++) {
        const occA = a.chart[k]!.universe_id;
        const occB = b.chart[k]!.universe_id;
        if (occA === occB) {
          same[k]! += kw;
          poolSame += kw;
        } else {
          ch[k]! += kw;
          poolCh += kw;
          const o = originOf(a.rankOf.get(occB));
          mix[k]![o] += kw;
          poolMix[o] += kw;
        }
      }
    }
    const poolTot = poolSame + poolCh;
    const pooledPersist = poolTot > 0 ? poolSame / poolTot : 0.25;
    const poolMixTot = ORIGINS.reduce((s, o) => s + poolMix[o], 0);
    const pooledMix: Record<Origin, number> = { top10: 0.2, r11_20: 0.2, r21_40: 0.2, deep: 0.2, new: 0.2 };
    if (poolMixTot > 0) for (const o of ORIGINS) pooledMix[o] = poolMix[o] / poolMixTot;
    const sH = hedge(h);
    for (let k = 1; k <= places; k++) {
      const occupant = last.chart[k - 1];
      if (!occupant) continue;
      const tot = same[k - 1]! + ch[k - 1]!;
      const persist = (same[k - 1]! + PRIOR * pooledPersist) / (tot + PRIOR);
      const mTot = ORIGINS.reduce((s, o) => s + mix[k - 1]![o], 0);
      add(k, occupant.universe_id, wH * persist * sH);
      const changeShare = wH * (1 - persist) * sH;
      for (const o of ORIGINS) {
        if (o === 'deep' || o === 'new') continue; // left unnamed on purpose
        const mixP = (mix[k - 1]![o] + PRIOR * pooledMix[o]) / (mTot + PRIOR);
        const share = changeShare * mixP;
        if (share <= 0) continue;
        const pool = pools.get(o)!.filter((g) => g.universe_id !== occupant.universe_id);
        const totW = pool.reduce((s, g) => s + weightOf(g), 0);
        if (totW <= 0) continue;
        for (const g of pool) add(k, g.universe_id, (share * weightOf(g)) / totW);
      }
    }
  }
  const probabilities: Forecast['probabilities'] = [];
  const presence = new Map<number, number>();
  const expPlace = new Map<number, number>();
  for (let k = 1; k <= places; k++) {
    const m = dist.get(k);
    if (!m) continue;
    const rows = [...m.entries()].sort((x, y) => y[1] - x[1] || x[0] - y[0]).slice(0, 100);
    const sum = rows.reduce((s, [, p]) => s + p, 0);
    const scale = sum > MAX_NAMED ? MAX_NAMED / sum : 1;
    for (const [id, p] of rows) {
      const rp = Math.round(p * scale * 1e6) / 1e6;
      if (rp <= 0) continue;
      probabilities.push({ universe_id: id, place: k, p: rp });
      presence.set(id, (presence.get(id) ?? 0) + rp);
      expPlace.set(id, (expPlace.get(id) ?? 0) + rp * k);
    }
  }
  const ranked = [...presence.entries()]
    .map(([id, pr]) => ({ id, pr, ep: (expPlace.get(id) ?? 0) / pr }))
    .sort((a, b) => b.pr - a.pr || a.ep - b.ep)
    .slice(0, 10)
    .sort((a, b) => a.ep - b.ep || b.pr - a.pr);
  const top10 = ranked.map((r) => r.id);
  // Canonical row order (place, then probability desc, then id): the forecast bytes are then a
  // pure function of the distributions, stable across runtimes and easy to diff between runs.
  probabilities.sort((x, y) => x.place - y.place || y.p - x.p || x.universe_id - y.universe_id);
  for (const g of last.chart) {
    if (top10.length >= 10) break;
    if (!top10.includes(g.universe_id)) top10.push(g.universe_id);
  }
  return { probabilities, top10 };
}

export function run(inputDir: string, outputDir: string): void {
  const task = JSON.parse(readFileSync(join(inputDir, 'task.json'), 'utf8')) as Task;
  const places = Math.max(1, Math.min(10, task.places ?? 10));
  const reads = parseHistory(readFileSync(join(inputDir, 'history.ndjson'), 'utf8'));
  if (reads.length === 0) throw new Error('no reads in history');
  const forecast = buildForecast(reads, places);
  mkdirSync(outputDir, { recursive: true });
  writeFileSync(join(outputDir, 'forecast.json'), JSON.stringify(forecast), 'utf8');
}

const isMain = process.argv[1] !== undefined && import.meta.url === new URL(`file://${process.argv[1]}`).href;
if (isMain) {
  run(process.env.MAKEFUN_INPUT_DIR ?? '/input', process.env.MAKEFUN_OUTPUT_DIR ?? '/output');
}
