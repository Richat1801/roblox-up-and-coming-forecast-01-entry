// roblox-up-and-coming-forecast-01 entry: an empirical forecaster for Roblox's Up-and-Coming chart.
//
// Method (fit entirely from the history the program is given at run time — nothing about a
// specific week is baked in):
//   The scored window is the 72 hours after the history ends, read hourly, so the quantity to
//   estimate, for each place 1..10, is the distribution of that place's occupant over horizons
//   of roughly 0 to 72 hours past the last read. The history itself measures exactly that, at
//   every pair of reads: how often the occupant of a place is still there after a given gap, and
//   when it changed, where the new occupant came from (a neighbouring place, the rest of the top
//   10, ranks 11-20, ranks 21-40, deep in the chart, or off the chart entirely). This program
//   bins those pairs by gap, applies the measured persistence and origin mix to the chart as it
//   stands in the last read — the incoming shares spread over the corresponding games of the
//   last read weighted by their player counts — and blends the bins by how much of the window
//   each covers. The share measured to arrive from deep or off the chart is left unnamed, which
//   the scoring spreads over the chart, as a hedge no named game can give.
//   Deterministic: no clock (the window comes from task.json), no randomness.

import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

export interface ChartGame {
  universe_id: number;
  rank: number;
  playing: number | null;
}

export interface Read {
  t: number; // epoch ms
  chart: ChartGame[]; // in chart order
}

interface Task {
  history_until: string;
  window: { from: string; until: string };
  places: number;
}

const PLACES = 10;
/** Horizon bins (hours past the earlier read) and the share of the 72 window hours each covers. */
const BINS: { lo: number; hi: number; w: number }[] = [
  { lo: 0, hi: 12, w: 12 / 72 },
  { lo: 12, hi: 30, w: 18 / 72 },
  { lo: 30, hi: 54, w: 24 / 72 },
  { lo: 54, hi: 96, w: 18 / 72 },
];

type Origin = 'top10' | 'r11_20' | 'r21_40' | 'deep' | 'new';

function originOf(rankInEarlier: number | null): Origin {
  if (rankInEarlier === null) return 'new';
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
    reads.push({ t: Date.parse(d.at), chart });
  }
  reads.sort((a, b) => a.t - b.t);
  return reads;
}

/** For each place and bin: persistence (same occupant) and the origin mix of a changed occupant. */
interface PlaceStats {
  persist: number[]; // per bin
  mix: Record<Origin, number>[]; // per bin, shares among changes
}

function measure(reads: Read[]): PlaceStats[] {
  const stats: PlaceStats[] = [];
  for (let k = 0; k < PLACES; k++) {
    const perBin = BINS.map(() => ({
      same: 0,
      changed: 0,
      mix: { top10: 0, r11_20: 0, r21_40: 0, deep: 0, new: 0 } as Record<Origin, number>,
    }));
    for (let i = 0; i < reads.length; i++) {
      const a = reads[i]!;
      if (a.chart.length <= k) continue;
      const rankInA = new Map(a.chart.map((g) => [g.universe_id, g.rank] as const));
      for (let j = i + 1; j < reads.length; j++) {
        const b = reads[j]!;
        if (b.chart.length <= k) continue;
        const dh = (b.t - a.t) / 3_600_000;
        const bin = BINS.findIndex((x) => dh >= x.lo && dh < x.hi);
        if (bin === -1) continue;
        const occA = a.chart[k]!.universe_id;
        const occB = b.chart[k]!.universe_id;
        const cell = perBin[bin]!;
        if (occA === occB) cell.same += 1;
        else {
          cell.changed += 1;
          cell.mix[originOf(rankInA.get(occB) ?? null)] += 1;
        }
      }
    }
    stats.push({
      persist: perBin.map((c) => (c.same + c.changed === 0 ? 0.5 : c.same / (c.same + c.changed))),
      mix: perBin.map((c) => {
        const tot = Object.values(c.mix).reduce((x, y) => x + y, 0);
        const out = { top10: 0, r11_20: 0, r21_40: 0, deep: 0, new: 0 } as Record<Origin, number>;
        if (tot === 0) {
          out.top10 = 0.45; out.r11_20 = 0.25; out.r21_40 = 0.15; out.deep = 0.08; out.new = 0.07;
        } else {
          (Object.keys(out) as Origin[]).forEach((o) => (out[o] = c.mix[o] / tot));
        }
        return out;
      }),
    });
  }
  return stats;
}

export interface Forecast {
  probabilities: { universe_id: number; place: number; p: number }[];
  top10: number[];
}

export function buildForecast(reads: Read[]): Forecast {
  const last = reads[reads.length - 1]!;
  const stats = measure(reads);
  const weightOf = (g: ChartGame): number => Math.max(1, g.playing ?? 1);
  const byOrigin = (origin: Origin): ChartGame[] =>
    last.chart.filter((g) => originOf(g.rank) === origin);
  // dist[place][universe_id] = probability
  const dist: Map<number, Map<number, number>> = new Map();
  const add = (place: number, id: number, p: number) => {
    if (p <= 0) return;
    const m = dist.get(place) ?? new Map<number, number>();
    m.set(id, (m.get(id) ?? 0) + p);
    dist.set(place, m);
  };
  for (let k = 1; k <= PLACES; k++) {
    const occupant = last.chart[k - 1];
    if (!occupant) continue;
    const st = stats[k - 1]!;
    for (let b = 0; b < BINS.length; b++) {
      const w = BINS[b]!.w;
      const persist = st.persist[b]!;
      const mix = st.mix[b]!;
      add(k, occupant.universe_id, w * persist);
      const changeShare = w * (1 - persist);
      for (const origin of Object.keys(mix) as Origin[]) {
        const share = changeShare * mix[origin]!;
        if (share <= 0) continue;
        if (origin === 'deep' || origin === 'new') continue; // left unnamed on purpose
        const pool = byOrigin(origin).filter((g) => g.universe_id !== occupant.universe_id);
        const totW = pool.reduce((s, g) => s + weightOf(g), 0);
        if (totW <= 0) continue;
        for (const g of pool) add(k, g.universe_id, (share * weightOf(g)) / totW);
      }
    }
  }
  const probabilities: Forecast['probabilities'] = [];
  const presence = new Map<number, number>();
  const expPlace = new Map<number, number>();
  for (let k = 1; k <= PLACES; k++) {
    const m = dist.get(k);
    if (!m) continue;
    const rows = [...m.entries()].sort((a, b) => b[1] - a[1]).slice(0, 100);
    const sum = rows.reduce((s, [, p]) => s + p, 0);
    const scale = sum > 0.995 ? 0.995 / sum : 1;
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
  return { probabilities, top10: ranked.map((r) => r.id) };
}

export function run(inputDir: string, outputDir: string): void {
  const task = JSON.parse(readFileSync(join(inputDir, 'task.json'), 'utf8')) as Task;
  void task;
  const reads = parseHistory(readFileSync(join(inputDir, 'history.ndjson'), 'utf8'));
  if (reads.length === 0) throw new Error('no reads in history');
  const forecast = buildForecast(reads);
  mkdirSync(outputDir, { recursive: true });
  writeFileSync(join(outputDir, 'forecast.json'), JSON.stringify(forecast), 'utf8');
}

const isMain = process.argv[1] !== undefined && import.meta.url === new URL(`file://${process.argv[1]}`).href;
if (isMain) {
  run(process.env.MAKEFUN_INPUT_DIR ?? '/input', process.env.MAKEFUN_OUTPUT_DIR ?? '/output');
}
