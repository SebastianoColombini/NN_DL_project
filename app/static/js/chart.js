// Train/validation loss curve as inline SVG.
import { s } from './dom.js';

const W = 360;
const H = 170;
const PAD = { l: 36, r: 10, t: 10, b: 26 };

function niceTicks(min, max, n = 4) {
  const span = max - min || 1;
  const step0 = span / n;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((v) => v >= step0);
  const ticks = [];
  for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) ticks.push(+v.toFixed(6));
  return ticks;
}

export function lossChart(training, title) {
  const train = training.train_loss || [];
  const val = training.val_loss || [];
  const n = Math.max(train.length, val.length);
  const all = [...train, ...val].filter(Number.isFinite);
  if (!n || !all.length) return s('svg', { class: 'chart', viewBox: `0 0 ${W} ${H}` });
  const lo = Math.min(...all);
  const hi = Math.max(...all);
  const yMin = lo - (hi - lo) * 0.05;
  const yMax = hi + (hi - lo) * 0.05;
  const x = (i) => PAD.l + (n === 1 ? 0 : (i / (n - 1)) * (W - PAD.l - PAD.r));
  const y = (v) => PAD.t + (1 - (v - yMin) / (yMax - yMin || 1)) * (H - PAD.t - PAD.b);
  const path = (arr) => arr.map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join('');

  const svg = s('svg', { class: 'chart', viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': title });
  svg.append(s('title', {}, title));
  for (const t of niceTicks(yMin, yMax)) {
    svg.append(s('line', { class: 'chart-grid', x1: PAD.l, x2: W - PAD.r, y1: y(t), y2: y(t) }));
    svg.append(s('text', { class: 'chart-tick', x: PAD.l - 6, y: y(t) + 3.5, 'text-anchor': 'end' }, String(t)));
  }
  for (const t of niceTicks(1, n, 5).filter((v) => Number.isInteger(v))) {
    svg.append(s('text', { class: 'chart-tick', x: x(t - 1), y: H - PAD.b + 14, 'text-anchor': 'middle' }, String(t)));
  }
  svg.append(s('text', { class: 'chart-tick', x: W - PAD.r, y: H - 2, 'text-anchor': 'end' }, 'epoch'));

  const best = training.best_epoch;
  if (best && best <= n) {
    svg.append(s('line', { class: 'chart-best', x1: x(best - 1), x2: x(best - 1), y1: PAD.t, y2: H - PAD.b }));
  }
  svg.append(s('path', { class: 'chart-line chart-train', d: path(train) }));
  svg.append(s('path', { class: 'chart-line chart-val', d: path(val) }));
  if (best && val[best - 1] != null) {
    svg.append(s('circle', { class: 'chart-dot', cx: x(best - 1), cy: y(val[best - 1]), r: 3.5 }));
  }
  return svg;
}
