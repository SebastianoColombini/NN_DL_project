// Saliency GUI: state, data flow and rendering.
import { api } from './js/api.js';
import { colorize, decodeGray, gradientCss, loadImage } from './js/colormap.js';
import { lossChart } from './js/chart.js';
import { h, icon } from './js/dom.js';

const $ = (sel) => document.querySelector(sel);

const state = {
  health: null,
  models: [],          // from /api/models
  samples: [],
  images: [],          // loaded images (samples and uploads)
  current: null,       // key of the selected image
  mode: 'v1.1',        // model id or 'compare'
  view: 'overlay',     // overlay | map | side
  cmap: 'jet',
  opacity: 0.7,
  showGt: true,
};

let uploadSeq = 0;
let liveOverlays = []; // overlay canvases redrawn on opacity change

// ---------- helpers ----------

const modelById = (id) => state.models.find((m) => m.id === id);
const isAvailable = (id) => {
  const m = modelById(id);
  return !!m && m.available !== false;
};
const currentImage = () => state.images.find((i) => i.key === state.current) || null;

function selectedModels() {
  if (state.mode === 'compare') return state.models.filter((m) => m.available !== false).map((m) => m.id);
  return [state.mode];
}

const fmtParams = (n) => `${(n / 1e6).toFixed(2)} M`;
const fmtMs = (ms) => (ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${Math.round(ms)} ms`);
const baseName = (name) => name.replace(/\.[^.]+$/, '') || 'image';

function modelLabel(id) {
  const m = modelById(id);
  return m ? m.label : id;
}

// ---------- alerts ----------

function showError(message, title = 'Error') {
  const box = $('#alerts');
  const alert = h('div', { class: 'alert', role: 'alert' },
    icon('alert'),
    h('div', { class: 'alert-body' }, h('strong', {}, title), h('p', {}, message)),
    h('button', { class: 'icon-btn', type: 'button', 'aria-label': 'Dismiss', onclick: () => alert.remove() }, icon('close')));
  box.append(alert);
}

// ---------- image entries ----------

function makeEntry({ key, kind, name, title, src, file = null, sampleId = null, hasGt = false, width = 0, height = 0 }) {
  const entry = {
    key, kind, name, title, src, file, sampleId, hasGt, width, height,
    gtFile: null,
    gen: 0,              // bumped when the GT changes; stale responses are dropped
    results: new Map(),  // model id -> {inference_ms, input_size, metrics, gray}
    errors: new Map(),   // model id -> message
    pending: new Set(),
    gt: null,            // {w, h, data}
    colored: new Map(),  // `${model}|${cmap}` -> canvas
    verified: false,     // server accepted the image at least once
    img: null,
  };
  entry.img = loadImage(src).then((img) => {
    if (!entry.width) {
      entry.width = img.naturalWidth;
      entry.height = img.naturalHeight;
      if (entry.key === state.current) renderResults();
    }
    return img;
  }).catch(() => null);
  return entry;
}

function resetPredictions(entry) {
  entry.gen += 1;
  entry.results = new Map();
  entry.errors = new Map();
  entry.pending = new Set();
  entry.gt = null;
  entry.colored = new Map();
}

function addSample(sample) {
  const key = `s:${sample.id}`;
  if (!state.images.some((i) => i.key === key)) {
    state.images.push(makeEntry({
      key, kind: 'sample', name: sample.image_url.split('/').pop(), title: sample.title,
      src: sample.image_url, sampleId: sample.id, hasGt: sample.has_gt, width: sample.width, height: sample.height,
    }));
  }
  select(key);
}

function addFiles(files) {
  const list = [...files];
  if (!list.length) return;
  let first = null;
  for (const file of list) {
    const key = `u:${++uploadSeq}`;
    state.images.push(makeEntry({
      key, kind: 'upload', name: file.name || 'image', title: file.name || 'Pasted image',
      src: URL.createObjectURL(file), file,
    }));
    first = first || key;
  }
  select(first);
}

function removeImage(key) {
  const idx = state.images.findIndex((i) => i.key === key);
  if (idx < 0) return;
  const [entry] = state.images.splice(idx, 1);
  if (entry.kind === 'upload') URL.revokeObjectURL(entry.src);
  if (state.current === key) {
    const next = state.images[Math.min(idx, state.images.length - 1)];
    state.current = next ? next.key : null;
    if (next) ensureResults(next);
  }
  renderLoaded();
  renderResults();
}

function select(key) {
  state.current = key;
  renderLoaded();
  const entry = currentImage();
  if (entry) ensureResults(entry);
  renderResults();
}

function setGtFile(entry, file) {
  entry.gtFile = file;
  resetPredictions(entry);
  ensureResults(entry);
  renderLoaded();
  renderResults();
}

// ---------- prediction ----------

async function ensureResults(entry) {
  const wanted = selectedModels().filter((id) =>
    isAvailable(id) && !entry.results.has(id) && !entry.pending.has(id) && !entry.errors.has(id));
  if (!wanted.length) return;

  const gen = entry.gen;
  const pending = entry.pending;
  wanted.forEach((id) => pending.add(id));

  const form = new FormData();
  if (entry.kind === 'sample') {
    form.append('sample_id', entry.sampleId);
  } else {
    form.append('image', entry.file, entry.file.name || 'image.png');
    if (entry.gtFile) form.append('gt', entry.gtFile, entry.gtFile.name || 'fixmap.png');
  }
  form.append('models', wanted.join(','));

  try {
    const res = await api.predict(form);
    if (entry.gen !== gen) return;
    entry.verified = true;
    entry.width = res.image.width;
    entry.height = res.image.height;
    const [gt, ...maps] = await Promise.all([
      res.gt && res.gt.png ? decodeGray(res.gt.png) : null,
      ...res.results.map((r) => decodeGray(r.map_png)),
    ]);
    if (entry.gen !== gen) return;
    if (gt) entry.gt = gt;
    res.results.forEach((r, i) => {
      entry.results.set(r.model, {
        inference_ms: r.inference_ms, input_size: r.input_size, metrics: r.metrics, gray: maps[i],
      });
    });
  } catch (err) {
    if (entry.gen !== gen) return;
    const status = err.status;
    if (status === 400 && entry.kind === 'upload' && !entry.verified) {
      if (entry.gtFile) {
        // Image or fixation map rejected: retry without the map to tell which.
        showError(err.message, `${entry.title}: fixation map not used`);
        wanted.forEach((id) => pending.delete(id));
        setGtFile(entry, null);
        return;
      }
      showError(err.message, `${entry.title}: not an image`);
      removeImage(entry.key);
      return;
    }
    if (status === 400 && entry.gtFile) {
      showError(err.message, `${entry.title}: fixation map rejected`);
      wanted.forEach((id) => pending.delete(id));
      setGtFile(entry, null);
      return;
    }
    wanted.forEach((id) => entry.errors.set(id, err.message));
    showError(err.message, status === 503 ? 'Model weights missing' : `Prediction failed (${entry.title})`);
  } finally {
    wanted.forEach((id) => pending.delete(id));
  }
  if (entry.key === state.current) renderResults();
  renderLoaded();
}

// ---------- canvases ----------

function coloredFor(entry, id, gray, weighted = false) {
  const k = `${id}|${state.cmap}|${weighted ? 'w' : 'o'}`;
  if (!entry.colored.has(k)) entry.colored.set(k, colorize(gray, state.cmap, weighted));
  return entry.colored.get(k);
}

function drawOverlay(canvas, img, colored, opacity) {
  const ctx = canvas.getContext('2d');
  ctx.globalAlpha = 1;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (img) ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  ctx.globalAlpha = opacity;
  ctx.drawImage(colored, 0, 0, canvas.width, canvas.height);
  ctx.globalAlpha = 1;
}

function overlayCanvas(img, colored) {
  const c = document.createElement('canvas');
  c.width = colored.width;
  c.height = colored.height;
  drawOverlay(c, img, colored, state.opacity);
  return c;
}

function mapCanvas(colored) {
  const c = document.createElement('canvas');
  c.width = colored.width;
  c.height = colored.height;
  c.getContext('2d').drawImage(colored, 0, 0);
  return c;
}

function download(canvas, filename) {
  canvas.toBlob((blob) => {
    if (!blob) return showError('Could not encode the PNG.', 'Download failed');
    const url = URL.createObjectURL(blob);
    const a = h('a', { href: url, download: filename });
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }, 'image/png');
}

// ---------- rendering: sidebar ----------

function renderSamples() {
  const box = $('#samples');
  box.replaceChildren(...state.samples.map((smp) =>
    h('div', { role: 'listitem' },
      h('button', {
        class: 'sample', type: 'button', title: `${smp.title} (${smp.width}×${smp.height})`,
        'data-sample-id': smp.id, onclick: () => addSample(smp),
      },
      h('img', { src: smp.image_url, alt: '', loading: 'lazy', width: 96, height: 72 }),
      h('span', {}, smp.title)))));
  if (!state.samples.length) box.replaceChildren(h('p', { class: 'muted small' }, 'No samples available.'));
}

function renderLoaded() {
  const list = $('#loaded');
  if (!state.images.length) {
    list.replaceChildren(h('li', { class: 'muted small empty-line' }, 'Nothing loaded yet.'));
    return;
  }
  list.replaceChildren(...state.images.map((entry) => {
    const active = entry.key === state.current;
    const busy = entry.pending.size > 0;
    const hasGt = entry.kind === 'sample' ? entry.hasGt : !!entry.gtFile;
    const thumb = h('img', { src: entry.src, alt: '', width: 40, height: 40 });
    thumb.addEventListener('error', () => thumb.classList.add('broken'));
    return h('li', { class: `loaded-item${active ? ' active' : ''}` },
      h('button', {
        class: 'loaded-select', type: 'button', 'aria-current': active ? 'true' : null,
        onclick: () => select(entry.key),
      },
      thumb,
      h('span', { class: 'loaded-text' },
        h('span', { class: 'loaded-name' }, entry.title),
        h('span', { class: 'loaded-meta' },
          entry.kind === 'sample' ? 'sample' : 'upload',
          hasGt ? ' · GT' : '',
          busy ? h('span', { class: 'spinner tiny', 'aria-label': 'working' }) : null))),
      h('button', {
        class: 'icon-btn', type: 'button', 'aria-label': `Remove ${entry.title}`,
        onclick: () => removeImage(entry.key),
      }, icon('close')));
  }));
}

function renderModelControls() {
  const note = $('#model-note');
  const missing = state.models.filter((m) => m.available === false);
  for (const input of document.querySelectorAll('input[name="model"]')) {
    const v = input.value;
    const m = modelById(v);
    let disabled = false;
    let why = '';
    if (v === 'compare') {
      disabled = state.models.filter((x) => x.available !== false).length < 2;
      why = disabled ? 'Needs both models available' : 'Run both models side by side';
    } else if (!m) {
      disabled = true;
      why = 'Unknown model';
    } else if (m.available === false) {
      disabled = true;
      why = `Weights not found: ${weightsPath(m)}`;
    } else {
      why = m.label;
    }
    input.disabled = disabled;
    input.checked = v === state.mode;
    input.parentElement.title = why;
    input.parentElement.classList.toggle('disabled', disabled);
  }
  if (missing.length) {
    note.hidden = false;
    note.replaceChildren(...missing.map((m) => h('span', {},
      h('strong', {}, `${m.id} unavailable`), ` — weights file not found (expected ${weightsPath(m)}). `,
      'Set SALIENCY_WEIGHTS_DIR or add the file, then restart.')));
  } else {
    note.hidden = true;
  }
}

function weightsPath(m) {
  const hm = state.health && state.health.models && state.health.models.find((x) => x.id === m.id);
  return (hm && hm.weights_path) || m.weights_file;
}

function renderBadge() {
  const badge = $('#device-badge');
  const text = badge.querySelector('.badge-text');
  const hl = state.health;
  if (!hl) {
    badge.dataset.state = 'error';
    text.textContent = 'Server offline';
    return;
  }
  const d = hl.device || {};
  let label = 'CPU';
  if (d.type === 'cuda') {
    label = `GPU · ${d.name || 'CUDA'}`;
    if (d.vram_total_mb) label += ` · ${(d.vram_total_mb / 1024).toFixed(d.vram_total_mb % 1024 ? 1 : 0)} GB`;
  } else if (d.type === 'mps') {
    label = d.name || 'Apple MPS';
  } else if (d.name) {
    label = `CPU · ${d.name}`;
  }
  badge.dataset.state = d.type || 'cpu';
  badge.dataset.device = d.type || 'cpu';
  text.textContent = label;
  const v = hl.versions || {};
  badge.title = `Inference device: ${label}` + (d.vram_free_mb ? ` (${d.vram_free_mb} MB free)` : '');
  $('#versions').textContent = Object.entries(v).map(([k, val]) => `${k} ${val}`).join(' · ');
}

// ---------- rendering: results ----------

function figure(caption, media, { tag = null, actions = null, cls = '' } = {}) {
  return h('figure', { class: `figure ${cls}` },
    h('div', { class: 'frame' }, media),
    h('figcaption', {},
      h('span', { class: 'fig-label' }, caption, tag ? h('span', { class: 'tag' }, tag) : null),
      actions ? h('span', { class: 'fig-actions' }, actions) : null));
}

function placeholder(entry, content) {
  const ratio = entry.width && entry.height ? `${entry.width} / ${entry.height}` : '4 / 3';
  return h('div', { class: 'placeholder', style: `aspect-ratio: ${ratio}` }, content);
}

function renderResults() {
  const root = $('#results');
  liveOverlays = [];
  const entry = currentImage();
  if (!entry) {
    root.replaceChildren(h('section', { class: 'card empty' },
      h('div', { class: 'empty-art', 'aria-hidden': 'true' }),
      h('h2', {}, 'Pick a sample or drop an image'),
      h('p', { class: 'muted' }, 'The predicted saliency map is shown over the image, next to the ground-truth fixation map when available.')));
    return;
  }

  const ids = selectedModels();
  const figures = [];
  const origImg = h('img', { src: entry.src, alt: entry.title, class: 'orig' });
  figures.push(figure('Original', origImg, { tag: entry.width ? `${entry.width}×${entry.height}` : null }));

  const base = baseName(entry.name);
  for (const id of ids) {
    const res = entry.results.get(id);
    const label = modelLabel(id);
    if (!res) {
      const err = entry.errors.get(id) || (isAvailable(id) ? null : `Model ${id} is not available.`);
      const body = err
        ? h('div', { class: 'ph-error' }, icon('alert'), h('p', {}, err),
          h('button', { class: 'btn small', type: 'button' }, icon('retry'), 'Retry'))
        : h('div', { class: 'ph-loading' }, h('span', { class: 'spinner', 'aria-hidden': 'true' }), h('p', {}, `Predicting with ${id}…`));
      const views = state.view === 'side' ? ['Map', 'Overlay'] : [state.view === 'map' ? 'Map' : 'Overlay'];
      for (const v of views) {
        figures.push(figure(label, placeholder(entry, body.cloneNode(true)), { tag: v, cls: err ? 'is-error' : 'is-loading' }));
      }
      continue;
    }
    const colored = coloredFor(entry, id, res.gray);
    const dl = h('span', { class: 'dl-group' },
      h('button', { class: 'btn small ghost', type: 'button', title: 'Download map PNG', 'data-dl': `${id}-map`,
        onclick: () => download(mapCanvas(coloredFor(entry, id, res.gray)), `${base}_${id}_map.png`) }, icon('download'), 'Map'),
      h('button', { class: 'btn small ghost', type: 'button', title: 'Download overlay PNG', 'data-dl': `${id}-overlay`,
        onclick: async () => download(overlayCanvas(await entry.img, coloredFor(entry, id, res.gray, true)), `${base}_${id}_overlay.png`) }, icon('download'), 'Overlay'));
    const time = `${fmtMs(res.inference_ms)}`;
    if (state.view === 'map' || state.view === 'side') {
      figures.push(figure(label, mapCanvas(colored), { tag: `Map · ${time}`, actions: state.view === 'side' ? null : dl, cls: 'model' }));
    }
    if (state.view === 'overlay' || state.view === 'side') {
      const c = h('canvas', { width: colored.width, height: colored.height, class: 'overlay', 'data-model': id });
      liveOverlays.push({ canvas: c, colored: coloredFor(entry, id, res.gray, true), entry });
      figures.push(figure(label, c, { tag: `Overlay · ${time}`, actions: dl, cls: 'model' }));
    }
  }

  if (entry.gt && state.showGt) {
    const colored = coloredFor(entry, 'gt', entry.gt);
    if (state.view === 'overlay') {
      const c = h('canvas', { width: colored.width, height: colored.height, class: 'overlay', 'data-model': 'gt' });
      liveOverlays.push({ canvas: c, colored: coloredFor(entry, 'gt', entry.gt, true), entry });
      figures.push(figure('Ground truth', c, { tag: 'Fixation map', cls: 'gt' }));
    } else {
      figures.push(figure('Ground truth', mapCanvas(colored), { tag: 'Fixation map', cls: 'gt' }));
    }
  }

  const grid = h('div', { class: 'figures', 'data-count': figures.length }, figures);
  // Rebind retry buttons on cloned placeholders.
  grid.querySelectorAll('.ph-error .btn').forEach((b) => b.addEventListener('click', () => {
    entry.errors.clear();
    ensureResults(entry);
    renderResults();
  }));

  entry.img.then((img) => {
    for (const o of liveOverlays) if (o.entry === entry) drawOverlay(o.canvas, img, o.colored, state.opacity);
  });
  for (const o of liveOverlays) drawOverlay(o.canvas, null, o.colored, state.opacity);

  const gtControl = entry.kind === 'upload'
    ? (entry.gtFile
      ? h('span', { class: 'gt-chip' }, h('span', {}, `GT: ${entry.gtFile.name}`),
        h('button', { class: 'icon-btn', type: 'button', 'aria-label': 'Remove fixation map', onclick: () => setGtFile(entry, null) }, icon('close')))
      : h('label', { class: 'btn small', for: 'gt-input' }, icon('plus'), 'Add fixation map',
        h('input', { id: 'gt-input', class: 'visually-hidden', type: 'file', accept: 'image/*',
          onchange: (e) => { if (e.target.files[0]) setGtFile(entry, e.target.files[0]); } })))
    : null;

  const busy = entry.pending.size > 0;
  const head = h('div', { class: 'results-head' },
    h('div', {},
      h('h2', { class: 'card-title' }, entry.title),
      h('p', { class: 'muted small' }, entry.kind === 'sample' ? 'Sample from MIT1003' : `Uploaded · ${entry.name}`,
        busy ? h('span', { class: 'status-busy' }, h('span', { class: 'spinner tiny', 'aria-hidden': 'true' }), ' predicting…') : null)),
    gtControl);

  root.replaceChildren(h('section', { class: 'card results', 'aria-busy': busy ? 'true' : 'false', 'data-image': entry.key },
    head, grid, metricsTable(entry, ids)));
}

// "v1.1 · CNN encoder" -> id + secondary part (hidden on narrow screens)
function labelParts(id) {
  const [first, ...rest] = modelLabel(id).split(' · ');
  return [first, rest.length ? h('span', { class: 'label-rest' }, ` · ${rest.join(' · ')}`) : null];
}

function metricsTable(entry, ids) {
  const rows = ids.map((id) => [id, entry.results.get(id)]).filter(([, r]) => r);
  if (!rows.length) return null;
  const withMetrics = rows.some(([, r]) => r.metrics);
  const best = {};
  if (withMetrics && rows.length > 1) {
    for (const [k, dir] of [['cc', 1], ['kld', -1], ['sim', 1]]) {
      const vals = rows.map(([, r]) => (r.metrics ? r.metrics[k] : NaN));
      const target = dir > 0 ? Math.max(...vals) : Math.min(...vals);
      best[k] = vals.indexOf(target);
    }
  }
  const head = ['Model'];
  if (withMetrics) head.push(h('span', {}, 'CC ', h('span', { class: 'dir' }, '↑')),
    h('span', {}, 'KLD ', h('span', { class: 'dir' }, '↓')), h('span', {}, 'SIM ', h('span', { class: 'dir' }, '↑')));
  head.push('Inference', h('span', { class: 'col-input' }, 'Input'));
  const table = h('table', { class: 'metrics' },
    h('thead', {}, h('tr', {}, head.map((x, i) => h('th', {
      scope: 'col', class: [i ? 'num' : '', i === head.length - 1 ? 'col-input' : ''].join(' ').trim() || null,
    }, x)))),
    h('tbody', {}, rows.map(([id, r], ri) => h('tr', {},
      h('th', { scope: 'row' }, ...labelParts(id)),
      withMetrics ? ['cc', 'kld', 'sim'].map((k) => h('td', { class: `num${best[k] === ri ? ' best' : ''}` },
        r.metrics ? r.metrics[k].toFixed(3) : '—')) : null,
      h('td', { class: 'num' }, fmtMs(r.inference_ms)),
      h('td', { class: 'num col-input' }, r.input_size ? `${r.input_size[0]}×${r.input_size[1]}` : '—')))));
  const note = withMetrics
    ? 'Metrics against the fixation map at model resolution. Inference = forward pass only.'
    : (entry.kind === 'upload' ? 'Add a fixation map to compute CC, KLD and SIM.' : 'No fixation map: metrics unavailable.');
  return h('div', { class: 'metrics-wrap' }, h('div', { class: 'table-scroll' }, table), h('p', { class: 'muted small' }, note));
}

// ---------- rendering: model info ----------

function renderModelInfo() {
  const box = $('#model-info');
  const ids = state.models.map((m) => m.id);
  const cards = ids.map(modelById).filter(Boolean).map((m) => {
    const t = m.training || {};
    const ref = m.reference || {};
    const facts = [
      ['Input', m.input_size ? `${m.input_size[0]}×${m.input_size[1]}` : '—'],
      ['Params', m.params ? fmtParams(m.params) : '—'],
      ['Best epoch', t.best_epoch ? `${t.best_epoch} / ${t.epochs_run ?? '—'}` : '—'],
      ['Val loss', t.best_val_loss != null ? t.best_val_loss.toFixed(4) : '—'],
    ];
    const hp = t.hparams || {};
    return h('article', { class: 'model-card', 'data-model': m.id },
      h('header', {},
        h('h3', {}, m.label),
        h('span', { class: `pill ${m.available === false ? 'pill-bad' : 'pill-ok'}` }, m.available === false ? 'weights missing' : 'ready')),
      m.description ? h('p', { class: 'muted small' }, m.description) : null,
      h('dl', { class: 'facts' }, facts.map(([k, v]) => h('div', {}, h('dt', {}, k), h('dd', {}, v)))),
      h('div', { class: 'ref' },
        h('p', { class: 'section-label' }, `Reference · ${ref.dataset || 'MIT1003'}`),
        h('dl', { class: 'facts ref-facts' },
          [['Total', ref.total_loss, 4], ['KLD ↓', ref.kld, 4], ['CC ↑', ref.cc, 4], ['SIM ↑', ref.sim, 4]]
            .map(([k, v, d]) => h('div', {}, h('dt', {}, k), h('dd', {}, v != null ? v.toFixed(d) : '—'))))),
      h('div', { class: 'chart-wrap' },
        h('p', { class: 'section-label' }, 'Training loss (SALICON)'),
        lossChart(t, `${m.id} train and validation loss per epoch`),
        h('p', { class: 'legend' },
          h('span', { class: 'key key-train' }), 'train',
          h('span', { class: 'key key-val' }), 'validation',
          h('span', { class: 'key key-best' }), 'best epoch')),
      Object.keys(hp).length
        ? h('details', { class: 'hparams' }, h('summary', {}, 'Hyperparameters'),
          h('dl', { class: 'kv' }, Object.entries(hp).map(([k, v]) => [h('dt', {}, k), h('dd', {}, String(v))])))
        : null);
  });
  box.dataset.count = String(cards.length);
  box.replaceChildren(...(cards.length ? cards : [h('p', { class: 'muted' }, 'Model information unavailable.')]));
}

// ---------- events ----------

function bindControls() {
  for (const input of document.querySelectorAll('input[name="model"]')) {
    input.addEventListener('change', () => {
      state.mode = input.value;
      renderModelInfo();
      const entry = currentImage();
      if (entry) ensureResults(entry);
      renderResults();
    });
  }
  for (const input of document.querySelectorAll('input[name="view"]')) {
    input.addEventListener('change', () => { state.view = input.value; renderResults(); });
  }
  const cmap = $('#cmap');
  const preview = $('#cmap-preview');
  const setPreview = () => { preview.style.backgroundImage = gradientCss(state.cmap); };
  cmap.addEventListener('change', () => { state.cmap = cmap.value; setPreview(); renderResults(); });
  setPreview();

  const opacity = $('#opacity');
  const out = $('#opacity-out');
  opacity.addEventListener('input', () => {
    state.opacity = Number(opacity.value);
    out.textContent = `${Math.round(state.opacity * 100)}%`;
    for (const o of liveOverlays) o.entry.img.then((img) => drawOverlay(o.canvas, img, o.colored, state.opacity));
  });
  $('#show-gt').addEventListener('change', (e) => { state.showGt = e.target.checked; renderResults(); });

  const fileInput = $('#file-input');
  fileInput.addEventListener('change', () => { addFiles(fileInput.files); fileInput.value = ''; });

  const zone = $('#dropzone');
  let depth = 0;
  window.addEventListener('dragover', (e) => e.preventDefault());
  window.addEventListener('drop', (e) => e.preventDefault());
  zone.addEventListener('dragenter', (e) => { e.preventDefault(); depth++; zone.classList.add('over'); });
  zone.addEventListener('dragleave', () => { if (--depth <= 0) { depth = 0; zone.classList.remove('over'); } });
  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    depth = 0;
    zone.classList.remove('over');
    if (e.dataTransfer && e.dataTransfer.files.length) addFiles(e.dataTransfer.files);
  });
  window.addEventListener('paste', (e) => {
    const files = [...(e.clipboardData ? e.clipboardData.files : [])].filter((f) => f.type.startsWith('image/'));
    if (files.length) addFiles(files);
  });
}

// ---------- boot ----------

async function boot() {
  bindControls();
  renderLoaded();
  renderResults();
  const [health, models, samples] = await Promise.allSettled([api.health(), api.models(), api.samples()]);

  state.health = health.status === 'fulfilled' ? health.value : null;
  if (health.status === 'rejected') showError(health.reason.message, 'Server not reachable');
  renderBadge();

  if (models.status === 'fulfilled') {
    const healthAvail = new Map(((state.health && state.health.models) || []).map((m) => [m.id, m.available]));
    state.models = models.value.models.map((m) => ({
      ...m, available: m.available !== false && healthAvail.get(m.id) !== false,
    }));
    const def = models.value.default;
    const firstOk = state.models.find((m) => m.available);
    state.mode = isAvailable(def) ? def : (firstOk ? firstOk.id : def);
  } else {
    showError(models.reason.message, 'Could not load models');
  }
  renderModelControls();
  renderModelInfo();

  if (samples.status === 'fulfilled') state.samples = samples.value.samples || [];
  else showError(samples.reason.message, 'Could not load samples');
  renderSamples();
  document.body.dataset.ready = 'true';
}

boot();
