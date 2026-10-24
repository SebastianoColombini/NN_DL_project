// 256-entry RGB lookup tables and grayscale -> RGBA colorization.

const STOPS = {
  // Evenly spaced anchors of the matplotlib maps, linearly interpolated.
  inferno: ['000004', '1b0c41', '4a0c6b', '781c6d', 'a52c60', 'cf4446', 'ed6925', 'fb9b06', 'f7d13d', 'fcffa4'],
  viridis: ['440154', '482878', '3e4989', '31688e', '26828e', '1f9e89', '35b779', '6ece58', 'b5de2b', 'fde725'],
};

const clamp01 = (v) => Math.min(1, Math.max(0, v));

function fromStops(hex) {
  const pts = hex.map((s) => [0, 2, 4].map((i) => parseInt(s.slice(i, i + 2), 16)));
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = (i / 255) * (pts.length - 1);
    const k = Math.min(pts.length - 2, Math.floor(t));
    const f = t - k;
    for (let c = 0; c < 3; c++) lut[i * 3 + c] = Math.round(pts[k][c] + (pts[k + 1][c] - pts[k][c]) * f);
  }
  return lut;
}

function fromFn(fn) {
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const rgb = fn(i / 255);
    for (let c = 0; c < 3; c++) lut[i * 3 + c] = Math.round(clamp01(rgb[c]) * 255);
  }
  return lut;
}

export const LUTS = {
  jet: fromFn((x) => [1.5 - Math.abs(4 * x - 3), 1.5 - Math.abs(4 * x - 2), 1.5 - Math.abs(4 * x - 1)]),
  inferno: fromStops(STOPS.inferno),
  viridis: fromStops(STOPS.viridis),
  gray: fromFn((x) => [x, x, x]),
};

/** CSS linear-gradient for a colormap preview bar. */
export function gradientCss(name) {
  const lut = LUTS[name];
  const stops = [];
  for (let i = 0; i <= 8; i++) {
    const j = Math.round((i / 8) * 255) * 3;
    stops.push(`rgb(${lut[j]} ${lut[j + 1]} ${lut[j + 2]}) ${(i / 8) * 100}%`);
  }
  return `linear-gradient(90deg, ${stops.join(', ')})`;
}

// Overlay alpha per (stretched) saliency value: low saliency stays transparent so the photo shows through.
const ALPHA = Uint8ClampedArray.from({ length: 256 }, (_, v) => 255 * Math.min(1, Math.pow(v / 255, 0.8) * 1.6));

/**
 * gray: {w, h, data: Uint8Array} -> canvas colored with the LUT.
 * Values are stretched to the map's own maximum (display only). With `weighted`, the alpha
 * follows the saliency (for overlays); otherwise the canvas is opaque.
 */
export function colorize(gray, name, weighted = false) {
  const lut = LUTS[name];
  const canvas = document.createElement('canvas');
  canvas.width = gray.w;
  canvas.height = gray.h;
  const ctx = canvas.getContext('2d');
  const out = ctx.createImageData(gray.w, gray.h);
  const px = out.data;
  const src = gray.data;
  let max = 0;
  for (let i = 0; i < src.length; i++) if (src[i] > max) max = src[i];
  const scale = max > 0 ? 255 / max : 0;
  for (let i = 0, j = 0; i < src.length; i++, j += 4) {
    const v = Math.round(src[i] * scale);
    const k = v * 3;
    px[j] = lut[k];
    px[j + 1] = lut[k + 1];
    px[j + 2] = lut[k + 2];
    px[j + 3] = weighted ? ALPHA[v] : 255;
  }
  ctx.putImageData(out, 0, 0);
  return canvas;
}

/** Decode an image URL (e.g. a PNG data URL) into a single-channel array. */
export async function decodeGray(url) {
  const img = await loadImage(url);
  const w = img.naturalWidth;
  const h = img.naturalHeight;
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(img, 0, 0);
  const rgba = ctx.getImageData(0, 0, w, h).data;
  const data = new Uint8Array(w * h);
  for (let i = 0, j = 0; i < data.length; i++, j += 4) data[i] = rgba[j];
  return { w, h, data };
}

export function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.decoding = 'async';
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('Could not decode image'));
    img.src = url;
  });
}
