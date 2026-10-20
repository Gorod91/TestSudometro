// Parti comuni alle pagine web: caricare il classificatore addestrato, farlo girare su una foto, disegnare la mappa.

import { GH, GW, loadBase, fitImage, predict, cropBox } from './modello.js';
import { SCORE_TAPS, embed, predictScore } from './punteggio.js';

const TF_URL = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
const ROOT = new URL('../', import.meta.url);

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = src;
    s.onload = resolve;
    s.onerror = () => reject(new Error('TensorFlow.js non si carica (serve la rete)'));
    document.head.append(s);
  });
}

async function loadTf() {
  if (!window.tf) await loadScript(TF_URL);
  await window.tf.ready();
  return window.tf;
}

async function readVendor(p) {
  const r = await fetch(new URL(p, new URL('vendor/mobilenet_v2_0.35/', ROOT)));
  if (!r.ok) throw new Error(`Manca ${p}`);
  return r.arrayBuffer();
}

/** Classificatore delle zone addestrato (modello/), oppure null se non c'è ancora. */
export async function loadClassifier() {
  const res = await fetch(new URL('modello/info.json', ROOT), { cache: 'no-store' });
  if (!res.ok) return null;
  const info = await res.json();
  const tf = await loadTf();
  const [base, head] = await Promise.all([
    loadBase(tf, readVendor),
    tf.loadLayersModel(new URL('modello/testa/model.json', ROOT).href, { requestInit: { cache: 'no-store' } }),
  ]);
  return {
    info,
    /** source: canvas o immagine → { prob: probabilità per riquadro (GH×GW), box: ritaglio usato, in px della foto } */
    run(source) {
      const out = tf.tidy(() => predict(tf, base, head, fitImage(tf, tf.browser.fromPixels(source)).expandDims(0)));
      const prob = out.dataSync();
      out.dispose();
      const w = source.naturalWidth || source.width;
      const h = source.naturalHeight || source.height;
      return { prob, box: cropBox(w, h) };
    },
  };
}

/** Punteggio 0-100 addestrato sui voti (modello-punteggio/), oppure null se non c'è ancora. */
export async function loadScorer() {
  const res = await fetch(new URL('modello-punteggio/modello.json', ROOT), { cache: 'no-store' });
  if (!res.ok) return null;
  const model = await res.json();
  const tf = await loadTf();
  const base = await loadBase(tf, readVendor, model.livelli || SCORE_TAPS);
  return {
    info: model,
    /** source: canvas o immagine → punteggio 0-100 */
    run(source) {
      const e = tf.tidy(() => embed(tf, base, fitImage(tf, tf.browser.fromPixels(source)).expandDims(0)));
      const row = e.dataSync();
      e.dispose();
      return Math.round(predictScore(model, row));
    },
  };
}

/**
 * Disegna la mappa sopra la foto già disegnata in ctx (stesse misure della foto).
 * soft: sfumata; altrimenti a riquadri. Sotto soglia non si colora.
 */
export function drawHeat(ctx, { prob, box }, threshold, { soft = false } = {}) {
  const small = document.createElement('canvas');
  small.width = GW;
  small.height = GH;
  const sctx = small.getContext('2d');
  const img = sctx.createImageData(GW, GH);
  for (let i = 0; i < GW * GH; i++) {
    const p = prob[i];
    const a = p < threshold ? (soft ? p / threshold / 4 : 0) : 0.45 + 0.45 * ((p - threshold) / Math.max(1e-6, 1 - threshold));
    img.data.set([255, 45, 85, Math.round(a * 255)], i * 4);
  }
  sctx.putImageData(img, 0, 0);
  ctx.save();
  ctx.imageSmoothingEnabled = soft;
  ctx.drawImage(small, box.x, box.y, box.w, box.h);
  ctx.restore();
}

/** Quota di riquadri sopra soglia (0-1). */
export function share(prob, threshold) {
  let n = 0;
  for (const p of prob) if (p >= threshold) n++;
  return n / prob.length;
}

/** File → immagine caricata */
export function loadImage(blob) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('Immagine non leggibile'));
    img.src = URL.createObjectURL(blob);
  });
}
