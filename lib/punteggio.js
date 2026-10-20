// Punteggio 0-100 di una foto da sudato, imparato dai voti dati a mano (pagina punteggio/).
// Dalla stessa base MobileNetV2 si prende, per quattro livelli, la media e il massimo su tutta la foto
// (il massimo coglie i riflessi piccoli del sudore); sopra c'è una regressione ridge: una somma pesata,
// adatta a poche decine di foto perché i pesi restano piccoli. Va nel browser e in Node (`tf` come parametro).

import { preprocess } from './modello.js';

export const SCORE_TAPS = ['block_3_expand_relu', 'block_6_expand_relu', 'block_13_expand_relu', 'out_relu'];

/** [n, H, W, 3] 0-255 → [n, D]: media e massimo di ogni livello, anche sulla foto specchiata (media delle due). */
export function embed(tf, base, batch) {
  return tf.tidy(() => {
    const one = (x) => {
      const outs = base.predict(preprocess(tf, x));
      return tf.concat(outs.flatMap((o) => [o.mean([1, 2]), o.max([1, 2])]), 1);
    };
    return one(batch).add(one(batch.reverse(2))).div(2);
  });
}

/**
 * Regressione ridge. rows: array di Float32Array (stessa lunghezza), y: voti.
 * Le colonne vengono standardizzate; si risolve nella forma "duale" (n × n), comoda con poche foto e tante colonne.
 * Restituisce { w, b }: punteggio = b + somma(w[j] · x[j]), già con la standardizzazione dentro.
 */
export function fitRidge(rows, y, lambda) {
  const n = rows.length;
  const d = rows[0].length;
  const mean = new Float64Array(d);
  const std = new Float64Array(d);
  rows.forEach((r) => r.forEach((v, j) => (mean[j] += v / n)));
  rows.forEach((r) => r.forEach((v, j) => (std[j] += (v - mean[j]) ** 2 / n)));
  for (let j = 0; j < d; j++) std[j] = Math.sqrt(std[j]) + 1e-6;
  const z = rows.map((r) => Float64Array.from(r, (v, j) => (v - mean[j]) / std[j]));
  const yMean = y.reduce((a, b) => a + b, 0) / n;
  // (Z·Zᵀ + λI) α = y − media
  const k = Array.from({ length: n }, (_, i) => {
    const row = new Float64Array(n);
    for (let l = 0; l <= i; l++) {
      let s = 0;
      for (let j = 0; j < d; j++) s += z[i][j] * z[l][j];
      row[l] = s;
    }
    return row;
  });
  for (let i = 0; i < n; i++) {
    for (let l = i + 1; l < n; l++) k[i][l] = k[l][i];
    k[i][i] += lambda;
  }
  const alpha = cholSolve(k, y.map((v) => v - yMean));
  const w = new Float64Array(d);
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) w[j] += alpha[i] * z[i][j];
  let b = yMean;
  for (let j = 0; j < d; j++) {
    w[j] /= std[j];
    b -= w[j] * mean[j];
  }
  return { w: Array.from(w), b };
}

/** Punteggio 0-100 di una riga di caratteristiche */
export function predictScore({ w, b }, row) {
  let s = b;
  for (let j = 0; j < w.length; j++) s += w[j] * row[j];
  return Math.max(0, Math.min(100, s));
}

function cholSolve(a, y) {
  const n = a.length;
  const l = a.map(() => new Float64Array(n));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = a[i][j];
      for (let k = 0; k < j; k++) s -= l[i][k] * l[j][k];
      l[i][j] = i === j ? Math.sqrt(Math.max(s, 1e-12)) : s / l[j][j];
    }
  }
  const t = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = y[i];
    for (let k = 0; k < i; k++) s -= l[i][k] * t[k];
    t[i] = s / l[i][i];
  }
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = t[i];
    for (let k = i + 1; k < n; k++) s -= l[k][i] * x[k];
    x[i] = s / l[i][i];
  }
  return x;
}
