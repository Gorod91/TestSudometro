// Addestra il punteggio 0-100 sui voti dati a mano e lo salva in modello-punteggio/.
//   node addestra/addestra-punteggio.js --foto sudometro-2026-10-20.zip --punteggi punteggi-2026-10-20.json
// Prima misura l'errore su persone mai viste (a turno una persona resta fuori), confrontandolo con chi
// dà sempre il voto medio; poi si addestra su tutto. I riferimenti senza voto valgono 0 (--riferimenti-zero no per saltarli).

import fs from 'node:fs';
import path from 'node:path';
import tf from '@tensorflow/tfjs-node';
import { readZip } from '../lib/zip.js';
import { H, W, loadBase, fitImage, personOf } from '../lib/modello.js';
import { SCORE_TAPS, embed, fitRidge, predictScore } from '../lib/punteggio.js';

const args = Object.fromEntries(
  process.argv.slice(2).reduce((acc, a, i, all) => (a.startsWith('--') ? [...acc, [a.slice(2), all[i + 1]]] : acc), []),
);
const OUT = args.out || 'modello-punteggio';
const REF_ZERO = args['riferimenti-zero'] !== 'no';
const AUG = Number(args.variazioni || 4); // copie con ritaglio e luce diversi per ogni foto, solo in addestramento
if (!args.foto || !args.punteggi) {
  console.log('Uso: node addestra/addestra-punteggio.js --foto <zip del Sudometro> --punteggi <punteggi.json>');
  process.exit(1);
}

const photos = new Map((await readZip(fs.readFileSync(args.foto))).map((f) => [f.name, f.data]));
const scores = JSON.parse(fs.readFileSync(args.punteggi, 'utf8')).punteggi || {};

const read = (p) => fs.promises.readFile(path.join('vendor/mobilenet_v2_0.35', p)).then((b) => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
const base = await loadBase(tf, read, SCORE_TAPS);

function variants(img) {
  return tf.tidy(() => {
    const boxes = [];
    for (let i = 0; i < AUG; i++) {
      const s = 0.85 + Math.random() * 0.15;
      const y1 = Math.random() * (1 - s);
      const x1 = Math.random() * (1 - s);
      boxes.push([y1, x1, y1 + s, x1 + s]);
    }
    const out = tf.image.cropAndResize(img.expandDims(0), tf.tensor2d(boxes), tf.zeros([AUG], 'int32'), [H, W]);
    return out.mul(tf.randomUniform([AUG, 1, 1, 1], 0.85, 1.15)).mul(tf.randomUniform([AUG, 1, 1, 3], 0.95, 1.05)).clipByValue(0, 255);
  });
}

const samples = []; // { file, person, score, row, extra: [righe delle variazioni] }
for (const [file, bytes] of photos) {
  if (!/\.jpe?g$/i.test(file)) continue;
  let score = scores[file];
  if (score == null && REF_ZERO && file.startsWith('riferimenti/')) score = 0;
  if (typeof score !== 'number') continue; // senza voto o "non valutabile"
  const img = tf.tidy(() => fitImage(tf, tf.node.decodeImage(bytes, 3)));
  const e = embed(tf, base, img.expandDims(0));
  const v = variants(img);
  const ev = embed(tf, base, v);
  const row = e.dataSync().slice();
  const flat = ev.dataSync();
  const d = row.length;
  const extra = Array.from({ length: AUG }, (_, i) => flat.slice(i * d, (i + 1) * d));
  tf.dispose([img, e, v, ev]);
  samples.push({ file, person: personOf(file), score, row, extra });
}
const people = [...new Set(samples.map((s) => s.person))];
console.log(`${samples.length} foto con voto di ${people.length} persone, ${samples[0]?.row.length} caratteristiche per foto`);
if (samples.length < 5) throw new Error('Troppo poche foto con voto');

const fit = (set, lambda) => fitRidge(set.flatMap((s) => [s.row, ...s.extra]), set.flatMap((s) => Array(AUG + 1).fill(s.score)), lambda);

// Validazione: ogni persona resta fuori a turno; si sceglie anche quanto "frenare" i pesi (lambda)
const D = samples[0].row.length;
const LAMBDAS = [0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10].map((f) => f * D);
let best = null;
if (people.length >= 2) {
  for (const lambda of LAMBDAS) {
    const preds = [];
    for (const p of people) {
      const train = samples.filter((s) => s.person !== p);
      const model = fit(train, lambda);
      const mean = train.reduce((a, s) => a + s.score, 0) / train.length;
      samples.filter((s) => s.person === p).forEach((s) => preds.push({ s, pred: predictScore(model, s.row), mean }));
    }
    const mae = avg(preds.map((x) => Math.abs(x.pred - x.s.score)));
    console.log(`  lambda ${lambda.toFixed(0).padStart(6)}: errore medio ${mae.toFixed(1)} punti`);
    if (!best || mae < best.mae) best = { lambda, mae, preds };
  }
  const { preds } = best;
  const y = preds.map((x) => x.s.score);
  const p = preds.map((x) => x.pred);
  best.report = {
    erroreMedio: round(best.mae), // punti, su persone mai viste
    erroreVotoMedio: round(avg(preds.map((x) => Math.abs(x.mean - x.s.score)))), // chi dà sempre la media
    correlazione: round(pearson(y, p)),
    correlazioneOrdine: round(pearson(ranks(y), ranks(p))), // Spearman: quanto ordina bene le foto
    foto: preds.map((x) => ({ file: x.s.file, voto: x.s.score, stima: Math.round(x.pred) })).sort((a, b) => b.voto - a.voto),
  };
  console.log('Su persone mai viste:');
  best.report.foto.forEach((f) => console.log(`  ${String(f.voto).padStart(3)} → ${String(f.stima).padStart(3)}  ${f.file}`));
  const { foto, ...summary } = best.report;
  console.log(JSON.stringify(summary, null, 2));
} else {
  console.log('Una sola persona: niente validazione, lambda di default');
}

const lambda = best ? best.lambda : 1 * D;
const model = fit(samples, lambda);
fs.mkdirSync(OUT, { recursive: true });
const out = {
  tipo: 'ridge',
  creato: new Date().toISOString(),
  ingresso: { H, W },
  livelli: SCORE_TAPS,
  lambda,
  foto: { totali: samples.length, persone: people.length },
  validazione: best ? best.report : null,
  b: model.b,
  w: model.w.map((v) => Number(v.toPrecision(5))),
};
fs.writeFileSync(path.join(OUT, 'modello.json'), JSON.stringify(out));
console.log(`Salvato in ${OUT}/modello.json`);

function avg(a) {
  return a.reduce((x, y) => x + y, 0) / a.length;
}
function pearson(a, b) {
  const ma = avg(a);
  const mb = avg(b);
  let s = 0;
  let sa = 0;
  let sb = 0;
  a.forEach((v, i) => {
    s += (v - ma) * (b[i] - mb);
    sa += (v - ma) ** 2;
    sb += (b[i] - mb) ** 2;
  });
  return s / Math.sqrt(sa * sb || 1);
}
function ranks(a) {
  const idx = a.map((v, i) => [v, i]).sort((x, y) => x[0] - y[0]);
  const r = new Array(a.length);
  for (let i = 0; i < idx.length; ) {
    let j = i;
    while (j + 1 < idx.length && idx[j + 1][0] === idx[i][0]) j++;
    for (let k = i; k <= j; k++) r[idx[k][1]] = (i + j) / 2;
    i = j + 1;
  }
  return r;
}
function round(v) {
  return Number.isFinite(v) ? Math.round(v * 100) / 100 : null;
}
