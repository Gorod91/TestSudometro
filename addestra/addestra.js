// Addestra la testa del classificatore sulle foto del Sudometro etichettate e la salva in modello/.
//   node addestra/addestra.js --foto sudometro-2026-10-20.zip --etichette etichette.zip [--epoche 40] [--validazione no]
// Prima misura quanto funziona su persone che non ha mai visto (validazione a gruppi per persona),
// poi si addestra su tutto e salva modello/testa/ + modello/info.json (soglia, misure, quante foto).

import fs from 'node:fs';
import path from 'node:path';
import tf from '@tensorflow/tfjs-node';
import { readZip } from '../lib/zip.js';
import { H, W, GH, GW, loadBase, buildHead, fitImage, preprocess, maskToTarget, loss, personOf } from '../lib/modello.js';

const args = Object.fromEntries(
  process.argv.slice(2).reduce((acc, a, i, all) => (a.startsWith('--') ? [...acc, [a.slice(2), all[i + 1]]] : acc), []),
);
const EPOCHS = Number(args.epoche || 40);
const BATCH = 8;
const OUT = args.out || 'modello';
const REF_DRY = args['riferimenti-asciutti'] !== 'no'; // riferimenti senza etichetta = tutto asciutto
const VALIDATE = args.validazione !== 'no'; // --validazione no: salta la misura su persone mai viste (più veloce)
if (!args.foto || !args.etichette) {
  console.log('Uso: node addestra/addestra.js --foto <zip del Sudometro> --etichette <zip delle etichette> [--epoche 40]');
  process.exit(1);
}

// ---------- Dati ----------

async function filesOf(p) {
  if (fs.statSync(p).isDirectory()) {
    const walk = (d) => fs.readdirSync(d, { withFileTypes: true }).flatMap((e) => (e.isDirectory() ? walk(path.join(d, e.name)) : [path.join(d, e.name)]));
    return new Map(walk(p).map((f) => [path.relative(p, f).split(path.sep).join('/'), new Uint8Array(fs.readFileSync(f))]));
  }
  return new Map((await readZip(fs.readFileSync(p))).map((f) => [f.name, f.data]));
}

const photos = await filesOf(args.foto);
const labelFiles = await filesOf(args.etichette);
const labels = JSON.parse(new TextDecoder().decode(labelFiles.get('etichette.json') || new Uint8Array([123, 125]))).foto || {};

const samples = []; // { file, person, sweaty, data: tensore [H, W, 4] (RGB + maschera) }
const skipped = { scartate: 0, senzaEtichetta: 0 };
for (const [file, bytes] of photos) {
  if (!/\.jpe?g$/i.test(file)) continue;
  const lab = labels[file] || (REF_DRY && file.startsWith('riferimenti/') ? { stato: 'asciutto' } : null);
  if (!lab) {
    skipped.senzaEtichetta++;
    continue;
  }
  if (lab.stato === 'scartata') {
    skipped.scartate++;
    continue;
  }
  const data = tf.tidy(() => {
    const img = tf.node.decodeImage(bytes, 3);
    let mask = tf.zerosLike(img.slice([0, 0, 0], [-1, -1, 1]));
    if (lab.stato === 'fatto') {
      const png = labelFiles.get(lab.maschera);
      if (!png) throw new Error(`Manca la maschera ${lab.maschera}`);
      mask = tf.image.resizeNearestNeighbor(tf.node.decodeImage(png, 1), img.shape.slice(0, 2)).toInt();
    }
    return fitImage(tf, tf.concat([img, mask], 2));
  });
  samples.push({ file, person: personOf(file), sweaty: lab.stato === 'fatto', data });
}
const people = [...new Set(samples.map((s) => s.person))];
console.log(`${samples.length} foto (${samples.filter((s) => s.sweaty).length} con zone sudate) di ${people.length} persone;`
  + ` saltate: ${skipped.scartate} scartate, ${skipped.senzaEtichetta} senza etichetta`);
if (samples.length < 4) throw new Error('Troppo poche foto etichettate');

// Peso dei riquadri sudati: sono pochi, senza un peso in più il modello direbbe sempre "asciutto"
const posWeight = tf.tidy(() => {
  const t = maskToTarget(tf, tf.stack(samples.map((s) => s.data.slice([0, 0, 3], [-1, -1, 1]))));
  const pos = t.slice([0, 0, 0, 0], [-1, -1, -1, 1]).mul(t.slice([0, 0, 0, 1], [-1, -1, -1, 1])).sum().dataSync()[0];
  const all = t.slice([0, 0, 0, 1], [-1, -1, -1, 1]).sum().dataSync()[0];
  return Math.min(8, Math.max(1, Math.sqrt((all - pos) / Math.max(pos, 1))));
});

const read = (p) => fs.promises.readFile(path.join('vendor/mobilenet_v2_0.35', p)).then((b) => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
const base = await loadBase(tf, read);

// ---------- Variazioni casuali (ritaglio, specchio, luce) ----------

function augment(batch) {
  return tf.tidy(() => {
    const n = batch.shape[0];
    const boxes = [];
    for (let i = 0; i < n; i++) {
      const s = 0.8 + Math.random() * 0.2; // zoom fino al 25%
      const y1 = Math.random() * (1 - s);
      const x1 = Math.random() * (1 - s);
      const flip = Math.random() < 0.5;
      boxes.push(flip ? [y1, x1 + s, y1 + s, x1] : [y1, x1, y1 + s, x1 + s]); // x1 > x2 = specchiato
    }
    const out = tf.image.cropAndResize(batch, tf.tensor2d(boxes), tf.range(0, n, 1, 'int32'), [H, W], 'bilinear');
    const rgb = out.slice([0, 0, 0, 0], [-1, -1, -1, 3]);
    const mask = out.slice([0, 0, 0, 3], [-1, -1, -1, 1]).round();
    const light = tf.randomUniform([n, 1, 1, 1], 0.8, 1.2); // luminosità
    const tint = tf.randomUniform([n, 1, 1, 3], 0.94, 1.06); // dominante della luce
    return [rgb.mul(light).mul(tint).clipByValue(0, 255), mask];
  });
}

function features(rgb) {
  return tf.tidy(() => base.predict(preprocess(tf, rgb)));
}

// ---------- Addestramento ----------

function train(set, epochs = EPOCHS, log = false) {
  const head = buildHead(tf);
  const opt = tf.train.adam(1e-3);
  for (let e = 0; e < epochs; e++) {
    const order = tf.util.createShuffledIndices(set.length);
    let sum = 0;
    let batches = 0;
    for (let i = 0; i < order.length; i += BATCH) {
      const idx = Array.from(order.slice(i, i + BATCH));
      const [rgb, mask] = augment(tf.stack(idx.map((k) => set[k].data)));
      const feats = features(rgb);
      const target = maskToTarget(tf, mask);
      const value = opt.minimize(() => {
        const pred = head.apply(feats, { training: true });
        return loss(tf, target, pred, posWeight).add(tf.addN(head.losses.length ? head.losses.map((l) => l()) : [tf.scalar(0)]));
      }, true);
      sum += value.dataSync()[0];
      batches++;
      tf.dispose([rgb, mask, feats, target, value]);
    }
    if (log && (e % 5 === 4 || e === epochs - 1)) console.log(`  epoca ${e + 1}/${epochs}: perdita ${(sum / batches).toFixed(3)}`);
  }
  opt.dispose();
  return head;
}

function evaluate(head, set) {
  const cells = []; // [prob, positivo]
  const perPhoto = [];
  for (const s of set) {
    const [prob, target] = tf.tidy(() => {
      const x = s.data.expandDims(0);
      const p = head.predict(features(x.slice([0, 0, 0, 0], [-1, -1, -1, 3]))).dataSync();
      const t = maskToTarget(tf, x.slice([0, 0, 0, 3], [-1, -1, -1, 1])).dataSync();
      return [p, t];
    });
    for (let i = 0; i < GH * GW; i++) {
      if (target[i * 2 + 1] < 0.5) continue; // riquadro incerto
      const frac = target[i * 2];
      if (frac > 0.2 && frac < 0.5) continue; // bordo della zona: non è chiaro né sudato né asciutto
      cells.push([prob[i], frac >= 0.5]);
    }
    perPhoto.push({ file: s.file, sweaty: s.sweaty, prob });
  }
  return { cells, perPhoto };
}

function metrics(cells, perPhoto, threshold) {
  const pos = cells.filter((c) => c[1]).length;
  const neg = cells.length - pos;
  // AUC: probabilità che un riquadro sudato abbia un punteggio più alto di uno asciutto
  const sorted = [...cells].sort((a, b) => a[0] - b[0]);
  let rankSum = 0;
  sorted.forEach((c, i) => c[1] && (rankSum += i + 1));
  const auc = pos && neg ? (rankSum - (pos * (pos + 1)) / 2) / (pos * neg) : NaN;
  const at = (t) => {
    let tp = 0;
    let fp = 0;
    cells.forEach(([p, y]) => p >= t && (y ? tp++ : fp++));
    const fn = pos - tp;
    return { t, f1: (2 * tp) / Math.max(1, 2 * tp + fp + fn), iou: tp / Math.max(1, tp + fp + fn), precisione: tp / Math.max(1, tp + fp), richiamo: tp / Math.max(1, pos) };
  };
  if (threshold == null) {
    threshold = 0.5;
    let best = -1;
    for (let t = 0.05; t < 0.96; t += 0.05) {
      const m = at(t);
      if (m.f1 > best) [best, threshold] = [m.f1, Number(t.toFixed(2))];
    }
  }
  const m = at(threshold);
  // Per foto: quota di riquadri sopra soglia, foto asciutte contro foto con zone sudate
  const share = (p) => p.filter((v) => v >= threshold).length / p.length;
  const mean = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : NaN);
  return {
    soglia: threshold,
    auc: round(auc),
    f1: round(m.f1),
    iou: round(m.iou),
    precisione: round(m.precisione),
    richiamo: round(m.richiamo),
    riquadri: { sudati: pos, asciutti: neg },
    quotaSopraSoglia: {
      fotoAsciutte: round(mean(perPhoto.filter((p) => !p.sweaty).map((p) => share(p.prob)))),
      fotoSudate: round(mean(perPhoto.filter((p) => p.sweaty).map((p) => share(p.prob)))),
    },
  };
}

const round = (v) => (Number.isFinite(v) ? Math.round(v * 1000) / 1000 : null);

// Validazione: ogni gruppo di persone resta fuori a turno, il modello non le vede mai in addestramento
let validation = null;
if (VALIDATE && people.length >= 2) {
  const k = Math.min(5, people.length);
  const groups = people.map((p, i) => [p, i % k]);
  const cells = [];
  const perPhoto = [];
  for (let f = 0; f < k; f++) {
    const out = new Set(groups.filter(([, g]) => g === f).map(([p]) => p));
    console.log(`Validazione ${f + 1}/${k}: fuori ${[...out].join(', ')}`);
    const head = train(samples.filter((s) => !out.has(s.person)));
    const r = evaluate(head, samples.filter((s) => out.has(s.person)));
    cells.push(...r.cells);
    perPhoto.push(...r.perPhoto);
    head.dispose();
  }
  validation = metrics(cells, perPhoto);
  console.log('Su persone mai viste:', JSON.stringify(validation, null, 2));
} else if (VALIDATE) {
  console.log('Una sola persona: niente validazione (servono foto di almeno 2 persone)');
}

console.log('Addestramento finale su tutte le foto');
const head = train(samples, EPOCHS, true);
const final = evaluate(head, samples);
const onTrain = metrics(final.cells, final.perPhoto, validation ? validation.soglia : null);
fs.mkdirSync(OUT, { recursive: true });
await head.save(`file://${path.resolve(OUT, 'testa')}`);
const info = {
  creato: new Date().toISOString(),
  ingresso: { H, W, griglia: [GH, GW] },
  soglia: validation ? validation.soglia : onTrain.soglia,
  foto: { totali: samples.length, conZoneSudate: samples.filter((s) => s.sweaty).length, persone: people.length },
  epoche: EPOCHS,
  pesoSudato: round(posWeight),
  validazione: validation, // la misura che conta: persone mai viste
  suFotoDiAddestramento: onTrain, // sempre ottimista
};
fs.writeFileSync(path.join(OUT, 'info.json'), JSON.stringify(info, null, 2) + '\n');
console.log(`Salvato in ${OUT}/ (soglia ${info.soglia})`);
