// Classificatore delle zone di pelle sudata: per ogni riquadro di 8×8 px della foto (ritaglio 3:4 sul viso,
// come le foto del Sudometro) dà la probabilità che sia pelle sudata.
// Base: MobileNetV2 0.35 pre-addestrata su ImageNet, ferma (non si addestra). Sopra: una piccola "testa"
// che unisce tre livelli della base (dettagli fini, medi, contesto) e si addestra sulle nostre etichette.
// Va sia nel browser (tf da CDN) sia in Node (@tensorflow/tfjs-node): `tf` si passa sempre come parametro.

export const H = 384; // altezza dell'ingresso (px)
export const W = 288; // larghezza: 3:4 come le foto del Sudometro
export const STRIDE = 8; // lato di un riquadro
export const GH = H / STRIDE; // 48 righe
export const GW = W / STRIDE; // 36 colonne

// Livelli della base usati dalla testa: passo 4 (riflessi piccoli), 8, 16 (dove siamo nel viso)
const TAPS = ['block_3_expand_relu', 'block_6_expand_relu', 'block_13_expand_relu'];

// Valori delle maschere disegnate con lo strumento di etichettatura (canale rosso del PNG)
export const MASK = { sudato: 255, incerto: 128, asciutto: 0 };

/**
 * Carica la base MobileNetV2 e la taglia ai livelli che servono (`taps`: nomi dei livelli in uscita).
 * `read(path)` → ArrayBuffer del file relativo alla cartella del modello (fetch nel browser, fs in Node).
 */
export async function loadBase(tf, read, taps = TAPS) {
  const json = JSON.parse(new TextDecoder().decode(await read('model.json')));
  const topo = json.modelTopology;
  const cfg = (topo.model_config || topo).config;
  cfg.layers[0].config.batch_input_shape = [null, H, W, 3]; // la base nasce a 224×224, i filtri vanno bene a ogni misura
  const specs = json.weightsManifest.flatMap((g) => g.weights);
  const bins = await Promise.all(json.weightsManifest.flatMap((g) => g.paths).map(read));
  const weightData = concat(bins);
  const full = await tf.loadLayersModel(tf.io.fromMemory({ modelTopology: topo, weightSpecs: specs, weightData }));
  const base = tf.model({ inputs: full.inputs, outputs: taps.map((n) => full.getLayer(n).output) });
  base.trainable = false;
  return base;
}

/** La testa: tre ingressi (le uscite della base) → griglia GH×GW×1 di probabilità. Circa 35 mila pesi. */
export function buildHead(tf, { dropout = 0.25, l2 = 1e-4 } = {}) {
  const reg = () => tf.regularizers.l2({ l2 });
  const conv = (filters, kernel, name, activation = 'relu') =>
    tf.layers.conv2d({ filters, kernelSize: kernel, padding: 'same', activation, kernelRegularizer: reg(), name });
  const f4 = tf.input({ shape: [GH * 2, GW * 2, 48], name: 'f4' });
  const f8 = tf.input({ shape: [GH, GW, 96], name: 'f8' });
  const f16 = tf.input({ shape: [GH / 2, GW / 2, 192], name: 'f16' });
  const a = conv(16, 1, 'fini').apply(tf.layers.averagePooling2d({ poolSize: 2, name: 'fini_pool' }).apply(f4));
  const b = conv(32, 1, 'medi').apply(f8);
  const c = tf.layers
    .upSampling2d({ size: [2, 2], interpolation: 'bilinear', name: 'contesto_up' })
    .apply(conv(32, 1, 'contesto').apply(f16));
  let x = tf.layers.concatenate({ name: 'unione' }).apply([a, b, c]);
  x = conv(32, 3, 'misto').apply(x);
  x = tf.layers.dropout({ rate: dropout, name: 'dropout' }).apply(x);
  const out = conv(1, 1, 'sudato', 'sigmoid').apply(x);
  return tf.model({ inputs: [f4, f8, f16], outputs: out, name: 'testa' });
}

/**
 * Immagine (tensore [h, w, c] 0-255) → tensore float [H, W, c] ritagliato al centro in 3:4 e ridimensionato.
 * Le foto del Sudometro sono già 3:4: il ritaglio serve solo per foto d'altra forma.
 */
export function fitImage(tf, img) {
  return tf.tidy(() => {
    const { x, y, w, h } = cropBox(img.shape[1], img.shape[0]);
    return tf.image.resizeBilinear(img.slice([y, x, 0], [h, w, -1]).toFloat(), [H, W]);
  });
}

/** Ritaglio 3:4 al centro di una foto larga `width` e alta `height` (lo stesso di fitImage). */
export function cropBox(width, height) {
  let h = height;
  let w = Math.round((height * W) / H);
  if (w > width) {
    w = width;
    h = Math.round((width * H) / W);
  }
  return { x: Math.floor((width - w) / 2), y: Math.floor((height - h) / 2), w, h };
}

/** [n, H, W, 3] 0-255 → ingresso della base (MobileNetV2 vuole valori fra −1 e 1). */
export function preprocess(tf, batch) {
  return tf.tidy(() => batch.div(127.5).sub(1));
}

/** Probabilità per riquadro: [n, H, W, 3] 0-255 → tensore [n, GH, GW]. */
export function predict(tf, base, head, batch) {
  return tf.tidy(() => {
    const feats = base.predict(preprocess(tf, batch));
    return head.predict(feats).squeeze([3]);
  });
}

/**
 * Maschere ([n, H, W, 1] con i valori di MASK) → bersagli [n, GH, GW, 2]:
 * canale 0 = quota di pixel sudati nel riquadro, canale 1 = peso (i pixel "incerti" non contano).
 */
export function maskToTarget(tf, masks) {
  return tf.tidy(() => {
    const m = masks.toFloat();
    const sweat = m.greater(191).toFloat();
    const unsure = m.greater(63).logicalAnd(m.lessEqual(191)).toFloat();
    const pool = (t) => tf.avgPool(t, STRIDE, STRIDE, 'valid');
    const known = pool(tf.onesLike(m).sub(unsure));
    const frac = pool(sweat).div(known.maximum(1e-6)); // quota fra i pixel sicuri
    return tf.concat([frac, known], 3);
  });
}

/**
 * Perdita: entropia incrociata pesata (i riquadri incerti pesano meno) e con più peso ai riquadri sudati,
 * che sono pochi rispetto alla pelle asciutta e allo sfondo.
 */
export function loss(tf, target, pred, posWeight = 3) {
  return tf.tidy(() => {
    const t = target.slice([0, 0, 0, 0], [-1, -1, -1, 1]);
    const w = target.slice([0, 0, 0, 1], [-1, -1, -1, 1]);
    const p = pred.clipByValue(1e-6, 1 - 1e-6);
    const bce = t.mul(p.log()).mul(posWeight).add(tf.scalar(1).sub(t).mul(tf.scalar(1).sub(p).log())).neg();
    return bce.mul(w).sum().div(w.sum().add(1e-6));
  });
}

/** Persona dal nome del file dello zip del Sudometro (serve per la validazione: mai la stessa persona in entrambi). */
export function personOf(file) {
  const base = file.split('/').pop().replace(/\.\w+$/, '');
  let m = base.match(/^\d{4}-\d{2}-\d{2}_\d+_(.+)_\d+(_debug)?$/); // sudato/2026-10-15_1930_Nome_72.jpg
  if (m) return m[1];
  m = base.match(/^(.+?)_(\d{4}-\d{2}-\d{2}_[\w-]+|attuale)$/); // riferimenti/Nome_2026-10-15_abc123.jpg
  if (m) return m[1];
  return base;
}

function concat(buffers) {
  const total = buffers.reduce((n, b) => n + b.byteLength, 0);
  const out = new Uint8Array(total);
  let o = 0;
  buffers.forEach((b) => {
    out.set(new Uint8Array(b), o);
    o += b.byteLength;
  });
  return out.buffer;
}
