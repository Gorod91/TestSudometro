// Strumento per colorare le zone di pelle sudata sulle foto del Sudometro.
// Foto ed etichette restano nel browser (IndexedDB); "Esporta etichette" crea lo zip per l'addestramento:
// etichette.json + maschere/<foto>.png (grigio: 255 sudato, 128 incerto, 0 asciutto).

import { readZip, makeZip } from '../lib/zip.js';
import { MASK, GH, GW } from '../lib/modello.js';

const $ = (id) => document.getElementById(id);
const COLORS = { sudato: [255, 45, 85], incerto: [255, 204, 0] };

// ---------- Archivio nel browser ----------

const dbPromise = new Promise((resolve, reject) => {
  const req = indexedDB.open('etichette-sudometro', 1);
  req.onupgradeneeded = () => {
    req.result.createObjectStore('foto');
    req.result.createObjectStore('etichette');
  };
  req.onsuccess = () => resolve(req.result);
  req.onerror = () => reject(req.error);
});

async function store(name, mode = 'readonly') {
  return (await dbPromise).transaction(name, mode).objectStore(name);
}
const done = (req) => new Promise((resolve, reject) => {
  req.onsuccess = () => resolve(req.result);
  req.onerror = () => reject(req.error);
});
const put = async (s, key, value) => done((await store(s, 'readwrite')).put(value, key));
const get = async (s, key) => done((await store(s)).get(key));
const keys = async (s) => done((await store(s)).getAllKeys());
const clear = async (s) => done((await store(s, 'readwrite')).clear());

// ---------- Elenco ----------

let photos = []; // nomi, sudato prima dei riferimenti
let labels = new Map(); // nome → { stato, mask: Blob | null }
let onlyTodo = true;

const isRef = (name) => name.startsWith('riferimenti/');
/** Stato effettivo: i riferimenti senza etichetta contano come asciutti. */
const stateOf = (name) => labels.get(name)?.stato || (isRef(name) ? 'asciutto-auto' : null);
const TAGS = { fatto: '✓ zone', asciutto: '💧✕ asciutto', 'asciutto-auto': 'rif. asciutto', scartata: '✕ scartata' };

async function refresh() {
  const names = await keys('foto');
  photos = names.sort((a, b) => isRef(a) - isRef(b) || a.localeCompare(b));
  const all = await keys('etichette');
  labels = new Map(await Promise.all(all.map(async (k) => [k, await get('etichette', k)])));
  const sweat = photos.filter((n) => !isRef(n));
  const todo = sweat.filter((n) => !stateOf(n)).length;
  const zones = photos.filter((n) => stateOf(n) === 'fatto').length;
  $('summary').textContent = photos.length
    ? `${photos.length} foto (${sweat.length} da sudato, ${photos.length - sweat.length} riferimenti). Da fare: ${todo}. Con zone sudate: ${zones}.`
    : 'Nessuna foto. Lo zip si scarica da Palestrina → Admin → "Scarica le foto del Sudometro".';
  renderGrid();
}

const thumbUrls = new Map();
async function thumbUrl(name) {
  if (!thumbUrls.has(name)) thumbUrls.set(name, URL.createObjectURL(await get('foto', name)));
  return thumbUrls.get(name);
}

async function renderGrid() {
  const grid = $('grid');
  grid.textContent = '';
  const list = visible();
  for (const name of list) {
    const div = document.createElement('div');
    div.className = 'thumb';
    const img = document.createElement('img');
    img.loading = 'lazy';
    img.alt = name;
    img.src = await thumbUrl(name);
    div.append(img);
    const st = stateOf(name);
    if (st) {
      const tag = document.createElement('span');
      tag.className = `tag ${st.replace('-auto', '')}`;
      tag.textContent = TAGS[st];
      div.append(tag);
    }
    div.onclick = () => openEditor(name);
    grid.append(div);
  }
}

function visible() {
  return onlyTodo ? photos.filter((n) => !stateOf(n)) : photos;
}

// ---------- Importa / esporta ----------

$('in-photos').onchange = async (e) => {
  const files = [...e.target.files];
  e.target.value = '';
  let n = 0;
  try {
    for (const f of files) {
      if (/\.zip$/i.test(f.name) || f.type.includes('zip')) {
        for (const entry of await readZip(await f.arrayBuffer())) {
          if (!/\.(jpe?g|png)$/i.test(entry.name)) continue;
          await put('foto', entry.name, new Blob([entry.data], { type: /png$/i.test(entry.name) ? 'image/png' : 'image/jpeg' }));
          n++;
        }
      } else if (f.type.startsWith('image/')) {
        await put('foto', `sudato/${f.name}`, f);
        n++;
      }
    }
    toast(`${n} foto caricate`);
  } catch (err) {
    toast(`Errore: ${err.message}`);
  }
  refresh();
};

$('in-labels').onchange = async (e) => {
  const f = e.target.files[0];
  e.target.value = '';
  if (!f) return;
  try {
    const files = new Map((await readZip(await f.arrayBuffer())).map((x) => [x.name, x.data]));
    const json = JSON.parse(new TextDecoder().decode(files.get('etichette.json')));
    let n = 0;
    for (const [name, lab] of Object.entries(json.foto || {})) {
      const data = lab.maschera && files.get(lab.maschera);
      await put('etichette', name, { stato: lab.stato, mask: data ? new Blob([data], { type: 'image/png' }) : null });
      n++;
    }
    toast(`${n} etichette importate`);
  } catch (err) {
    toast(`Errore: ${err.message}`);
  }
  refresh();
};

$('btn-export').onclick = async () => {
  const json = { versione: 1, esportato: new Date().toISOString(), foto: {} };
  const files = [];
  for (const [name, lab] of labels) {
    const entry = { stato: lab.stato };
    if (lab.stato === 'fatto' && lab.mask) {
      entry.maschera = `maschere/${name.replace(/\.\w+$/, '')}.png`;
      files.push({ name: entry.maschera, data: new Uint8Array(await lab.mask.arrayBuffer()) });
    }
    json.foto[name] = entry;
  }
  if (!files.length && !labels.size) return toast('Ancora nessuna etichetta');
  files.push({ name: 'etichette.json', data: new TextEncoder().encode(JSON.stringify(json, null, 2)) });
  const blob = makeZip(files);
  const name = `etichette-${new Date().toLocaleDateString('en-CA')}.zip`;
  const file = new File([blob], name, { type: 'application/zip' });
  try {
    if (navigator.canShare && navigator.canShare({ files: [file] })) return await navigator.share({ files: [file], title: name });
  } catch (err) {
    if (err.name === 'AbortError') return;
  }
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 60000);
};

$('btn-clear').onclick = async () => {
  if (!confirm('Cancellare foto ed etichette da questo dispositivo? (Esportate prima le etichette)')) return;
  await clear('foto');
  await clear('etichette');
  thumbUrls.forEach((u) => URL.revokeObjectURL(u));
  thumbUrls.clear();
  refresh();
};

$('f-todo').onclick = () => setFilter(true);
$('f-all').onclick = () => setFilter(false);
function setFilter(todo) {
  onlyTodo = todo;
  $('f-todo').classList.toggle('on', todo);
  $('f-all').classList.toggle('on', !todo);
  renderGrid();
}

// ---------- Editor ----------

const photoCv = $('photo');
const maskCv = $('mask');
const mctx = maskCv.getContext('2d', { willReadFrequently: true });
let current = null; // nome della foto aperta
let currentImg = null;
let tool = 'sudato';
let undo = [];
let view = { base: 1, z: 1, tx: 0, ty: 0 };
let dirty = false;

async function openEditor(name) {
  current = name;
  dirty = false;
  undo = [];
  const img = await createImageBitmap(await get('foto', name));
  currentImg = img;
  photoCv.width = maskCv.width = img.width;
  photoCv.height = maskCv.height = img.height;
  photoCv.getContext('2d').drawImage(img, 0, 0);
  mctx.clearRect(0, 0, img.width, img.height);
  const lab = labels.get(name);
  if (lab?.mask) await drawGrayMask(lab.mask);
  $('ed-title').textContent = `${name}${stateOf(name) ? ` · ${TAGS[stateOf(name)]}` : ''}`;
  $('editor').hidden = false;
  document.body.style.overflow = 'hidden';
  resetView();
  loadSuggester();
}

function closeEditor() {
  $('editor').hidden = true;
  document.body.style.overflow = '';
  current = null;
  refresh();
}

function resetView() {
  const st = $('stage').getBoundingClientRect();
  view.base = Math.min(st.width / maskCv.width, st.height / maskCv.height);
  view.z = 1;
  view.tx = (st.width - maskCv.width * view.base) / 2;
  view.ty = (st.height - maskCv.height * view.base) / 2;
  const layers = $('layers');
  layers.style.width = `${maskCv.width * view.base}px`;
  layers.style.height = `${maskCv.height * view.base}px`;
  applyView();
}

function applyView() {
  $('layers').style.transform = `translate(${view.tx}px, ${view.ty}px) scale(${view.z})`;
  $('ed-zoom').textContent = `${view.z.toFixed(view.z < 2 ? 1 : 0)}×`;
}

async function drawGrayMask(blob) {
  const bmp = await createImageBitmap(blob);
  const tmp = document.createElement('canvas');
  tmp.width = maskCv.width;
  tmp.height = maskCv.height;
  const tctx = tmp.getContext('2d', { willReadFrequently: true });
  tctx.drawImage(bmp, 0, 0, tmp.width, tmp.height);
  const src = tctx.getImageData(0, 0, tmp.width, tmp.height).data;
  const out = mctx.createImageData(tmp.width, tmp.height);
  for (let i = 0; i < src.length; i += 4) {
    const v = src[i];
    const c = v > 191 ? COLORS.sudato : v > 63 ? COLORS.incerto : null;
    if (c) out.data.set([...c, 255], i);
  }
  mctx.putImageData(out, 0, 0);
}

/** Maschera colorata → PNG grigio (null se non c'è niente di colorato) */
async function grayMask() {
  const src = mctx.getImageData(0, 0, maskCv.width, maskCv.height).data;
  const tmp = document.createElement('canvas');
  tmp.width = maskCv.width;
  tmp.height = maskCv.height;
  const tctx = tmp.getContext('2d');
  const out = tctx.createImageData(tmp.width, tmp.height);
  let any = false;
  for (let i = 0; i < src.length; i += 4) {
    let v = MASK.asciutto;
    if (src[i + 3] > 127) {
      v = src[i + 1] > 120 ? MASK.incerto : MASK.sudato;
      any = true;
    }
    out.data.set([v, v, v, 255], i);
  }
  if (!any) return null;
  tctx.putImageData(out, 0, 0);
  return new Promise((resolve) => tmp.toBlob(resolve, 'image/png'));
}

async function save(stato) {
  let mask = null;
  if (stato === 'fatto') {
    mask = await grayMask();
    if (!mask) stato = 'asciutto'; // niente colorato = tutto asciutto
  }
  await put('etichette', current, { stato, mask });
  labels.set(current, { stato, mask });
  dirty = false;
  next(1, true);
}

function next(step, afterSave = false) {
  const list = afterSave && onlyTodo ? photos.filter((n) => !stateOf(n)) : photos;
  if (!list.length) return closeEditor();
  let i = list.indexOf(current);
  if (i < 0) i = afterSave ? -1 : 0; // salvata: è uscita dall'elenco "da fare"
  const n = list[(i + step + list.length) % list.length];
  if (afterSave && n === current) return closeEditor();
  openEditor(n);
}

function leave(fn) {
  if (dirty && !confirm('Ci sono modifiche non salvate. Uscire lo stesso?')) return;
  fn();
}

$('ed-close').onclick = () => leave(closeEditor);
$('ed-prev').onclick = () => leave(() => next(-1));
$('ed-next').onclick = () => leave(() => next(1));
$('ed-save').onclick = () => save('fatto');
$('ed-dry').onclick = () => save('asciutto');
$('ed-skip').onclick = () => save('scartata');
$('ed-undo').onclick = () => {
  const last = undo.pop();
  if (last) mctx.putImageData(last, 0, 0);
};
$('ed-eye').onclick = () => maskCv.classList.toggle('hide');
$('ed-zoom').onclick = resetView;
['sudato', 'incerto', 'gomma'].forEach((t) => {
  $(`t-${t}`).onclick = () => {
    tool = t;
    ['sudato', 'incerto', 'gomma'].forEach((x) => $(`t-${x}`).classList.toggle('on', x === t));
  };
});
window.addEventListener('resize', () => current && resetView());

// ---------- Disegno con le dita: un dito colora, due dita ingrandiscono e spostano ----------

const pointers = new Map();
let stroke = null; // { last: [x, y] }
let pinch = null;

function toCanvas(e) {
  const r = maskCv.getBoundingClientRect();
  return [((e.clientX - r.left) / r.width) * maskCv.width, ((e.clientY - r.top) / r.height) * maskCv.height];
}

function paint(from, to) {
  const radius = Number($('brush').value) / 2 / (view.base * view.z);
  mctx.save();
  mctx.globalCompositeOperation = tool === 'gomma' ? 'destination-out' : 'source-over';
  mctx.strokeStyle = mctx.fillStyle = tool === 'gomma' ? '#000' : `rgb(${COLORS[tool].join(',')})`;
  mctx.lineCap = mctx.lineJoin = 'round';
  mctx.lineWidth = radius * 2;
  mctx.beginPath();
  mctx.moveTo(...from);
  mctx.lineTo(...to);
  mctx.stroke();
  mctx.restore();
  dirty = true;
}

const stage = $('stage');
stage.addEventListener('pointerdown', (e) => {
  stage.setPointerCapture(e.pointerId);
  pointers.set(e.pointerId, e);
  if (pointers.size === 1 && e.button === 0) {
    undo.push(mctx.getImageData(0, 0, maskCv.width, maskCv.height));
    if (undo.length > 25) undo.shift();
    const p = toCanvas(e);
    stroke = { last: p };
    paint(p, p);
  } else if (pointers.size === 2) {
    if (stroke) {
      mctx.putImageData(undo.pop(), 0, 0); // il primo dito non voleva disegnare
      stroke = null;
    }
    pinch = startPinch();
  }
});
stage.addEventListener('pointermove', (e) => {
  if (!pointers.has(e.pointerId)) return;
  pointers.set(e.pointerId, e);
  if (stroke) {
    const p = toCanvas(e);
    paint(stroke.last, p);
    stroke.last = p;
  } else if (pinch && pointers.size === 2) {
    const { d, m } = pinchState();
    const r = stage.getBoundingClientRect();
    view.z = Math.min(8, Math.max(1, (pinch.z * d) / pinch.d));
    const cx = (pinch.m[0] - r.left - pinch.tx) / pinch.z;
    const cy = (pinch.m[1] - r.top - pinch.ty) / pinch.z;
    view.tx = m[0] - r.left - cx * view.z;
    view.ty = m[1] - r.top - cy * view.z;
    applyView();
  }
});
const up = (e) => {
  pointers.delete(e.pointerId);
  if (pointers.size < 2) pinch = null;
  if (!pointers.size) stroke = null;
};
stage.addEventListener('pointerup', up);
stage.addEventListener('pointercancel', up);
stage.addEventListener('wheel', (e) => {
  e.preventDefault();
  const r = stage.getBoundingClientRect();
  const z = Math.min(8, Math.max(1, view.z * (e.deltaY < 0 ? 1.15 : 1 / 1.15)));
  const cx = (e.clientX - r.left - view.tx) / view.z;
  const cy = (e.clientY - r.top - view.ty) / view.z;
  view.tx = e.clientX - r.left - cx * z;
  view.ty = e.clientY - r.top - cy * z;
  view.z = z;
  applyView();
}, { passive: false });

function pinchState() {
  const [a, b] = [...pointers.values()];
  return { d: Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY) || 1, m: [(a.clientX + b.clientX) / 2, (a.clientY + b.clientY) / 2] };
}
function startPinch() {
  return { ...pinchState(), z: view.z, tx: view.tx, ty: view.ty };
}

// ---------- Proposta del modello (se ce n'è già uno addestrato) ----------

let suggester = null;
let suggesterTried = false;
async function loadSuggester() {
  if (suggesterTried) return;
  suggesterTried = true;
  try {
    const { loadClassifier } = await import('../lib/browser.js');
    suggester = await loadClassifier();
    $('ed-suggest').hidden = !suggester;
  } catch {
    suggester = null;
  }
}
$('ed-suggest').onclick = () => {
  if (!suggester || !currentImg) return;
  const { prob, box } = suggester.run(photoCv);
  undo.push(mctx.getImageData(0, 0, maskCv.width, maskCv.height));
  mctx.fillStyle = `rgb(${COLORS.sudato.join(',')})`;
  const cw = box.w / GW;
  const ch = box.h / GH;
  let n = 0;
  for (let i = 0; i < prob.length; i++) {
    if (prob[i] < suggester.info.soglia) continue;
    mctx.fillRect(box.x + (i % GW) * cw, box.y + Math.floor(i / GW) * ch, Math.ceil(cw), Math.ceil(ch));
    n++;
  }
  dirty = true;
  toast(n ? `${n} riquadri proposti: correggeteli` : 'Il modello non vede zone sudate');
};

// ---------- Varie ----------

function toast(text) {
  const t = document.createElement('div');
  t.className = 'toast';
  t.textContent = text;
  document.body.append(t);
  setTimeout(() => t.remove(), 2500);
}

refresh();
