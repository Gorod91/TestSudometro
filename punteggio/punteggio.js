// Voti 0-100 alle foto del Sudometro, per addestrare il punteggio.
// Foto e voti restano nel browser (IndexedDB); "Esporta voti" crea punteggi.json:
// { versione: 1, punteggi: { "<file nello zip>": 0-100 | null (non valutabile) } }

import { readZip } from '../lib/zip.js';

const $ = (id) => document.getElementById(id);

// ---------- Archivio nel browser ----------

const dbPromise = new Promise((resolve, reject) => {
  const req = indexedDB.open('punteggi-sudometro', 1);
  req.onupgradeneeded = () => {
    req.result.createObjectStore('foto');
    req.result.createObjectStore('punteggi');
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

let photos = [];
let scores = new Map(); // nome → 0-100 oppure null (non valutabile)
let onlyTodo = true;

const isRef = (name) => name.startsWith('riferimenti/');
/** Voto effettivo: undefined = da fare; i riferimenti senza voto valgono 0 */
const scoreOf = (name) => (scores.has(name) ? scores.get(name) : isRef(name) ? 0 : undefined);

async function refresh() {
  photos = (await keys('foto')).sort((a, b) => isRef(a) - isRef(b) || a.localeCompare(b));
  const all = await keys('punteggi');
  scores = new Map(await Promise.all(all.map(async (k) => [k, await get('punteggi', k)])));
  const todo = photos.filter((n) => scoreOf(n) === undefined).length;
  $('summary').textContent = photos.length
    ? `${photos.length} foto. Da votare: ${todo}. Con voto: ${photos.filter((n) => typeof scoreOf(n) === 'number').length}.`
    : 'Nessuna foto. Lo zip si scarica da Palestrina → Admin → "Scarica le foto del Sudometro".';
  renderGrid();
}

const urls = new Map();
async function urlOf(name) {
  if (!urls.has(name)) urls.set(name, URL.createObjectURL(await get('foto', name)));
  return urls.get(name);
}

const visible = () => (onlyTodo ? photos.filter((n) => scoreOf(n) === undefined) : photos);

async function renderGrid() {
  const grid = $('grid');
  grid.textContent = '';
  for (const name of visible()) {
    const div = document.createElement('div');
    div.className = 'thumb';
    const img = document.createElement('img');
    img.loading = 'lazy';
    img.alt = name;
    img.src = await urlOf(name);
    div.append(img);
    const s = scoreOf(name);
    if (s !== undefined) {
      const tag = document.createElement('span');
      tag.className = `tag ${s === null ? 'scartata' : 'fatto'}`;
      tag.textContent = s === null ? '✕' : `${s}${scores.has(name) ? '' : ' rif.'}`;
      div.append(tag);
    }
    div.onclick = () => openEditor(name);
    grid.append(div);
  }
}

// ---------- Importa / esporta ----------

$('in-photos').onchange = async (e) => {
  const f = e.target.files[0];
  e.target.value = '';
  if (!f) return;
  try {
    let n = 0;
    for (const entry of await readZip(await f.arrayBuffer())) {
      if (!/\.jpe?g$/i.test(entry.name)) continue;
      await put('foto', entry.name, new Blob([entry.data], { type: 'image/jpeg' }));
      n++;
    }
    toast(`${n} foto caricate`);
  } catch (err) {
    toast(`Errore: ${err.message}`);
  }
  refresh();
};

$('in-scores').onchange = async (e) => {
  const f = e.target.files[0];
  e.target.value = '';
  if (!f) return;
  try {
    const json = JSON.parse(await f.text());
    const entries = Object.entries(json.punteggi || {});
    for (const [name, v] of entries) await put('punteggi', name, v);
    toast(`${entries.length} voti importati`);
  } catch (err) {
    toast(`Errore: ${err.message}`);
  }
  refresh();
};

$('btn-export').onclick = async () => {
  if (!scores.size) return toast('Ancora nessun voto');
  const json = { versione: 1, esportato: new Date().toISOString(), punteggi: Object.fromEntries(scores) };
  const name = `punteggi-${new Date().toLocaleDateString('en-CA')}.json`;
  const file = new File([JSON.stringify(json, null, 2)], name, { type: 'application/json' });
  try {
    if (navigator.canShare && navigator.canShare({ files: [file] })) return await navigator.share({ files: [file], title: name });
  } catch (err) {
    if (err.name === 'AbortError') return;
  }
  const a = document.createElement('a');
  a.href = URL.createObjectURL(file);
  a.download = name;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 60000);
};

$('btn-clear').onclick = async () => {
  if (!confirm('Cancellare foto e voti da questo dispositivo? (Esportate prima i voti)')) return;
  await clear('foto');
  await clear('punteggi');
  urls.forEach((u) => URL.revokeObjectURL(u));
  urls.clear();
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

// ---------- Voto ----------

let current = null;
let value = null; // voto in corso, null = non ancora toccato

const SCALE = [
  [90, 'fradicio'],
  [65, 'molto sudato, gocce'],
  [40, 'sudato, si vede bene'],
  [15, 'un po\' lucido'],
  [0, 'asciutto e fresco'],
];

function setValue(v) {
  value = v == null ? null : Math.max(0, Math.min(100, Math.round(v)));
  $('score').textContent = value == null ? '–' : value;
  $('score').classList.toggle('unset', value == null);
  $('slider').value = value ?? 0;
  $('scale').textContent = value == null ? 'Toccate la barra o un numero' : SCALE.find(([t]) => value >= t)[1];
}

/** Nome e giorno, senza il punteggio del Sudometro che c'è nel nome del file (non deve influenzare il voto) */
function titleOf(name) {
  const base = name.split('/').pop().replace(/\.\w+$/, '');
  const m = base.match(/^(\d{4}-\d{2}-\d{2})_\d+_(.+)_\d+(_debug)?$/);
  if (m) return `${m[2]} · ${m[1].split('-').reverse().join('/')}`;
  return isRef(name) ? `Riferimento · ${base.split('_')[0]}` : base;
}

async function openEditor(name) {
  current = name;
  $('ed-photo').src = await urlOf(name);
  $('ed-title').textContent = titleOf(name);
  const s = scoreOf(name);
  setValue(typeof s === 'number' && scores.has(name) ? s : null);
  $('editor').hidden = false;
  document.body.style.overflow = 'hidden';
}

function closeEditor() {
  $('editor').hidden = true;
  document.body.style.overflow = '';
  current = null;
  refresh();
}

async function save(v) {
  await put('punteggi', current, v);
  scores.set(current, v);
  const list = onlyTodo ? photos.filter((n) => scoreOf(n) === undefined) : photos;
  if (!list.length) return closeEditor();
  const i = list.indexOf(current); // in "Da fare" la foto appena votata è già uscita dall'elenco
  openEditor(list[(i + 1) % list.length]);
}

function step(d) {
  const i = photos.indexOf(current);
  openEditor(photos[(i + d + photos.length) % photos.length]);
}

$('slider').oninput = (e) => setValue(Number(e.target.value));
$('ed-minus').onclick = () => setValue((value ?? 0) - 5);
$('ed-plus').onclick = () => setValue((value ?? 0) + 5);
document.querySelectorAll('.quick button').forEach((b) => (b.onclick = () => setValue(Number(b.dataset.v))));
$('ed-save').onclick = () => (value == null ? toast('Prima scegliete un voto') : save(value));
$('ed-skip').onclick = () => save(null);
$('ed-close').onclick = closeEditor;
$('ed-prev').onclick = () => step(-1);
$('ed-next').onclick = () => step(1);

function toast(text) {
  const t = document.createElement('div');
  t.className = 'toast';
  t.textContent = text;
  document.body.append(t);
  setTimeout(() => t.remove(), 2500);
}

refresh();
