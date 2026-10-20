// Prova dei modelli: punteggio 0-100 e mappa delle zone sudate su un selfie o su tutte le foto di uno zip del Sudometro.

import { readZip } from '../lib/zip.js';
import { loadClassifier, loadScorer, drawHeat, share, loadImage } from '../lib/browser.js';

const $ = (id) => document.getElementById(id);
let clf = null;
let scorer = null;
let soft = false;
let results = []; // { name, img, out, score }
let selected = null;

async function start() {
  try {
    [scorer, clf] = await Promise.all([loadScorer(), loadClassifier()]);
  } catch (err) {
    $('status').textContent = `Errore: ${err.message}`;
    return;
  }
  if (!clf && !scorer) {
    $('status').textContent = 'Nessun modello addestrato ancora: prima etichette o voti, poi l\'addestramento (vedi README).';
    return;
  }
  const day = (d) => new Date(d).toLocaleDateString('it-IT');
  const lines = [];
  if (scorer) {
    const v = scorer.info.validazione;
    lines.push(`Punteggio del ${day(scorer.info.creato)}, ${scorer.info.foto.totali} foto` + (v ? ` · su persone mai viste sbaglia in media di ${v.erroreMedio} punti` : ''));
  }
  if (clf) {
    const v = clf.info.validazione;
    $('thr').value = clf.info.soglia;
    showThreshold();
    lines.push(`Zone del ${day(clf.info.creato)}, ${clf.info.foto.totali} foto` + (v ? ` · F1 ${v.f1}, AUC ${v.auc}` : ''));
  } else {
    $('thr-row').hidden = true; // soglia e sfumata servono solo alla mappa
  }
  $('status').textContent = lines.join('\n');
  $('status').style.whiteSpace = 'pre-line';
  const { w, ...scoreInfo } = scorer?.info || {};
  $('info').textContent = JSON.stringify({ punteggio: scorer ? scoreInfo : null, zone: clf?.info || null }, null, 2);
}

function describe(r, thr) {
  const parts = [];
  if (r.score != null) parts.push(`punteggio ${r.score}`);
  if (r.out) parts.push(`riquadri sudati ${Math.round(share(r.out.prob, thr) * 100)}%`);
  return parts.join(' · ');
}

function showThreshold() {
  $('thr-v').textContent = Number($('thr').value).toFixed(2);
}

function render(canvas, { img, out }) {
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0);
  if (out) drawHeat(ctx, out, Number($('thr').value), { soft });
}

function renderAll() {
  const thr = Number($('thr').value);
  if (selected) {
    render($('big'), selected);
    $('big-info').textContent = [selected.name, describe(selected, thr)].filter(Boolean).join(' · ');
  }
  const grid = $('grid');
  grid.textContent = '';
  results.forEach((r) => {
    const div = document.createElement('div');
    div.className = 'thumb';
    const cv = document.createElement('canvas');
    render(cv, r);
    const tag = document.createElement('span');
    tag.className = 'tag';
    tag.textContent = r.score != null ? r.score : `${Math.round(share(r.out.prob, thr) * 100)}%`;
    div.append(cv, tag);
    div.onclick = () => {
      selected = r;
      $('big').hidden = false;
      renderAll();
      $('big').scrollIntoView({ behavior: 'smooth' });
    };
    grid.append(div);
  });
}

async function add(name, blob) {
  const img = await loadImage(blob);
  const r = { name, img, out: clf ? clf.run(img) : null, score: scorer ? scorer.run(img) : null };
  results.push(r);
  return r;
}

$('in-photo').onchange = async (e) => {
  const f = e.target.files[0];
  e.target.value = '';
  if (!f || !(clf || scorer)) return;
  selected = await add(f.name, f);
  $('big').hidden = false;
  renderAll();
};

$('in-zip').onchange = async (e) => {
  const f = e.target.files[0];
  e.target.value = '';
  if (!f || !(clf || scorer)) return;
  results = [];
  const entries = (await readZip(await f.arrayBuffer())).filter((x) => /\.jpe?g$/i.test(x.name));
  entries.sort((a, b) => a.name.localeCompare(b.name));
  for (let i = 0; i < entries.length; i++) {
    $('status').textContent = `Analizzo ${i + 1}/${entries.length}…`;
    await add(entries[i].name, new Blob([entries[i].data], { type: 'image/jpeg' }));
    await new Promise((r) => setTimeout(r)); // lascia respirare la pagina
  }
  $('status').textContent = `${entries.length} foto analizzate`;
  renderAll();
};

$('thr').oninput = () => {
  showThreshold();
  renderAll();
};
$('soft').onclick = () => {
  soft = !soft;
  $('soft').classList.toggle('on', soft);
  renderAll();
};

start();
