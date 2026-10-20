// Prova del classificatore: mappa delle zone sudate su un selfie o su tutte le foto di uno zip del Sudometro.

import { readZip } from '../lib/zip.js';
import { loadClassifier, drawHeat, share, loadImage } from '../lib/browser.js';

const $ = (id) => document.getElementById(id);
let clf = null;
let soft = false;
let results = []; // { name, img, out }
let selected = null;

async function start() {
  try {
    clf = await loadClassifier();
  } catch (err) {
    $('status').textContent = `Errore: ${err.message}`;
    return;
  }
  if (!clf) {
    $('status').textContent = 'Nessun modello addestrato ancora: prima le etichette, poi l\'addestramento (vedi README).';
    return;
  }
  const { info } = clf;
  $('thr').value = info.soglia;
  showThreshold();
  const v = info.validazione;
  $('status').textContent = `Modello del ${new Date(info.creato).toLocaleDateString('it-IT')}, ${info.foto.totali} foto di ${info.foto.persone} persone`
    + (v ? ` · su persone mai viste F1 ${v.f1}, AUC ${v.auc}` : '');
  $('info').textContent = JSON.stringify(info, null, 2);
}

function showThreshold() {
  $('thr-v').textContent = Number($('thr').value).toFixed(2);
}

function render(canvas, { img, out }) {
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0);
  drawHeat(ctx, out, Number($('thr').value), { soft });
}

function renderAll() {
  const thr = Number($('thr').value);
  if (selected) {
    render($('big'), selected);
    $('big-info').textContent = `${selected.name} · riquadri sudati: ${Math.round(share(selected.out.prob, thr) * 100)}%`;
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
    tag.textContent = `${Math.round(share(r.out.prob, thr) * 100)}%`;
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
  const r = { name, img, out: clf.run(img) };
  results.push(r);
  return r;
}

$('in-photo').onchange = async (e) => {
  const f = e.target.files[0];
  e.target.value = '';
  if (!f || !clf) return;
  selected = await add(f.name, f);
  $('big').hidden = false;
  renderAll();
};

$('in-zip').onchange = async (e) => {
  const f = e.target.files[0];
  e.target.value = '';
  if (!f || !clf) return;
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
