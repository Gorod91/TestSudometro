// Dati finti per provare tutto il giro (etichette → addestramento → prova) senza le foto vere:
// "visi" color pelle con luce variabile, e su quelli da sudato gruppi di puntini lucidi (le zone sudate).
// Scrive dati/finti/sudometro-finto.zip (stessa forma dello zip dell'admin) e dati/finti/etichette-finte.zip.
//   node addestra/dati-finti.js

import fs from 'node:fs';
import tf from '@tensorflow/tfjs-node';
import { makeZip } from '../lib/zip.js';
import { MASK } from '../lib/modello.js';

const OUT = 'dati/finti';
const PW = 360;
const PH = 480;
const PEOPLE = ['Anna', 'Bruno', 'Carla', 'Dario', 'Elisa', 'Fabio'];
let seed = 7;
const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
const between = (a, b) => a + (b - a) * rnd();

function face(sweaty) {
  const img = new Uint8Array(PW * PH * 3);
  const mask = new Uint8Array(PW * PH);
  const bg = [between(40, 220), between(40, 220), between(40, 220)];
  const skin = [between(170, 235), between(120, 175), between(95, 145)];
  const light = between(0.75, 1.15);
  const lx = between(-0.5, 0.5);
  const cx = PW / 2 + between(-15, 15);
  const cy = PH / 2 + between(-10, 20);
  const rx = between(110, 135);
  const ry = between(150, 175);
  // macchia lucida larga (pelle asciutta sotto un faretto): da non confondere con il sudore
  const glare = rnd() < 0.6 ? { x: cx + between(-60, 60), y: cy + between(-120, 40), r: between(25, 45) } : null;
  // zone sudate: gruppi di puntini piccoli e netti
  const zones = sweaty
    ? Array.from({ length: 1 + Math.floor(rnd() * 3) }, () => ({ x: cx + between(-70, 70), y: cy + between(-130, 60), r: between(25, 50) }))
    : [];
  const dots = zones.flatMap((z) =>
    Array.from({ length: 25 + Math.floor(rnd() * 30) }, () => {
      const a = rnd() * Math.PI * 2;
      const d = Math.sqrt(rnd()) * z.r;
      return { x: z.x + Math.cos(a) * d, y: z.y + Math.sin(a) * d, r: between(1.2, 3) };
    }),
  );
  for (let y = 0; y < PH; y++) {
    for (let x = 0; x < PW; x++) {
      const i = y * PW + x;
      const nx = (x - cx) / rx;
      const ny = (y - cy) / ry;
      const d = nx * nx + ny * ny;
      let c;
      if (d <= 1) {
        const shade = light * (1 - 0.25 * d) * (1 + 0.25 * lx * nx);
        c = skin.map((v) => v * shade + between(-4, 4));
        if (glare) {
          const g = Math.exp(-((x - glare.x) ** 2 + (y - glare.y) ** 2) / (2 * glare.r ** 2));
          c = c.map((v) => v + g * 35);
        }
        for (const p of dots) {
          const q = (x - p.x) ** 2 + (y - p.y) ** 2;
          if (q < p.r * p.r * 4) c = c.map((v) => v + (255 - v) * Math.exp(-q / (p.r * p.r)) * 0.9);
        }
        for (const z of zones) if ((x - z.x) ** 2 + (y - z.y) ** 2 < (z.r + 6) ** 2) mask[i] = MASK.sudato;
      } else {
        c = bg.map((v) => v + between(-6, 6));
      }
      for (let k = 0; k < 3; k++) img[i * 3 + k] = Math.max(0, Math.min(255, c[k]));
    }
  }
  return { img, mask };
}

async function jpeg(img) {
  return tf.node.encodeJpeg(tf.tensor3d(img, [PH, PW, 3], 'int32'), 'rgb', 85);
}
async function png(mask) {
  const rgb = new Uint8Array(PW * PH * 3);
  mask.forEach((v, i) => rgb.fill(v, i * 3, i * 3 + 3));
  return tf.node.encodePng(tf.tensor3d(rgb, [PH, PW, 3], 'int32'));
}

const photos = [];
const labels = [];
const json = { versione: 1, foto: {} };
for (const name of PEOPLE) {
  const ref = `riferimenti/${name}_attuale.jpg`;
  photos.push({ name: ref, data: await jpeg(face(false).img) });
  for (let k = 0; k < 4; k++) {
    const sweaty = k < 3; // una foto "da sudato" che in realtà è asciutta
    const file = `sudato/2026-10-1${k}_1930_${name}_${sweaty ? 70 : 10}.jpg`;
    const { img, mask } = face(sweaty);
    photos.push({ name: file, data: await jpeg(img) });
    if (sweaty) {
      const m = `maschere/${file.replace(/\.jpg$/, '.png')}`;
      labels.push({ name: m, data: await png(mask) });
      json.foto[file] = { stato: 'fatto', maschera: m };
    } else {
      json.foto[file] = { stato: 'asciutto' };
    }
  }
}
labels.push({ name: 'etichette.json', data: new TextEncoder().encode(JSON.stringify(json, null, 2)) });
fs.mkdirSync(OUT, { recursive: true });
const write = async (file, files) => fs.writeFileSync(`${OUT}/${file}`, Buffer.from(await makeZip(files).arrayBuffer()));
await write('sudometro-finto.zip', photos);
await write('etichette-finte.zip', labels);
console.log(`${photos.length} foto finte e ${labels.length - 1} maschere in ${OUT}/`);
