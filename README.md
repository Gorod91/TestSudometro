# 💧 TestSudometro

Laboratorio dei modelli del sudore per il Sudometro di [Palestrina](https://github.com/Gorod91/Palestrina). Due strade:

- **Punteggio** (più semplice da preparare): date un voto 0-100 a ogni foto, il modello impara a dare lo stesso voto a foto nuove.
- **Zone**: colorate le zone sudate, il modello disegna la mappa.

## Punteggio 0-100

1. **Foto**: Palestrina → Admin → *Scarica le foto del Sudometro (zip)*.
2. **Voti**: aprite `punteggio/`, *Apri lo zip delle foto*, e per ogni foto scegliete il voto (barra, ±5 o 0/25/50/75/100) → *Salva e avanti*. 0 asciutto, 50 sudato ben visibile, 100 fradicio; *Non valutabile* per foto inutilizzabili. I riferimenti valgono già 0. Il punteggio del Sudometro non si vede, per non influenzare il voto.
3. *Esporta voti* → `punteggi-AAAA-MM-GG.json`: mettetelo nella cartella **TestSudometro** di Drive insieme allo zip delle foto e chiedete a Claude di addestrare, oppure:
   ```
   node addestra/addestra-punteggio.js --foto sudometro-AAAA-MM-GG.zip --punteggi punteggi-AAAA-MM-GG.json
   ```
   Prima misura l'errore medio su persone mai viste (contro chi dà sempre il voto medio), poi salva `modello-punteggio/modello.json`.
4. **Prova**: `prova/` mostra il punteggio stimato di ogni foto.

Come funziona: dalla stessa MobileNetV2 si prendono 3232 caratteristiche della foto (media e massimo di quattro livelli, anche sulla foto specchiata) e sopra c'è una **regressione ridge** (somma pesata con pesi "frenati"), che funziona anche con poche decine di foto. In addestramento ogni foto conta anche con 4 variazioni di ritaglio e luce.

## Zone sudate

Il **classificatore delle zone di pelle sudata**: su una foto del viso (ritaglio 3:4 come quelle del Sudometro) dà una **mappa a riquadri** (36 × 48, uno ogni 8 px su 288 × 384) con la probabilità che lì la pelle sia sudata. Gira tutto nel browser del telefono con TensorFlow.js, senza server.

Per ora resta qui: in Palestrina entra solo quando funziona bene sulle foto vere.

### Come è fatto

```
foto 3:4 ──► MobileNetV2 0.35 (pre-addestrata su ImageNet, ferma, 1,6 MB)
               ├─ livello a passo 4  (riflessi piccoli, gocce)
               ├─ livello a passo 8
               └─ livello a passo 16 (dove siamo nel viso)
                     ▼
             "testa" piccola (~35 mila pesi, ~150 KB) ──► griglia 48 × 36 di probabilità
```

Si addestra solo la testa: con poche decine di foto etichettate è l'unico modo di non imparare a memoria.
L'addestramento usa variazioni casuali (ritaglio, specchio, luce) e pesa di più i riquadri sudati, che sono pochi.

| Cartella | Cosa c'è |
| --- | --- |
| `etichetta/` | Strumento web per colorare le zone sudate (anche da telefono) |
| `punteggio/` | Voti 0-100 alle foto |
| `prova/` | Punteggio e mappa su un selfie o su tutto uno zip del Sudometro |
| `addestra/` | Addestramento (Node) e generatore di dati finti per le prove |
| `lib/` | Modello, zip e parti comuni (stesso codice nel browser e in Node) |
| `modello-punteggio/` | Il punteggio addestrato (`modello.json`); vuota finché non c'è |
| `modello/` | Il modello delle zone addestrato (`testa/` + `info.json` con soglia e misure); vuota finché non c'è |
| `vendor/` | MobileNetV2 0.35 (pesi Keras, licenza Apache 2.0) |

### I passi

1. **Foto**: Palestrina → Admin → *Scarica le foto del Sudometro (zip)*.
2. **Etichette**: aprite `etichetta/` (dal sito Pages del repo o in locale), *Apri lo zip delle foto*, e per ogni foto da sudato:
   - **Sudato** (rosa): pelle lucida a puntini, gocce, pelle bagnata;
   - **Incerto** (giallo): i dubbi, non contano;
   - niente colore = asciutto; **Tutto asciutto** per le foto senza sudore, **Scarta** per le foto inutilizzabili.
   I riferimenti contano già come asciutti. Il lavoro resta salvato nel browser; *Esporta etichette* crea lo zip.
3. **Addestramento**: mettete i due zip (foto + etichette) su Google Drive e chiedete a Claude di addestrare, oppure da computer:
   ```
   npm install
   node addestra/addestra.js --foto sudometro-AAAA-MM-GG.zip --etichette etichette-AAAA-MM-GG.zip
   ```
   Prima misura il modello su **persone mai viste** (validazione a gruppi per persona: F1, AUC, IoU dei riquadri), poi si addestra su tutto e salva in `modello/`. Il commit di `modello/` lo pubblica.
4. **Prova**: `prova/` mostra la mappa su un selfie o su tutte le foto di uno zip, con la soglia regolabile.
5. **Giro dopo**: con un modello già pubblicato, nello strumento delle etichette compare **✨ Proponi**: colora i riquadri che il modello vede sudati, basta correggerli. Più foto etichettate → modello migliore.

**Le foto non vanno mai nel repo**: `*.zip` e `dati/` sono esclusi da `.gitignore`.

## Prove senza foto vere

```
npm install
node addestra/dati-finti.js        # dati/finti/: 30 "visi" finti con zone di puntini lucidi e voti finti
node addestra/addestra.js --foto dati/finti/sudometro-finto.zip --etichette dati/finti/etichette-finte.zip --epoche 10 --out dati/modello-finto
node addestra/addestra-punteggio.js --foto dati/finti/sudometro-finto.zip --punteggi dati/finti/punteggi-finti.json --out dati/punteggio-finto
npm run serve                      # http://localhost:8080
```

Sui dati finti il modello delle zone separa bene i puntini sudati dalla macchia lucida della luce (F1 ~0,94 su persone mai viste): serve solo a provare che il giro funziona, non dice niente sulle foto vere.

## Pubblicazione

*Settings → Pages → Source: GitHub Actions*, poi *Actions → Pubblica su GitHub Pages → Run workflow*. Indirizzo: `https://gorod91.github.io/TestSudometro/` (con repo privato GitHub Pages richiede un piano a pagamento: in quel caso `npm run serve` in locale).
