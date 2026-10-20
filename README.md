# 💧 TestSudometro

Laboratorio del **classificatore delle zone di pelle sudata** per il Sudometro di [Palestrina](https://github.com/Gorod91/Palestrina).
Su una foto del viso (ritaglio 3:4 come quelle del Sudometro) dà una **mappa a riquadri** (36 × 48, uno ogni 8 px su 288 × 384) con la probabilità che lì la pelle sia sudata. Gira tutto nel browser del telefono con TensorFlow.js, senza server.

Per ora resta qui: in Palestrina entra solo quando funziona bene sulle foto vere.

## Come è fatto

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
| `prova/` | Mappa del modello su un selfie o su tutto uno zip del Sudometro |
| `addestra/` | Addestramento (Node) e generatore di dati finti per le prove |
| `lib/` | Modello, zip e parti comuni (stesso codice nel browser e in Node) |
| `modello/` | Il modello addestrato (`testa/` + `info.json` con soglia e misure); vuota finché non c'è |
| `vendor/` | MobileNetV2 0.35 (pesi Keras, licenza Apache 2.0) |

## I passi

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
node addestra/dati-finti.js        # dati/finti/: 30 "visi" finti con zone di puntini lucidi
node addestra/addestra.js --foto dati/finti/sudometro-finto.zip --etichette dati/finti/etichette-finte.zip --epoche 10 --out dati/modello-finto
npm run serve                      # http://localhost:8080
```

Sui dati finti il modello separa bene i puntini sudati dalla macchia lucida della luce (F1 ~0,94 su persone mai viste): serve solo a provare che il giro funziona, non dice niente sulle foto vere.

## Pubblicazione

*Settings → Pages → Source: GitHub Actions*, poi *Actions → Pubblica su GitHub Pages → Run workflow*. Indirizzo: `https://gorod91.github.io/TestSudometro/` (con repo privato GitHub Pages richiede un piano a pagamento: in quel caso `npm run serve` in locale).
