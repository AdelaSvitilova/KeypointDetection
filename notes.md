# Text

## Dataset

je potřeba načíst tak aby byl v numpy, či podobně, nesmí být vázaný na framework pro neuronové sítě

formát obrázků je canel-height-weight

## Zeptat se


Jaké hodnoty má bod, co není anotován??? 

Co dělat s visible hodnotou? Co ty body, co jsou anotovány, ale nejsou viditelné? 

Punta - kam mám v local data ukládat věci - já teď mám složku v home, protože se mi moc nelíbí to mít ve Skretch, ale je to jedno.... 

Jak spouštět skripty a moct zavčít terminál? Můžu ho prostě jen zavřít? 

Já si musela napsat collate funkci, kontrola?

Kde a jak bych ideálně měla řešit resize - do modelu vstupuje 256x256, ale výstupy jsou 64x64.... 

Jak mám škálovat metriky? 

parametr save every epoch? 

Logování něčeho?

Epoch 1 | Train Loss: 0.0043 | Train Metrics: {'PCKHeatmaps': '0.0112'} | Val Loss: 949922044051.4561 | Val Metrics: {'PCKHeatmaps': '0.0064'}
Epoch 2 | Train Loss: 0.0004 | Train Metrics: {'PCKHeatmaps': '0.0112'} | Val Loss: 876201.6160 | Val Metrics: {'PCKHeatmaps': '0.0052'}
Epoch 3 | Train Loss: 0.0002 | Train Metrics: {'PCKHeatmaps': '0.0098'} | Val Loss: 17.3408 | Val Metrics: {'PCKHeatmaps': '0.0231'}
Epoch 4 | Train Loss: 0.0001 | Train Metrics: {'PCKHeatmaps': '0.0093'} | Val Loss: 0.0667 | Val Metrics: {'PCKHeatmaps': '0.0320'}
Epoch 5 | Train Loss: 0.0001 | Train Metrics: {'PCKHeatmaps': '0.0099'} | Val Loss: 0.0042 | Val Metrics: {'PCKHeatmaps': '0.0127'}
Epoch 6 | Train Loss: 0.0001 | Train Metrics: {'PCKHeatmaps': '0.0121'} | Val Loss: 0.0004 | Val Metrics: {'PCKHeatmaps': '0.0087'}
Epoch 7 | Train Loss: 0.0000 | Train Metrics: {'PCKHeatmaps': '0.0134'} | Val Loss: 0.0001 | Val Metrics: {'PCKHeatmaps': '0.0094'}
Epoch 8 | Train Loss: 0.0000 | Train Metrics: {'PCKHeatmaps': '0.0139'} | Val Loss: 0.0001 | Val Metrics: {'PCKHeatmaps': '0.0120'}
Epoch 9 | Train Loss: 0.0000 | Train Metrics: {'PCKHeatmaps': '0.0132'} | Val Loss: 0.0000 | Val Metrics: {'PCKHeatmaps': '0.0132'}
Epoch 10 | Train Loss: 0.0000 | Train Metrics: {'PCKHeatmaps': '0.0165'} | Val Loss: 0.0000 | Val Metrics: {'PCKHeatmaps': '0.0155'}
Epoch 11 | Train Loss: 0.0000 | Train Metrics: {'PCKHeatmaps': '0.0172'} | Val Loss: 0.0000 | Val Metrics: {'PCKHeatmaps': '0.0153'}
Epoch 12 | Train Loss: 0.0000 | Train Metrics: {'PCKHeatmaps': '0.0174'} | Val Loss: 0.0000 | Val Metrics: {'PCKHeatmaps': '0.0146'}
Epoch 13 | Train Loss: 0.0000 | Train Metrics: {'PCKHeatmaps': '0.0175'} | Val Loss: 0.0000 | Val Metrics: {'PCKHeatmaps': '0.0167'}
Epoch 14 | Train Loss: 0.0000 | Train Metrics: {'PCKHeatmaps': '0.0180'} | Val Loss: 0.0000 | Val Metrics: {'PCKHeatmaps': '0.0181'}
Epoch 15 | Train Loss: 0.0000 | Train Metrics: {'PCKHeatmaps': '0.0182'} | Val Loss: 0.0000 | Val Metrics: {'PCKHeatmaps': '0.0186'}
Epoch 16 | Train Loss: 0.0000 | Train Metrics: {'PCKHeatmaps': '0.0208'} | Val Loss: 0.0000 | Val Metrics: {'PCKHeatmaps': '0.0193'}
Epoch 17 | Train Loss: 0.0000 | Train Metrics: {'PCKHeatmaps': '0.0226'} | Val Loss: 0.0000 | Val Metrics: {'PCKHeatmaps': '0.0207'}
Epoch 18 | Train Loss: 0.0000 | Train Metrics: {'PCKHeatmaps': '0.0239'} | Val Loss: 0.0000 | Val Metrics: {'PCKHeatmaps': '0.0219'}
Epoch 19 | Train Loss: 0.0000 | Train Metrics: {'PCKHeatmaps': '0.0251'} | Val Loss: 0.0000 | Val Metrics: {'PCKHeatmaps': '0.0245'}
Epoch 20 | Train Loss: 0.0000 | Train Metrics: {'PCKHeatmaps': '0.0248'} | Val Loss: 0.0000 | Val Metrics: {'PCKHeatmaps': '0.0264'}
Epoch 21 | Train Loss: 0.0000 | Train Metrics: {'PCKHeatmaps': '0.0276'} | Val Loss: 0.0000 | Val Metrics: {'PCKHeatmaps': '0.0256'}
Epoch 22 | Train Loss: 0.0000 | Train Metrics: {'PCKHeatmaps': '0.0284'} | Val Loss: 0.0000 | Val Metrics: {'PCKHeatmaps': '0.0254'}
Epoch 23 | Train Loss: 0.0000 | Train Metrics: {'PCKHeatmaps': '0.0282'} | Val Loss: 0.0000 | Val Metrics: {'PCKHeatmaps': '0.0268'}
Epoch 24 | Train Loss: 0.0000 | Train Metrics: {'PCKHeatmaps': '0.0268'} | Val Loss: 0.0000 | Val Metrics: {'PCKHeatmaps': '0.0282'}
Epoch 25 | Train Loss: 0.0000 | Train Metrics: {'PCKHeatmaps': '0.0286'} | Val Loss: 0.0000 | Val Metrics: {'PCKHeatmaps': '0.0289'}
Epoch 26 | Train Loss: 0.0000 | Train Metrics: {'PCKHeatmaps': '0.0316'} | Val Loss: 0.0000 | Val Metrics: {'PCKHeatmaps': '0.0292'}
Epoch 27 | Train Loss: 0.0000 | Train Metrics: {'PCKHeatmaps': '0.0327'} | Val Loss: 0.0000 | Val Metrics: {'PCKHeatmaps': '0.0292'}
Epoch 28 | Train Loss: 0.0000 | Train Metrics: {'PCKHeatmaps': '0.0299'} | Val Loss: 0.0000 | Val Metrics: {'PCKHeatmaps': '0.0336'}
Epoch 29 | Train Loss: 0.0000 | Train Metrics: {'PCKHeatmaps': '0.0333'} | Val Loss: 0.0000 | Val Metrics: {'PCKHeatmaps': '0.0348'}
Epoch 30 | Train Loss: 0.0000 | Train Metrics: {'PCKHeatmaps': '0.0344'} | Val Loss: 0.0000 | Val Metrics: {'PCKHeatmaps': '0.0329'}