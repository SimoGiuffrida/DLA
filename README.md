# Sentiment Analysis con Reinforcement Learning su Recensioni Amazon

Questo progetto implementa un sistema di analisi del sentiment su recensioni Amazon utilizzando una combinazione di deep learning e reinforcement learning. Il modello è addestrato sul dataset McAuley-Lab/Amazon-Reviews-2023, focalizzandosi sulla categoria "Toys_and_Games".

## Struttura del Progetto

```
src/
├── data_preprocessing.py  # Preprocessing e caricamento dei dati
├── sentiment_model.py     # Modello BERT per sentiment analysis
├── rl_trainer.py         # Implementazione PPO per fine-tuning
└── main.py               # Script principale di training
```

## Requisiti

Installare le dipendenze necessarie:

```bash
pip install -r requirements.txt
```

## Funzionalità

1. **Preprocessing dei Dati**:
   - Caricamento del dataset Amazon Reviews
   - Tokenizzazione del testo usando BERT
   - Conversione dei rating in categorie di sentiment

2. **Modello di Sentiment Analysis**:
   - Basato su BERT pre-addestrato
   - Fine-tuning per classificazione del sentiment
   - Metriche: accuracy, loss

3. **Reinforcement Learning**:
   - Implementazione PPO per ottimizzazione
   - Ambiente personalizzato per sentiment analysis
   - Reward basato sul miglioramento delle performance

## Esecuzione

Per avviare il training:

```bash
python src/main.py
```

Il processo include:
1. Training supervisionato iniziale
2. Fine-tuning con reinforcement learning
3. Valutazione finale sul test set

## Monitoraggio

Il progetto utilizza Weights & Biases (wandb) per il monitoraggio degli esperimenti, tracciando:
- Loss e accuracy del training
- Performance di validazione
- Reward del reinforcement learning