import os

# Configurazione per il salvataggio del modello
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Configurazione per il salvataggio periodico
SAVE_FREQUENCY = 5  # Salva il modello ogni 5 epoche durante il training supervisionato
RL_SAVE_FREQUENCY = 10  # Salva il modello ogni 10 episodi durante il training RL
MAX_CHECKPOINTS = 3  # Mantieni solo gli ultimi 3 checkpoint