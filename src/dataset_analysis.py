import torch
from datasets import load_dataset
from data_preprocessing import DataPreprocessor

def analyze_dataset(category="raw_review_Toys_and_Games"):
    print(f"\n✅ Avvio analisi per la categoria: {category}\n")
    
    # Carica il dataset completo da Hugging Face
    print("Caricamento dataset da Hugging Face...")
    dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", category)

    # Mostra tutti gli split disponibili
    print("\n✅ Split disponibili nel dataset:")
    for split_name in dataset.keys():
        print(f" - {split_name} (esempi: {len(dataset[split_name])})")

    # Se 'train' non esiste, usa il primo disponibile
    split_to_use = 'train' if 'train' in dataset else list(dataset.keys())[0]
    print(f"\n✅ Split selezionato per l'analisi: {split_to_use}")

    data_split = dataset[split_to_use]

    # Mostra tutte le feature disponibili
    print("\n✅ Feature disponibili nel dataset:")
    for feature_name in data_split.column_names:
        print(f" - {feature_name}")

    # Controlla se 'text' è disponibile
    if 'text' not in data_split.column_names:
        print("\n⚠️  La feature 'text' non è presente nel dataset. Analisi terminata.")
        return

    # Visualizza i primi 3 esempi grezzi
    print("\n✅ Primi 3 esempi grezzi del dataset:")
    for idx in range(min(3, len(data_split))):
        print(f"\n--- Esempio {idx + 1} ---")
        print(f"Review Body: {data_split[idx]['text']}")
        print(f"Star Rating: {data_split[idx]['rating']}")

    # Ora procedi con la tua pipeline di preprocessing
    print("\n✅ Avvio del preprocessing...\n")
    preprocessor = DataPreprocessor(category=category)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = preprocessor.prepare_dataset(split_to_use=split_to_use)

    # Stampa informazioni di base sulle dimensioni dei dataset
    print(f"Numero di esempi nel Training Set: {len(y_train)}")
    print(f"Numero di esempi nel Validation Set: {len(y_val)}")
    print(f"Numero di esempi nel Test Set: {len(y_test)}")

    # Analizza una manciata di esempi dal training set
    print("\n✅ Esempi preprocessati dal Training Set:")
    for idx in range(min(3, len(y_train))):
        print(f"\n--- Esempio {idx + 1} ---")
        print("Input IDs:", X_train[idx]['input_ids'])
        print("Attention Mask:", X_train[idx]['attention_mask'])
        print("Label:", y_train[idx])

    print("\n✅ Analisi completata.\n")

if __name__ == "__main__":
    analyze_dataset()
