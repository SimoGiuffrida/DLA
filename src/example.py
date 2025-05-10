import torch
from transformers import AutoTokenizer
from sentiment_model import SentimentClassifier

def predict_sentiment(text, model, tokenizer, device):
    # Prepara il testo per il modello
    encoded = tokenizer(text,
                       truncation=True,
                       padding='max_length',
                       max_length=512,
                       return_tensors='pt')
    
    # Sposta i tensori sul device corretto
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    
    # Predizione
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        _, predicted = torch.max(outputs, 1)
    
    # Converti la predizione in sentiment
    sentiment_map = {0: 'Negativo', 1: 'Neutro', 2: 'Positivo'}
    return sentiment_map[predicted.item()]

def main():
    # Imposta il device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilizzo device: {device}")
    
    # Carica il modello e il tokenizer
    model = SentimentClassifier().to(device)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # Esempio di recensioni
    recensioni = [
        "Questo giocattolo è fantastico! Mio figlio lo adora e ci gioca ogni giorno.",
        "Prodotto nella media, niente di speciale ma fa il suo dovere.",
        "Pessima qualità, si è rotto dopo pochi giorni. Non lo consiglio."
    ]
    
    # Analizza ogni recensione
    print("\nAnalisi del sentiment delle recensioni:")
    for recensione in recensioni:
        sentiment = predict_sentiment(recensione, model, tokenizer, device)
        print(f"\nRecensione: {recensione}")
        print(f"Sentiment: {sentiment}")

if __name__ == "__main__":
    main()