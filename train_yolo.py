import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from ultralytics import YOLO

def main():
    # 1. Carica il modello base (che vogliamo migliorare con il fine-tuning)
    model = YOLO("yolo26s-seg.pt")
    print("Inizio il Fine-Tuning di YOLO con le etichette create da LangSAM...")
    
    # 2. Avvia l'addestramento
    # Assicurati che nel data.yaml i percorsi alle cartelle train/val siano corretti.
    results = model.train(
        data="data.yaml",             # Il file di configurazione del dataset
        epochs=50,                    # Numero di epoche (cicli) di addestramento
        imgsz=1080,                    # Risoluzione alta per api piccole
        batch=4,                      # Abbassato per avere più iterazioni ed evitare OOM
        device=0,                     # Usa la GPU primaria
        workers=2,                    # FONDAMENTALE SU WINDOWS
        project="runs/segment",       
        name="bee_model_finetuned_yolo26s",   
        patience=20,                  # Alzato un po' per dare più tempo al modello di imparare
        lr0=0.001,                    # Learning rate ridotto per evitare instabilità/NaN con il modello 's'
        optimizer='auto'              # Ottimizzatore automatico
    )
    
    print("\n Addestramento completato!")
    print("I pesi finali del tuo nuovo modello personalizzato si trovano in: runs/segment/bee_model_finetuned_yolo26s/weights/best.pt")

if __name__ == "__main__":
    main()
