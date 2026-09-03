import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from ultralytics import YOLO

def main():
    # 1. Carica il modello base (che vogliamo migliorare con il fine-tuning)
    model = YOLO("yolo26n-seg.pt")
    print("Inizio il Fine-Tuning di YOLO con le etichette create da LangSAM...")
    
    # 2. Avvia l'addestramento
    # Assicurati che nel data.yaml i percorsi alle cartelle train/val siano corretti.
    results = model.train(
        data="bee_dataset/data.yaml", # Il file di configurazione del dataset
        epochs=50,                    # Numero di epoche (cicli) di addestramento
        imgsz=800,                    # Risoluzione alta per api piccole
        batch=1,                      # Abbassato a 1 per evitare errori di memoria OOM
        device=0,                     # Usa la GPU primaria
        half=True,                    # FONDAMENTALE PER LA TUA GPU! Dimezza l'uso della VRAM senza perdere precisione.
        workers=2,                    # FONDAMENTALE SU WINDOWS! Evita che la CPU si ingolfi e faccia bloccare tutto il pc durante il training.
        project="runs/segment",       
        name="bee_model_finetuned",   
        patience=10                   
    )
    
    print("\n Addestramento completato!")
    print("I pesi finali del tuo nuovo modello personalizzato si trovano in: runs/segment/bee_model_finetuned/weights/best.pt")

if __name__ == "__main__":
    main()
