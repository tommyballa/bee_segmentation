import os
from ultralytics import YOLO

def main():
    print("⏳ Caricamento del modello...")
    # Carica i pesi del tuo addestramento (epoca 32)
    model = YOLO("runs/segment/runs/segment/bee_model_finetuned-5/weights/best.pt")

    print("\n🚀 Inizio la Validazione (calcolo metriche Precision, Recall, mAP)...")
    # model.val() esegue il calcolo delle metriche usando le etichette in bee_dataset/data.yaml
    # batch=2 serve per evitare l'errore di memoria (OOM) che abbiamo visto prima
    # imgsz=800 mantiene la stessa risoluzione usata in addestramento
    metrics = model.val(
        data="bee_dataset/data.yaml",
        imgsz=800,
        batch=2,
        half=True, # Usa mezza precisione per risparmiare memoria GPU
        device=0
    )
    
    print("\n✅ Validazione completata! Controlla i risultati numerici qui sopra.")
    print("I grafici (Curve Precision-Recall, Confusion Matrix, ecc.) sono stati salvati nella nuova cartella generata in 'runs/segment/val'")

if __name__ == "__main__":
    main()
