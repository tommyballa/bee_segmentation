import os
from ultralytics import YOLO

def main():
    print(" Caricamento del modello...")

    model = YOLO("runs/segment/runs/segment/bee_model_finetuned-5/weights/best.pt")

    print("\n Inizio la Validazione (calcolo metriche Precision, Recall, mAP)...")


    metrics = model.val(
        data="bee_dataset/data.yaml",
        imgsz=800,
        batch=16,
        device=0
    )
    
    print("\n Validazione completata! Controlla i risultati numerici qui sopra.")
    print("I grafici (Curve Precision-Recall, Confusion Matrix, ecc.) sono stati salvati nella nuova cartella generata in 'runs/segment/val'")

if __name__ == "__main__":
    main()
