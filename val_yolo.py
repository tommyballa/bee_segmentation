import os
from ultralytics import YOLO

def main():
    print(" Caricamento del modello...")

    model = YOLO("runs/segment/runs/segment/bee_model_finetuned_yolo26s-2/weights/best.pt")

    print("\n Inizio la Validazione (calcolo metriche Precision, Recall, mAP)...")


    metrics = model.val(
        data="data.yaml",
        imgsz=1280,
        batch=16,
        device=0,
        conf=0.15,
        max_det=2000
    )
    
    print("\n Validazione completata! Controlla i risultati numerici qui sopra.")
    
    # print mask segmentation metrics
    print(f"Mask mAP50:     {metrics.seg.map50:.4f}")
    print(f"Mask mAP50-95:  {metrics.seg.map:.4f}")
    # metrics.seg.p and metrics.seg.r might not have .mean() in older ultralytics version, but assuming standard format.
    try:
        print(f"Mask Precision: {metrics.seg.p.mean():.4f}")
        print(f"Mask Recall:    {metrics.seg.r.mean():.4f}")
    except AttributeError:
        # Fallback if .mean() is not available on list
        import numpy as np
        print(f"Mask Precision: {np.mean(metrics.seg.p):.4f}")
        print(f"Mask Recall:    {np.mean(metrics.seg.r):.4f}")

    # print bounding box metrics for comparison
    print(f"Box mAP50:      {metrics.box.map50:.4f}")
    print(f"Box mAP50-95:   {metrics.box.map:.4f}")

    print("I grafici (Curve Precision-Recall, Confusion Matrix, ecc.) sono stati salvati nella nuova cartella generata in 'runs/segment/val'")

if __name__ == "__main__":
    main()
