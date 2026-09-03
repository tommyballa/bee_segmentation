import os
from ultralytics import YOLO

def main():
    # Carica i pesi salvati all'ultima epoca (32) del tuo addestramento
    # Ho corretto il percorso che per qualche motivo aveva un doppio "runs/segment"
    model = YOLO("runs/segment/runs/segment/bee_model_finetuned-5/weights/best.pt")

    # Invece di una singola immagine, passiamo l'intera cartella di validazione!
    test_dir = "bee_dataset/images/val" 
    
    # Assicuriamoci che la cartella esista prima di procedere
    if not os.path.exists(test_dir):
        print(f"⚠️ Cartella non trovata: {test_dir}.")
        return

    import cv2
    import glob

    print(f"Eseguendo il TUO modello addestrato fino all'epoca 32 su tutto il validation set: {test_dir}")
    
    # Creiamo la cartella di output
    out_dir = "runs/segment/predict_resized"
    os.makedirs(out_dir, exist_ok=True)
    
    # Processiamo le immagini una per una ridimensionandole
    image_paths = glob.glob(os.path.join(test_dir, "*.[jJ][pP][gG]")) + glob.glob(os.path.join(test_dir, "*.[pP][nN][gG]"))
    
    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None: continue
        
        # Ridimensioniamo le immagini giganti per evitare OOM durante il disegno delle maschere
        MAX_DIM = 1200
        h, w = img.shape[:2]
        if max(h, w) > MAX_DIM:
            scale = MAX_DIM / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
            
        # Predizione sull'immagine ridimensionata
        results = model.predict(source=img, save=False, show=False, conf=0.10)        
        
        # Filtriamo le predizioni calcolando l'area MEDIANA delle api (come nel tuo pass1 di SAM)
        if len(results) > 0 and len(results[0].boxes) > 0:
            res = results[0]
            keep_indices = []
            
            # 1. Raccogliamo le aree di tutte le predizioni
            areas = []
            for box in res.boxes.xywh:
                w, h = box[2].item(), box[3].item()
                areas.append(w * h)
                
            # 2. Calcoliamo l'area mediana (ignorando così i picchi enormi o minuscoli)
            import numpy as np
            median_area = np.median(areas)
            
            # 3. La soglia minima sarà il 15% dell'area media di un'ape in QUESTA specifica foto
            MIN_AREA = median_area * 0.30
            print(f"[{img_name}] Area mediana: {median_area:.0f} px -> Scarto tutto sotto i {MIN_AREA:.0f} px")
            
            for i, area in enumerate(areas):
                if area >= MIN_AREA:
                    keep_indices.append(i)
                    
            # Aggiorna i risultati mantenendo solo le api sufficientemente grandi
            res = res[keep_indices]
            
            if len(keep_indices) > 0:
                # labels=False nasconde la scritta "bee" e la percentuale
                # boxes=False nasconde il riquadro rettangolare, lasciando SOLO la maschera sagomata
                res_img = res.plot(labels=False, boxes=False)
                cv2.imwrite(os.path.join(out_dir, img_name), res_img)
                print(f"Salvato {img_name} in {out_dir}")
    
    print("\n✅ Inferenza completata!")
    print(f"Controlla la cartella '{out_dir}' per vedere i risultati.")

if __name__ == "__main__":
    main()
