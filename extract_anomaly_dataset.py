import os
import cv2
import glob
import json
import numpy as np
from ultralytics import YOLO

# Nome del file manifesto che traccia le immagini sorgente già processate
MANIFEST_FILENAME = "_extraction_manifest.json"


def load_manifest(out_dir):
    """Carica il manifesto delle immagini sorgente già processate."""
    manifest_path = os.path.join(out_dir, MANIFEST_FILENAME)
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            return json.load(f)
    return {"processed_sources": {}, "next_bee_id": 0}


def save_manifest(out_dir, manifest):
    """Salva il manifesto aggiornato su disco."""
    manifest_path = os.path.join(out_dir, MANIFEST_FILENAME)
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)


def main():
    print(" Caricamento del modello YOLO...")
    model = YOLO("runs/segment/runs/segment/bee_model_finetuned_yolo26s-2/weights/best.pt")
    
    # Scegli la cartella da cui estrarre le api (qui usiamo il train set remoto)
    source_dir = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/train/images"
    
    # Output: cartella con TUTTE le api estratte grezze (senza filtri)
    out_dir = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/train/anomaly_bees"
    os.makedirs(out_dir, exist_ok=True)
    
    # Carica il manifesto per sapere quali immagini sorgente sono già state processate
    manifest = load_manifest(out_dir)
    
    print(f"Eseguendo l'estrazione delle api da: {source_dir}")
    
    image_paths = (
        glob.glob(os.path.join(source_dir, "*.[jJ][pP][gG]")) +
        glob.glob(os.path.join(source_dir, "*.[pP][nN][gG]"))
    )
    
    bee_counter = manifest["next_bee_id"]
    skipped = 0
    extracted = 0
    
    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        
        # Skip se questa immagine sorgente è già stata processata in una run precedente
        if img_name in manifest["processed_sources"]:
            skipped += 1
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Facciamo inferenza alla massima risoluzione. retina_masks=True migliora enormemente i bordi.
        results = model.predict(source=img, save=False, show=False, conf=0.35, imgsz=1920, max_det=2000)
        
        if len(results) == 0 or results[0].masks is None:
            manifest["processed_sources"][img_name] = {"bees": 0}
            save_manifest(out_dir, manifest)
            continue
            
        res = results[0]
        boxes = res.boxes.xyxy.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy()
        
        bees_from_this_image = 0
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            h, w = img.shape[:2]
            
            # Crea una maschera nera delle dimensioni dell'immagine originale
            blank_mask = np.zeros((h, w), dtype=np.uint8)
            
            # Prendi le coordinate del poligono di questa specifica ape
            polygon = res.masks.xy[i]
            if len(polygon) == 0:
                continue
            
            # Disegna il poligono riempito di bianco sulla maschera nera
            pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(blank_mask, [pts], 255)
            
            # ---------------------------------------------------------------
            # NOTA: le operazioni di raffinamento della maschera (chiusura
            # morfologica, dilatazione, GaussianBlur+threshold) NON vengono
            # più eseguite qui. Sono centralizzate in dataset_cleaning.py
            # per evitare duplicazioni e preservare la qualità originale.
            # ---------------------------------------------------------------
            
            # Crop con padding generoso attorno al bounding box
            padding = 20
            x1_p = max(0, x1 - padding)
            y1_p = max(0, y1 - padding)
            x2_p = min(w, x2 + padding)
            y2_p = min(h, y2 + padding)
            
            # Ritagliamo sia la foto che la maschera usando le coordinate maggiorate
            img_crop = img[y1_p:y2_p, x1_p:x2_p].copy()
            mask_crop = blank_mask[y1_p:y2_p, x1_p:x2_p]
            
            # ---------------------------------------------------------------
            # Salva come RGBA PNG: i canali BGR contengono l'immagine originale
            # non modificata, il canale Alpha contiene la maschera grezza del
            # poligono YOLO. Questo preserva sia i pixel originali sia la
            # maschera senza alcuna perdita di qualità.
            #
            # NON facciamo più il resize a 512x512 qui: il resize singolo
            # avviene in dataset_cleaning.py dopo tutti i filtri, evitando
            # il doppio resize che degradava l'immagine.
            # ---------------------------------------------------------------
            b_ch, g_ch, r_ch = cv2.split(img_crop)
            rgba = cv2.merge([b_ch, g_ch, r_ch, mask_crop])
            
            # Salva il risultato con la confidenza nel nome per debug (PNG lossless con zero compressione per massima fedeltà)
            bee_filename = f"bee_{bee_counter:05d}_conf{confs[i]:.2f}.png"
            cv2.imwrite(os.path.join(out_dir, bee_filename), rgba, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            bee_counter += 1
            bees_from_this_image += 1
            extracted += 1
        
        # Aggiorna il manifesto dopo ogni immagine sorgente per robustezza a interruzioni
        manifest["processed_sources"][img_name] = {"bees": bees_from_this_image}
        manifest["next_bee_id"] = bee_counter
        save_manifest(out_dir, manifest)
            
    print(f"\n✅ Estrazione completata! Estratte {extracted} nuove api ({skipped} immagini sorgente già processate, saltate).")
    print(f"Totale api nella cartella: {bee_counter}")
    print(f"Api grezze salvate in '{out_dir}'.")
    print("Ora lancia dataset_cleaning.py per pulire e normalizzare il dataset.")

if __name__ == "__main__":
    main()
