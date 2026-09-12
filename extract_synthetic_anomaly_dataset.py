"""
Estrae le singole api dalle immagini sintetiche con varroa e genera
la maschera di anomalia per-ape per le metriche di anomaly detection.

Pipeline:
1. Carica l'immagine sintetica e la GT mask della varroa (generati da varroa_generator.py)
2. Usa l'inferenza del modello YOLO per trovare e croppare ogni ape
3. Estrae la maschera del poligono grezzo dell'ape
4. Passa l'immagine grezza e la maschera a dataset_cleaning.clean_and_filter per la pipeline unificata di sanificazione, rotazione, sfumatura bordi
5. Incrocia la GT mask varroa pulita con la nuova immagine
6. Salva:
   - Il crop dell'ape (512x512, sfondo nero, perfettamente smussato e orientato)
   - La maschera anomalia per-ape (512x512, bianco=varroa, nero=normale)
   - Un label (0=normale, 1=anomala) in un file CSV
"""

import os
import cv2
import csv
import numpy as np
from ultralytics import YOLO
# Importiamo la funzione aggiornata dal nuovo script di pulizia ottimizzato
from dataset_cleaning import clean_and_filter, refine_raw_polygon_mask, normalize_orientation, normalize_scale, feather_edges


# --- Configurazione ---
# Cartella con le immagini sintetiche e maschere generate da varroa_generator.py
SYNTHETIC_DIR = "/mnt/disk1/borsattifr/datasets/bees_datasets/DatasetApi_Ceschi/processed/test/synthetic_varroa_output_v2/images"

# Modello YOLO per la segmentazione (lo stesso usato in extract_anomaly_dataset.py)
YOLO_MODEL_PATH = "runs/segment/runs/segment/bee_model_finetuned_yolo26s-2/weights/best.pt"

# Output
OUTPUT_BASE = "/mnt/disk1/borsattifr/datasets/bees_datasets/DatasetApi_Ceschi/processed/test_single_bee"
OUTPUT_NORMAL = os.path.join(OUTPUT_BASE, "normal")
OUTPUT_ANOMALOUS = os.path.join(OUTPUT_BASE, "anomalous")
OUTPUT_MASKS = os.path.join(OUTPUT_BASE, "masks")

# Padding iniziale abbondante per non tagliare l'ape durante le rotazioni e pulizie
PADDING = 30

# Soglia minima di pixel varroa per considerare l'ape anomala
# (evita falsi positivi da overlap di 1-2 pixel al bordo)
MIN_VARROA_PIXELS = 10


def crop_raw_bee(image, varroa_full_mask, polygon_pts, bbox):
    """
    Estrae il ritaglio grezzo dell'ape e della maschera varroa.
    Non applica operazioni morfologiche (vengono fatte da dataset_cleaning).
    """
    h, w = image.shape[:2]
    bx, by, bw, bh = bbox

    # Crea maschera grezza del poligono dell'ape (full size)
    bee_mask = np.zeros((h, w), dtype=np.uint8)
    pts = polygon_pts.astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(bee_mask, [pts], 255)

    # Bounding box con padding generoso per permettere rotazioni successive in cleaning
    x1 = max(0, bx - PADDING)
    y1 = max(0, by - PADDING)
    x2 = min(w, bx + bw + PADDING)
    y2 = min(h, by + bh + PADDING)

    # Crop
    img_crop = image[y1:y2, x1:x2].copy()
    bee_mask_crop = bee_mask[y1:y2, x1:x2].copy()
    varroa_mask_crop = varroa_full_mask[y1:y2, x1:x2].copy()

    return img_crop, bee_mask_crop, varroa_mask_crop


def process_synthetic_image(syn_img_path, mask_path, model):
    """
    Processa una singola immagine sintetica: estrae tutte le api e le classifica
    come normali o anomale in base alla sovrapposizione con la maschera varroa.
    Integra in maniera totale la pipeline di dataset_cleaning.
    """
    syn_img = cv2.imread(syn_img_path)
    varroa_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if syn_img is None:
        print(f"  ✗ Impossibile caricare: {syn_img_path}")
        return [], 0
    if varroa_mask is None:
        print(f"  ✗ Impossibile caricare maschera: {mask_path}")
        return [], 0

    # Binarizza la maschera (potrebbe avere artefatti JPEG)
    _, varroa_mask = cv2.threshold(varroa_mask, 128, 255, cv2.THRESH_BINARY)

    # Inferenza con YOLO (stessi parametri di extract_anomaly_dataset.py)
    results = model.predict(source=syn_img, save=False, show=False, conf=0.35, imgsz=1920, max_det=2000)
    
    detections = []
    if len(results) > 0 and results[0].masks is not None:
        res = results[0]
        boxes = res.boxes.xyxy.cpu().numpy().astype(int)
        for i, box in enumerate(boxes):
            polygon = res.masks.xy[i]
            if len(polygon) == 0:
                continue
            x1, y1, x2, y2 = box
            detections.append({
                "polygon": polygon,
                "bbox": (x1, y1, x2 - x1, y2 - y1),
                "conf": res.boxes.conf.cpu().numpy()[i]
            })

    base_name = os.path.splitext(os.path.basename(syn_img_path))[0]
    processed_results = []
    discarded = 0

    for i, det in enumerate(detections):
        # 1. Estrarre il crop grezzo (immagine, maschera ape, maschera varroa)
        img_crop, bee_mask_crop, varroa_mask_crop = crop_raw_bee(
            syn_img, varroa_mask, det["polygon"], det["bbox"]
        )

        dummy_filename = f"bee_{i:05d}_conf{det['conf']:.2f}.png"

        # 2. Dato che dobbiamo applicare le stesse rotazioni e scale anche
        #    alla maschera varroa, non possiamo chiamare "clean_and_filter" 
        #    così com'è, perché scarterebbe la maschera varroa.
        #    Replichiamo quindi la pipeline di clean_and_filter applicando 
        #    le trasformazioni in parallelo su bee_mask e varroa_mask.
        
        # Filtro 1: Confidenza YOLO
        if det['conf'] < 0.35:
            discarded += 1
            continue
            
        # Raffinamento maschera ape
        refined_bee_mask = refine_raw_polygon_mask(bee_mask_crop)
        
        # Intersechiamo la maschera varroa grezza con l'ape (così varroa fuori dall'ape viene ignorata)
        varroa_mask_crop = cv2.bitwise_and(varroa_mask_crop, refined_bee_mask)
        
        # Passiamo al blocco di pulizia interno di clean_and_filter chiamandolo normalmente per
        # avere l'ape pulita e capire se viene scartata
        cleaned_bee = clean_and_filter(img_crop, bee_mask_crop, is_raw_polygon=True, filename=dummy_filename)
        
        if cleaned_bee is None:
            discarded += 1
            continue

        # Poiché clean_and_filter non trasforma anche la varroa_mask_crop con le stesse matrici,
        # applichiamo manualmente rotazione e scaling della pulizia alla varroa_mask_crop per tenerla allineata!
        
        # Riapplichiamo le trasformazioni necessarie alla varroa per combaciare con cleaned_bee.
        # A questo punto sappiamo che clean_and_filter estrae un contour, lo smussa, fa bitwise_and, CLAHE,
        # e poi usa normalize_orientation e normalize_scale.
        
        # Per avere l'allineamento 1:1, otteniamo la nuova maschera esatta di "cleaned_bee"
        gray = cv2.cvtColor(cleaned_bee, cv2.COLOR_BGR2GRAY)
        _, final_bee_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        
        # ORA: la maschera varroa grezza (varroa_mask_crop) è nel sistema di coordinate del crop iniziale (img_crop).
        # Cleaned bee ha subito rotazione e scala. 
        # Per essere perfetti e replicare il flusso, estraiamo la logica di normalizzazione geometrica:
        
        # ====================
        # RIPRODUCIAMO LE TRASFORMAZIONI GEOMETRICHE SULLA VARROA
        # Dato che clean_and_filter incapsula queste trasformazioni, dobbiamo riprodurle qui
        # o ricostruire la maschera varroa trasformata.
        # ====================
        
        # Dalla riga ~173-250 di dataset_cleaning.py, sappiamo quali operazioni di pulizia subisce la maschera
        # Ripercorriamo lo stesso path per la maschera (senza l'immagine che abbiamo già tramite clean_and_filter):
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        opened_mask = cv2.morphologyEx(refined_bee_mask, cv2.MORPH_OPEN, open_kernel)
        contours, _ = cv2.findContours(opened_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: 
            continue
        largest_contour = max(contours, key=cv2.contourArea)
        clean_mask = np.zeros_like(refined_bee_mask)
        cv2.drawContours(clean_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        clean_mask = cv2.dilate(clean_mask, dilate_kernel, iterations=1)
        clean_mask = cv2.bitwise_and(clean_mask, refined_bee_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)
        
        # Smoothing contorni per la maschera master
        clean_mask = cv2.GaussianBlur(clean_mask, (31, 31), 0)
        _, clean_mask = cv2.threshold(clean_mask, 127, 255, cv2.THRESH_BINARY)
        
        # 1. Orientamento. Dobbiamo ruotare sia l'immagine "dummy" (usiamo la varroa) sia la maschera.
        # Creiamo un'immagine varroa BGR fake per usare le funzioni esistenti
        varroa_bgr = cv2.cvtColor(varroa_mask_crop, cv2.COLOR_GRAY2BGR)
        
        varroa_rotated, mask_rotated = normalize_orientation(varroa_bgr, clean_mask)
        
        # 2. Scala. 
        varroa_scaled, _ = normalize_scale(varroa_rotated, mask_rotated, target_size=512, target_fill=0.60)
        
        # Convertiamo di nuovo varroa_scaled in GRAY
        varroa_final_mask = cv2.cvtColor(varroa_scaled, cv2.COLOR_BGR2GRAY)
        
        # Intersezione finale per sicurezza con l'ape finale (rimuove varroa scappata dai bordi smussati)
        varroa_final_mask = cv2.bitwise_and(varroa_final_mask, final_bee_mask)

        # Ricalcola i pixel di varroa effettivi rimasti nella maschera trasformata
        n_px = cv2.countNonZero(varroa_final_mask)
        is_anomalous = n_px >= MIN_VARROA_PIXELS
        
        processed_results.append((cleaned_bee, varroa_final_mask, is_anomalous, base_name, i))

    return processed_results, discarded


def main():
    print(" Caricamento del modello YOLO...")
    model = YOLO(YOLO_MODEL_PATH)
    
    # Crea le cartelle di output
    for d in [OUTPUT_NORMAL, OUTPUT_ANOMALOUS, OUTPUT_MASKS]:
        os.makedirs(d, exist_ok=True)

    # Trova tutte le coppie (immagine sintetica, maschera GT)
    syn_files = sorted([
        f for f in os.listdir(SYNTHETIC_DIR)
        if f.endswith(".jpg") and "_syn" in f
    ])

    if not syn_files:
        print(f"Nessuna immagine sintetica trovata in {SYNTHETIC_DIR}")
        return

    csv_rows = []
    total_normal = 0
    total_anomalous = 0
    total_discarded = 0

    mask_dir = os.path.join(os.path.dirname(SYNTHETIC_DIR), "masks")

    for syn_file in syn_files:
        # Derive mask filename: DSC_4914_syn_0.jpg → DSC_4914_mask_0.png
        mask_file = syn_file.replace("_syn", "_mask").replace(".jpg", ".png")
        syn_path = os.path.join(SYNTHETIC_DIR, syn_file)
        mask_path = os.path.join(mask_dir, mask_file)

        if not os.path.exists(mask_path):
            print(f"  ✗ Maschera non trovata: {mask_path}")
            continue

        print(f"\nProcesso: {syn_file}")
        results, discarded = process_synthetic_image(syn_path, mask_path, model)
        total_discarded += discarded

        for bee_crop, varroa_crop, is_anomalous, base_name, bee_idx in results:
            bee_id = f"{base_name}_bee_{bee_idx:04d}"

            if is_anomalous:
                # Salva nella cartella anomalous + maschera
                img_path = os.path.join(OUTPUT_ANOMALOUS, f"{bee_id}.png")
                msk_path = os.path.join(OUTPUT_MASKS, f"{bee_id}_mask.png")
                cv2.imwrite(img_path, bee_crop)
                cv2.imwrite(msk_path, varroa_crop)
                csv_rows.append([bee_id, 1, img_path, msk_path])
                total_anomalous += 1
            else:
                # Salva nella cartella normal (nessuna maschera necessaria)
                img_path = os.path.join(OUTPUT_NORMAL, f"{bee_id}.png")
                cv2.imwrite(img_path, bee_crop)
                csv_rows.append([bee_id, 0, img_path, ""])
                total_normal += 1

    # Salva il CSV con tutti i label
    csv_path = os.path.join(OUTPUT_BASE, "labels.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["bee_id", "label", "image_path", "mask_path"])
        writer.writerows(csv_rows)

    print(f"\n{'='*60}")
    print(f"✅ Estrazione completata!")
    print(f"   Api normali:  {total_normal}")
    print(f"   Api anomale:  {total_anomalous}")
    print(f"   Api scartate (pulizia): {total_discarded}")
    print(f"   CSV labels:   {csv_path}")
    print(f"   Output dir:   {OUTPUT_BASE}")
    print(f"{'='*60}")
    print(f"\nStruttura output:")
    print(f"  {OUTPUT_BASE}/")
    print(f"    normal/        → {total_normal} api senza varroa")
    print(f"    anomalous/     → {total_anomalous} api con varroa")
    print(f"    masks/         → maschere GT per le api anomale")
    print(f"    labels.csv     → bee_id, label (0/1), paths")


if __name__ == "__main__":
    main()
