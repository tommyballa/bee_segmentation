"""
Estrae le singole api dalle immagini sintetiche con varroa e genera
la maschera di anomalia per-ape per le metriche di anomaly detection.

Pipeline:
1. Carica l'immagine sintetica e la GT mask della varroa (generati da varroa_generator.py)
2. Usa le label YOLO segmentation originali (poligoni delle api) per croppare ogni ape
3. Per ogni crop, incrocia la GT mask varroa con il poligono dell'ape
4. Salva:
   - Il crop dell'ape (224x224, sfondo nero, come extract_anomaly_dataset.py)
   - La maschera anomalia per-ape (224x224, bianco=varroa, nero=normale)
   - Un label (0=normale, 1=anomala) in un file CSV

Output:
  anomaly_dataset_synthetic/
    normal/          → api senza varroa (solo immagine, maschera tutta nera)
    anomalous/       → api con varroa (immagine + maschera)
    masks/           → maschere GT di anomalia per ogni ape anomala
    labels.csv       → bee_id, label (0/1), image_path, mask_path
"""

import os
import cv2
import csv
import numpy as np


# --- Configurazione ---
# Cartella con le immagini sintetiche e maschere generate da varroa_generator.py
SYNTHETIC_DIR = "/home/tommaso_ballarin/bee_segmentation/bee_segmentation/varroa_synthetic_image"

# Label YOLO segmentation originali (poligoni delle api)
LABELS_DIR = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/train/labels"

# Output
OUTPUT_BASE = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/anomaly_dataset_synthetic"
OUTPUT_NORMAL = os.path.join(OUTPUT_BASE, "normal")
OUTPUT_ANOMALOUS = os.path.join(OUTPUT_BASE, "anomalous")
OUTPUT_MASKS = os.path.join(OUTPUT_BASE, "masks")

# Dimensione finale dei crop (come PatchCore / MVTec)
CROP_SIZE = 224

# Padding attorno alla bounding box dell'ape
PADDING = 10

# Soglia minima di pixel varroa per considerare l'ape anomala
# (evita falsi positivi da overlap di 1-2 pixel al bordo)
MIN_VARROA_PIXELS = 10


def parse_yolo_segmentation_labels(label_path, img_w, img_h):
    """
    Legge le label YOLO segmentation: ogni riga è
       class_id x1 y1 x2 y2 ... xN yN
    dove le coordinate sono normalizzate [0,1].
    """
    detections = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue

            coords = list(map(float, parts[1:]))
            if len(coords) % 2 != 0:
                continue

            points = []
            for i in range(0, len(coords), 2):
                px = coords[i] * img_w
                py = coords[i + 1] * img_h
                points.append([px, py])

            polygon = np.array(points, dtype=np.float32)

            x_min, y_min = polygon.min(axis=0)
            x_max, y_max = polygon.max(axis=0)
            bbox = (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))

            detections.append({
                "polygon": polygon,
                "bbox": bbox,
            })

    return detections


def crop_and_square(image, mask, polygon_pts, bbox, crop_size=224, padding=10):
    """
    Croppa l'ape dall'immagine e dalla maschera varroa.

    1. Crea una maschera del poligono dell'ape
    2. Applica la maschera all'immagine (sfondo nero)
    3. Incrocia la maschera varroa con il poligono (solo varroa DENTRO l'ape)
    4. Fa il crop quadrato centrato e ridimensiona a crop_size x crop_size

    Returns:
        bee_crop: immagine dell'ape (crop_size x crop_size x 3)
        varroa_crop: maschera varroa per questa ape (crop_size x crop_size, 0 o 255)
        num_varroa_px: numero di pixel varroa dentro questa ape
    """
    h, w = image.shape[:2]
    bx, by, bw, bh = bbox

    # Crea maschera del poligono dell'ape (full size)
    bee_mask = np.zeros((h, w), dtype=np.uint8)
    pts = polygon_pts.astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(bee_mask, [pts], 255)

    # Incrocia maschera varroa con maschera ape → varroa solo dentro l'ape
    varroa_in_bee = cv2.bitwise_and(mask, bee_mask)

    # Conta pixel varroa dentro questa ape
    num_varroa_px = cv2.countNonZero(varroa_in_bee)

    # Bounding box con padding
    x1 = max(0, bx - padding)
    y1 = max(0, by - padding)
    x2 = min(w, bx + bw + padding)
    y2 = min(h, by + bh + padding)

    # Crop immagine con maschera ape (sfondo nero)
    bee_masked = cv2.bitwise_and(image, image, mask=bee_mask)
    img_crop = bee_masked[y1:y2, x1:x2].copy()
    varroa_crop = varroa_in_bee[y1:y2, x1:x2].copy()

    # Rendi quadrato e centrato
    crop_h, crop_w = img_crop.shape[:2]
    sq_size = max(crop_w, crop_h)

    square_img = np.zeros((sq_size, sq_size, 3), dtype=np.uint8)
    square_mask = np.zeros((sq_size, sq_size), dtype=np.uint8)

    off_x = (sq_size - crop_w) // 2
    off_y = (sq_size - crop_h) // 2

    square_img[off_y:off_y + crop_h, off_x:off_x + crop_w] = img_crop
    square_mask[off_y:off_y + crop_h, off_x:off_x + crop_w] = varroa_crop

    # Ridimensiona a dimensione finale
    bee_final = cv2.resize(square_img, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
    mask_final = cv2.resize(square_mask, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)

    return bee_final, mask_final, num_varroa_px


def process_synthetic_image(syn_img_path, mask_path, label_path):
    """
    Processa una singola immagine sintetica: estrae tutte le api e le classifica
    come normali o anomale in base alla sovrapposizione con la maschera varroa.

    Returns:
        list of (bee_crop, varroa_mask_crop, is_anomalous, source_name, bee_idx)
    """
    syn_img = cv2.imread(syn_img_path)
    varroa_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if syn_img is None:
        print(f"  ✗ Impossibile caricare: {syn_img_path}")
        return []
    if varroa_mask is None:
        print(f"  ✗ Impossibile caricare maschera: {mask_path}")
        return []

    H, W = syn_img.shape[:2]

    # Binarizza la maschera (potrebbe avere artefatti JPEG)
    _, varroa_mask = cv2.threshold(varroa_mask, 128, 255, cv2.THRESH_BINARY)

    # Leggi i poligoni delle api
    detections = parse_yolo_segmentation_labels(label_path, W, H)

    base_name = os.path.splitext(os.path.basename(syn_img_path))[0]
    results = []

    for i, det in enumerate(detections):
        bee_crop, varroa_crop, n_px = crop_and_square(
            syn_img, varroa_mask, det["polygon"], det["bbox"],
            crop_size=CROP_SIZE, padding=PADDING
        )

        is_anomalous = n_px >= MIN_VARROA_PIXELS
        results.append((bee_crop, varroa_crop, is_anomalous, base_name, i))

    return results


def main():
    # Crea le cartelle di output
    for d in [OUTPUT_NORMAL, OUTPUT_ANOMALOUS, OUTPUT_MASKS]:
        os.makedirs(d, exist_ok=True)

    # Trova tutte le coppie (immagine sintetica, maschera GT)
    syn_files = sorted([
        f for f in os.listdir(SYNTHETIC_DIR)
        if f.endswith(".jpg") and "_syn_" in f
    ])

    if not syn_files:
        print(f"Nessuna immagine sintetica trovata in {SYNTHETIC_DIR}")
        return

    csv_rows = []
    total_normal = 0
    total_anomalous = 0

    for syn_file in syn_files:
        # Derive mask filename: DSC_4914_syn_0.jpg → DSC_4914_mask_0.png
        mask_file = syn_file.replace("_syn_", "_mask_").replace(".jpg", ".png")
        syn_path = os.path.join(SYNTHETIC_DIR, syn_file)
        mask_path = os.path.join(SYNTHETIC_DIR, mask_file)

        if not os.path.exists(mask_path):
            print(f"  ✗ Maschera non trovata: {mask_path}")
            continue

        # Derive label file: DSC_4914_syn_0.jpg → DSC_4914.txt
        # Estraiamo il nome originale prima di _syn_
        original_name = syn_file.split("_syn_")[0]
        label_file = original_name + ".txt"
        label_path = os.path.join(LABELS_DIR, label_file)

        if not os.path.exists(label_path):
            print(f"  ✗ Label non trovata: {label_path}")
            continue

        print(f"\nProcesso: {syn_file}")
        results = process_synthetic_image(syn_path, mask_path, label_path)

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
