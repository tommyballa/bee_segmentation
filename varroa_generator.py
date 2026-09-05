import cv2
import numpy as np
import random
import os

# --- Configurazione ---
PATH_FAVO = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/train/images/DSC_4914.JPG"
PATH_VARROA = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/varroa_alpha.png"
PATH_LABELS = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/train/labels/DSC_4914.txt"
OUTPUT_DIR = "/home/tommaso_ballarin/bee_segmentation/bee_segmentation/varroa_synthetic_image"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Rapporto dimensionale Varroa/Ape: la Varroa destructor è ~1.6mm di lunghezza,
# il torace dell'ape è ~4-5mm, ma in foto il corpo intero occupa molto di più.
# Usiamo il lato più corto della bounding box del poligono come riferimento per il torace.
VARROA_BEE_RATIO_MIN = 0.18
VARROA_BEE_RATIO_MAX = 0.26

# Quante varroa inserire per immagine
NUM_VARROA_MIN = 1
NUM_VARROA_MAX = 3

# Feathering sul bordo della maschera (in pixel) per evitare ritaglio netto
FEATHER_RADIUS = 3


def load_images():
    """Carica l'immagine del favo e l'immagine RGBA della varroa."""
    favo = cv2.imread(PATH_FAVO)
    # Carichiamo la varroa mantenendo il canale Alpha (UNCHANGED)
    varroa = cv2.imread(PATH_VARROA, cv2.IMREAD_UNCHANGED)

    if favo is None:
        raise FileNotFoundError(f"Impossibile caricare l'immagine del favo: {PATH_FAVO}")
    if varroa is None:
        raise FileNotFoundError(f"Impossibile caricare l'immagine della varroa: {PATH_VARROA}")
    if varroa.shape[2] != 4:
        raise ValueError("L'immagine della Varroa deve avere un canale Alpha (PNG 32-bit).")

    return favo, varroa


def parse_yolo_segmentation_labels(label_path, img_w, img_h):
    """
    Legge le label YOLO segmentation: ogni riga è
       class_id x1 y1 x2 y2 ... xN yN
    dove le coordinate sono normalizzate [0,1].

    Restituisce una lista di dizionari con:
      - 'polygon': np.array di punti (N,2) in pixel
      - 'bbox': (x, y, w, h) bounding box in pixel
      - 'centroid': (cx, cy) centro del poligono in pixel
    """
    detections = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:  # class_id + almeno 3 punti (6 valori)
                continue

            # Primo valore è il class_id, il resto sono coppie x,y
            coords = list(map(float, parts[1:]))
            if len(coords) % 2 != 0:
                continue  # Numero dispari di coordinate, skip

            points = []
            for i in range(0, len(coords), 2):
                px = coords[i] * img_w
                py = coords[i + 1] * img_h
                points.append([px, py])

            polygon = np.array(points, dtype=np.float32)

            # Bounding box dal poligono
            x_min, y_min = polygon.min(axis=0)
            x_max, y_max = polygon.max(axis=0)
            bbox = (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))

            # Centroide
            cx = polygon[:, 0].mean()
            cy = polygon[:, 1].mean()

            detections.append({
                "polygon": polygon,
                "bbox": bbox,
                "centroid": (int(cx), int(cy)),
            })

    return detections


def rotate_image_rgba(image_rgba, angle):
    """
    Ruota un'immagine RGBA di un angolo arbitrario, espandendo il bounding box
    per non tagliare i bordi. I pixel aggiunti dall'espansione sono trasparenti.
    """
    h, w = image_rgba.shape[:2]
    cx, cy = w / 2, h / 2

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    # Calcola le nuove dimensioni per contenere l'immagine ruotata
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)

    # Aggiusta la matrice di trasformazione per il nuovo centro
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    rotated = cv2.warpAffine(
        image_rgba, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )
    return rotated


def adapt_color_to_region(varroa_rgb, target_region, alpha_mask):
    """
    Adatta leggermente il colore e la luminosità della varroa alla regione target.
    Lavora in spazio LAB. Usa solo i pixel opachi della varroa e dell'ape.
    Il blend è molto leggero per mantenere il colore rosso-bruno caratteristico.
    """
    if target_region.size == 0 or varroa_rgb.size == 0:
        return varroa_rgb

    # Maschera dei pixel opachi della varroa
    opaque = alpha_mask > 128

    if opaque.sum() < 10:
        return varroa_rgb

    # Converti in LAB
    varroa_lab = cv2.cvtColor(varroa_rgb, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target_region, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Calcola statistiche solo sui pixel opachi della varroa
    v_l = varroa_lab[:, :, 0][opaque]
    t_l = target_lab[:, :, 0]

    v_mean_l = v_l.mean()
    t_mean_l = t_l.mean()

    # Leggero adattamento della luminosità (20%) — manteniamo il colore scuro della varroa
    blend_factor = 0.20
    l_shift = (t_mean_l - v_mean_l) * blend_factor

    # Applica lo shift solo ai pixel opachi
    varroa_lab[:, :, 0] = np.where(
        opaque,
        np.clip(varroa_lab[:, :, 0] + l_shift, 0, 255),
        varroa_lab[:, :, 0]
    )

    # Shift colore minimo (10%) per coerenza cromatica
    for ch in [1, 2]:
        v_ch = varroa_lab[:, :, ch][opaque]
        t_ch = target_lab[:, :, ch]
        shift = (t_ch.mean() - v_ch.mean()) * 0.10
        varroa_lab[:, :, ch] = np.where(
            opaque,
            np.clip(varroa_lab[:, :, ch] + shift, 0, 255),
            varroa_lab[:, :, ch]
        )

    adapted = cv2.cvtColor(varroa_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return adapted


def feather_alpha(alpha, radius=2):
    """
    Applica un leggero feathering (sfumatura) solo sul bordo della maschera alpha.
    Questo evita il bordo netto tipo "ritaglio" senza rendere la varroa trasparente.
    Il corpo rimane completamente opaco, solo i 2-3 pixel del bordo vengono sfumati.
    """
    if radius <= 0:
        return alpha

    # Erodi la maschera per trovare il bordo interno
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    eroded = cv2.erode(alpha, kernel, iterations=1)

    # Il bordo è la differenza tra originale e eroso
    border = alpha.astype(np.float32) - eroded.astype(np.float32)

    # Sfuma leggermente il bordo con un blur
    blur_size = radius * 2 + 1
    blurred_alpha = cv2.GaussianBlur(alpha.astype(np.float32), (blur_size, blur_size), 0)

    # Combina: dove c'è il bordo usa il valore sfumato, altrimenti l'originale (opaco pieno)
    result = np.where(border > 0, blurred_alpha, alpha.astype(np.float32))
    return np.clip(result, 0, 255).astype(np.uint8)


def alpha_composite(target, src_rgb, alpha_float, x, y):
    """
    Compositing con alpha blending. Alpha è un float array [0,1].
    Il risultato è opaco dove alpha=1 (corpo varroa) e trasparente solo sui bordi sfumati.
    """
    h_s, w_s = src_rgb.shape[:2]
    h_t, w_t = target.shape[:2]

    # Clipping ai bordi dell'immagine target
    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + w_s, w_t)
    y2 = min(y + h_s, h_t)

    if x1 >= x2 or y1 >= y2:
        return target

    # Offset nella sorgente
    sx1 = x1 - x
    sy1 = y1 - y
    sx2 = sx1 + (x2 - x1)
    sy2 = sy1 + (y2 - y1)

    alpha = alpha_float[sy1:sy2, sx1:sx2, np.newaxis]  # (h, w, 1)
    rgb = src_rgb[sy1:sy2, sx1:sx2].astype(np.float32)

    roi = target[y1:y2, x1:x2].astype(np.float32)
    blended = roi * (1.0 - alpha) + rgb * alpha
    target[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    return target


def add_subtle_shadow(target, mask, offset=(2, 3), blur_size=7, opacity=0.3):
    """
    Aggiunge un'ombra sottile sotto la varroa per dare profondità.
    """
    h, w = target.shape[:2]

    # Sposta la maschera per creare l'ombra
    dx, dy = offset
    M_shift = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(mask.astype(np.float32), M_shift, (w, h))

    # Sfuma l'ombra
    shadow_mask = cv2.GaussianBlur(shifted, (blur_size, blur_size), 0)
    shadow_mask = (shadow_mask / 255.0) * opacity

    # Applica l'ombra scurendo il target
    for c in range(3):
        target[:, :, c] = np.clip(
            target[:, :, c].astype(np.float32) * (1.0 - shadow_mask), 0, 255
        ).astype(np.uint8)

    return target


def add_contact_shadow(target, varroa_mask, insert_x, insert_y, ring_width=3, opacity=0.18):
    """
    Aggiunge un alone scuro molto sottile attorno al bordo della varroa,
    simulando il contatto fisico del parassita con il corpo dell'ape.
    Molto leggero per non creare un bordo visibile.
    """
    H, W = target.shape[:2]
    rv_h, rv_w = varroa_mask.shape[:2]

    # Dilata la maschera per creare l'anello esterno
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_width * 2 + 1, ring_width * 2 + 1))
    dilated = cv2.dilate(varroa_mask, kernel, iterations=1)

    # L'anello è la differenza tra dilatata e originale
    ring = dilated.astype(np.float32) - varroa_mask.astype(np.float32)

    # Blur più ampio per sfumare gradualmente nel corpo dell'ape
    blur_k = max(ring_width * 4 + 1, 5)
    if blur_k % 2 == 0:
        blur_k += 1
    ring = cv2.GaussianBlur(ring, (blur_k, blur_k), 0)
    ring = ring / 255.0 * opacity

    # Proietta sulla dimensione piena dell'immagine
    sy1 = max(insert_y, 0)
    sy2 = min(insert_y + rv_h, H)
    sx1 = max(insert_x, 0)
    sx2 = min(insert_x + rv_w, W)
    vy1 = sy1 - insert_y
    vy2 = vy1 + (sy2 - sy1)
    vx1 = sx1 - insert_x
    vx2 = vx1 + (sx2 - sx1)

    ring_roi = ring[vy1:vy2, vx1:vx2]

    # Scurisci il target nella zona dell'anello di contatto
    for c in range(3):
        target[sy1:sy2, sx1:sx2, c] = np.clip(
            target[sy1:sy2, sx1:sx2, c].astype(np.float32) * (1.0 - ring_roi),
            0, 255
        ).astype(np.uint8)

    return target


def find_body_position(favo, det, rv_w, rv_h, num_candidates=12):
    """
    Trova la posizione migliore sul corpo dell'ape (non sulle ali).
    Campiona più posizioni candidate dentro il poligono e sceglie quella
    con la saturazione più alta (il corpo dell'ape è più colorato/scuro,
    le ali sono chiare e poco sature).
    """
    bx, by, bw, bh = det["bbox"]
    polygon = det["polygon"].astype(np.int32)
    H, W = favo.shape[:2]

    margin_w = int(bw * 0.20)
    margin_h = int(bh * 0.20)

    x_min = bx + margin_w
    y_min = by + margin_h
    x_max = bx + bw - margin_w - rv_w
    y_max = by + bh - margin_h - rv_h

    if x_max <= x_min or y_max <= y_min:
        # Fallback: usa il centroide
        cx, cy = det["centroid"]
        return int(cx - rv_w / 2), int(cy - rv_h / 2)

    # Converti la ROI in HSV per misurare la saturazione
    hsv = cv2.cvtColor(favo, cv2.COLOR_BGR2HSV)

    best_score = -1
    best_x, best_y = x_min, y_min

    for _ in range(num_candidates):
        cx = random.randint(x_min, x_max)
        cy = random.randint(y_min, y_max)

        # Il centro della varroa deve essere dentro il poligono
        center = (cx + rv_w // 2, cy + rv_h // 2)
        if cv2.pointPolygonTest(polygon, center, False) < 0:
            continue

        # Analizza la regione sotto la varroa
        ry1 = max(cy, 0)
        ry2 = min(cy + rv_h, H)
        rx1 = max(cx, 0)
        rx2 = min(cx + rv_w, W)

        if ry2 <= ry1 or rx2 <= rx1:
            continue

        roi_hsv = hsv[ry1:ry2, rx1:rx2]
        # Score = saturazione media + (255 - luminosità media)
        # Le zone del corpo dell'ape hanno alta saturazione e luminosità media
        # Le ali sono chiare (alta V) e poco sature (bassa S)
        sat_mean = roi_hsv[:, :, 1].mean()
        val_mean = roi_hsv[:, :, 2].mean()
        score = sat_mean + (255 - val_mean) * 0.5

        if score > best_score:
            best_score = score
            best_x, best_y = cx, cy

    # Clamp ai bordi immagine
    best_x = max(0, min(best_x, W - rv_w))
    best_y = max(0, min(best_y, H - rv_h))

    return best_x, best_y


def generate_synthetic_anomaly(favo, varroa_src, detections):
    """
    Pipeline principale per l'inserimento realistico della varroa sulle api.
    Usa alpha compositing diretto (no Poisson blending) per evitare artefatti rettangolari.
    """
    H, W = favo.shape[:2]
    synthetic_img = favo.copy()
    gt_mask = np.zeros((H, W), dtype=np.uint8)

    # Scegli quante Varroa inserire
    num_to_insert = random.randint(NUM_VARROA_MIN, NUM_VARROA_MAX)

    # Filtriamo le api troppo piccole (area bbox < 40x40 px)
    valid_indices = [
        i for i, det in enumerate(detections)
        if det["bbox"][2] > 40 and det["bbox"][3] > 40
    ]
    if not valid_indices:
        return None, None

    selected_indices = random.sample(valid_indices, min(num_to_insert, len(valid_indices)))

    inserted_count = 0
    for idx in selected_indices:
        det = detections[idx]
        bx, by, bw, bh = det["bbox"]

        # --- 1. Dimensionamento della Varroa ---
        reference_dim = min(bw, bh)
        ratio = random.uniform(VARROA_BEE_RATIO_MIN, VARROA_BEE_RATIO_MAX)
        target_v_w = int(reference_dim * ratio)

        # Manteniamo l'aspect ratio della Varroa originale
        v_h_orig, v_w_orig = varroa_src.shape[:2]
        aspect_ratio = v_h_orig / v_w_orig
        target_v_h = int(target_v_w * aspect_ratio)

        if target_v_w < 8 or target_v_h < 8:
            continue

        # Ridimensioniamo l'intera immagine RGBA della varroa
        varroa_resized = cv2.resize(varroa_src, (target_v_w, target_v_h), interpolation=cv2.INTER_AREA)

        # --- 2. Rotazione casuale con espansione ---
        angle = random.uniform(0, 360)
        varroa_rotated = rotate_image_rgba(varroa_resized, angle)

        # Estraiamo i canali dopo la rotazione
        v_rgb = varroa_rotated[:, :, :3]
        v_alpha_raw = varroa_rotated[:, :, 3]
        rv_h, rv_w = v_rgb.shape[:2]

        # Crea maschera binaria netta per la GT mask
        _, v_mask_binary = cv2.threshold(v_alpha_raw, 128, 255, cv2.THRESH_BINARY)

        # Crea alpha sfumato solo sul bordo per il compositing (corpo opaco, bordi soft)
        v_alpha_feathered = feather_alpha(v_mask_binary, radius=FEATHER_RADIUS)
        v_alpha_float = v_alpha_feathered.astype(np.float32) / 255.0

        # --- 3. Posizionamento intelligente sul corpo dell'ape ---
        # Cerca la posizione con più saturazione (corpo) invece che sulle ali
        insert_x, insert_y = find_body_position(synthetic_img, det, rv_w, rv_h, num_candidates=15)

        # --- 4. Adattamento colore leggero ---
        roi_y1 = max(insert_y, 0)
        roi_y2 = min(insert_y + rv_h, H)
        roi_x1 = max(insert_x, 0)
        roi_x2 = min(insert_x + rv_w, W)
        if roi_y2 <= roi_y1 or roi_x2 <= roi_x1:
            continue

        target_region = synthetic_img[roi_y1:roi_y2, roi_x1:roi_x2]
        v_rgb_adapted = adapt_color_to_region(v_rgb, target_region, v_mask_binary)

        # --- 5. Alpha compositing diretto (no rettangolo, no trasparenza eccessiva) ---
        synthetic_img = alpha_composite(synthetic_img, v_rgb_adapted, v_alpha_float, insert_x, insert_y)

        # --- 5b. Ombra di contatto (alone sottile attorno alla varroa) ---
        # Molto leggero per non creare un bordo artificiale
        contact_ring_w = max(2, int(min(rv_w, rv_h) * 0.08))
        synthetic_img = add_contact_shadow(
            synthetic_img, v_mask_binary, insert_x, insert_y,
            ring_width=contact_ring_w, opacity=0.15
        )

        # --- 6. Ombra sottile spostata ---
        full_shadow_mask = np.zeros((H, W), dtype=np.uint8)
        sy1 = max(insert_y, 0)
        sy2 = min(insert_y + rv_h, H)
        sx1 = max(insert_x, 0)
        sx2 = min(insert_x + rv_w, W)
        vy1 = sy1 - insert_y
        vy2 = vy1 + (sy2 - sy1)
        vx1 = sx1 - insert_x
        vx2 = vx1 + (sx2 - sx1)
        full_shadow_mask[sy1:sy2, sx1:sx2] = v_mask_binary[vy1:vy2, vx1:vx2]

        shadow_offset = max(1, int(min(rv_w, rv_h) * 0.08))
        shadow_blur = max(3, int(min(rv_w, rv_h) * 0.20)) | 1
        synthetic_img = add_subtle_shadow(
            synthetic_img, full_shadow_mask,
            offset=(shadow_offset, shadow_offset),
            blur_size=shadow_blur,
            opacity=0.30
        )

        # --- 7. Aggiorna la Ground Truth Mask ---
        gt_mask[sy1:sy2, sx1:sx2] = cv2.bitwise_or(
            gt_mask[sy1:sy2, sx1:sx2], v_mask_binary[vy1:vy2, vx1:vx2]
        )

        inserted_count += 1

    if inserted_count == 0:
        return None, None

    _, gt_mask_final = cv2.threshold(gt_mask, 1, 1, cv2.THRESH_BINARY)

    return synthetic_img, gt_mask_final


# --- Main ---
if __name__ == "__main__":
    try:
        favo_img, varroa_alpha = load_images()
        H, W = favo_img.shape[:2]
        print(f"Immagine favo: {W}x{H}")
        print(f"Immagine varroa: {varroa_alpha.shape[1]}x{varroa_alpha.shape[0]}, canali: {varroa_alpha.shape[2]}")

        # 1. Leggi le detection YOLO segmentation (poligoni)
        detections = parse_yolo_segmentation_labels(PATH_LABELS, W, H)
        print(f"Rilevate {len(detections)} api con poligoni di segmentazione.")

        if not detections:
            print("Nessuna ape rilevata. Impossibile generare anomalie.")
        else:
            num_variants = 5
            for i in range(num_variants):
                print(f"\nGenerazione immagine sintetica {i + 1}/{num_variants}...")
                syn_img, gt_mask = generate_synthetic_anomaly(favo_img, varroa_alpha, detections)

                if syn_img is not None:
                    base_name = os.path.splitext(os.path.basename(PATH_FAVO))[0]
                    out_img_path = os.path.join(OUTPUT_DIR, f"{base_name}_syn_{i}.jpg")
                    out_mask_path = os.path.join(OUTPUT_DIR, f"{base_name}_mask_{i}.png")
                    cv2.imwrite(out_img_path, syn_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    cv2.imwrite(out_mask_path, gt_mask * 255)
                    print(f"  ✓ Salvata: {out_img_path}")
                else:
                    print(f"  ✗ Fallito inserimento su questa variante.")

        print(f"\nPipeline completata. Controlla la cartella: {OUTPUT_DIR}")

    except Exception as e:
        import traceback
        print(f"Errore fatale: {e}")
        traceback.print_exc()