import cv2
import numpy as np
import random
import os

# --- Configurazione ---
IMAGES_DIR = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/test/images"
LABELS_DIR = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/test/labels"
VARROA_PATHS = [
    "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/varroa_alpha.png",
    "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/varroa_alpha2.png",
    "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/varroa_alpha3.png",
]
OUTPUT_DIR = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/test/synthetic_varroa_output_v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)
import glob

# Rapporto dimensionale Varroa/Ape: la Varroa destructor è ~1.6mm di lunghezza,
# il torace dell'ape è ~4-5mm, ma in foto il corpo intero occupa molto di più.
# Usiamo il lato più corto della bounding box del poligono come riferimento per il torace.
VARROA_BEE_RATIO_MIN = 0.20
VARROA_BEE_RATIO_MAX = 0.24

# Quante api infettare per immagine
NUM_BEES_TO_INFECT_MIN = 20
NUM_BEES_TO_INFECT_MAX =40

# Quante varroa inserire per ogni ape selezionata (da 1 a 3)
VARROA_PER_BEE_MIN = 1
VARROA_PER_BEE_MAX = 2

# Feathering sul bordo della maschera (in pixel) per evitare ritaglio netto
FEATHER_RADIUS = 5


def load_images(path_favo):
    """Carica l'immagine del favo e tutte le immagini RGBA delle varroa.

    Le varroa vengono normalizzate a una dimensione di riferimento comune
    (la larghezza mediana) in modo che le differenze di risoluzione tra i
    vari PNG non influenzino il dimensionamento finale sulle api.

    Restituisce: (favo, lista_varroa)
    """
    favo = cv2.imread(path_favo)
    if favo is None:
        raise FileNotFoundError(f"Impossibile caricare l'immagine del favo: {path_favo}")

    varroa_list = []
    widths = []
    for vp in VARROA_PATHS:
        v = cv2.imread(vp, cv2.IMREAD_UNCHANGED)
        if v is None:
            raise FileNotFoundError(f"Impossibile caricare l'immagine della varroa: {vp}")
        if v.shape[2] != 4:
            raise ValueError(f"L'immagine della Varroa deve avere un canale Alpha (PNG 32-bit): {vp}")
        varroa_list.append(v)
        widths.append(v.shape[1])

    # Normalizza tutte le varroa a una larghezza di riferimento comune
    # (la mediana delle larghezze originali) mantenendo l'aspect ratio
    ref_width = int(np.median(widths))
    normalized = []
    for v in varroa_list:
        h_orig, w_orig = v.shape[:2]
        if w_orig != ref_width:
            scale = ref_width / w_orig
            new_h = int(h_orig * scale)
            # Usa INTER_AREA per downscale, INTER_CUBIC per upscale
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
            v = cv2.resize(v, (ref_width, new_h), interpolation=interp)
        normalized.append(v)

    return favo, normalized


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
    Adatta colore, luminosità, contrasto e texture della varroa alla foto dell'ape.
    TECNICHE AVANZATE:
    1. Match del contrasto (deviazione standard) oltre che della media luminosa
    2. Blend della crominanza aumentato per assorbire i riflessi dell'ambiente
    3. Aggiunta di grana fotografica (noise) per simulare l'ISO della fotocamera
    """
    if target_region.size == 0 or varroa_rgb.size == 0:
        return varroa_rgb

    # Maschera dei pixel opachi della varroa
    opaque = alpha_mask > 128

    if opaque.sum() < 10:
        return varroa_rgb
        
    # Applica un micro-blur iniziale per togliere la nitidezza artificiale da "ritaglio perfetto"
    varroa_rgb_soft = cv2.GaussianBlur(varroa_rgb, (3, 3), 0)

    # Converti in spazio colore percettivo LAB
    varroa_lab = cv2.cvtColor(varroa_rgb_soft, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target_region, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Estrai i canali di luminosità (L)
    v_l = varroa_lab[:, :, 0][opaque]
    t_l = target_lab[:, :, 0]

    v_mean_l, v_std_l = v_l.mean(), v_l.std() + 1e-5
    t_mean_l, t_std_l = t_l.mean(), t_l.std() + 1e-5

    # 1. Match del Contrasto e Luminosità
    # Manteniamo più il contrasto originale della varroa (70%)
    target_contrast = (v_std_l * 0.70 + t_std_l * 0.30) / v_std_l
    
    # Adattamento luminosità ridotto per non sbiadirla troppo (35%)
    blend_factor = 0.35 
    l_shift = (t_mean_l - v_mean_l) * blend_factor

    # Applica trasformazione al canale L
    varroa_lab[:, :, 0] = np.where(
        opaque,
        np.clip((varroa_lab[:, :, 0] - v_mean_l) * target_contrast + v_mean_l + l_shift, 0, 255),
        varroa_lab[:, :, 0]
    )

    # 2. Match della Crominanza
    for ch in [1, 2]:
        v_ch = varroa_lab[:, :, ch][opaque]
        t_ch = target_lab[:, :, ch]
        # Shift ridotto (20%) per preservare il colore tipico del parassita
        shift = (t_ch.mean() - v_ch.mean()) * 0.20  
        varroa_lab[:, :, ch] = np.where(
            opaque,
            np.clip(varroa_lab[:, :, ch] + shift, 0, 255),
            varroa_lab[:, :, ch]
        )

    # Converti di nuovo in BGR
    adapted = cv2.cvtColor(varroa_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    
    # 3. Aggiunta Grana Fotografica (Camera Noise)
    # Genera rumore gaussiano per simulare la grana del sensore e rompere le campiture piatte
    noise = np.random.normal(0, 8, adapted.shape).astype(np.float32)  # Dev std = 8
    adapted_noisy = np.where(
        opaque[..., None],  # Applica il rumore solo dove c'è la varroa
        np.clip(adapted.astype(np.float32) + noise, 0, 255),
        adapted
    ).astype(np.uint8)

    return adapted_noisy


def feather_alpha(alpha, radius=5):
    """
    Applica un feathering (sfumatura) naturale sul bordo della maschera alpha.
    """
    if radius <= 0:
        return alpha

    # Riduciamo molto di più il nucleo solido (in base al raggio)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    inner_core = cv2.erode(alpha, kernel, iterations=max(1, radius - 1))
    
    # Aumentiamo enormemente la dimensione del blur per fare un gradiente
    # che parta da dentro la varroa e finisca lentamente all'esterno
    blur_size = radius * 4 + 1
    blurred_alpha = cv2.GaussianBlur(alpha.astype(np.float32), (blur_size, blur_size), 0)
    
    # Per non renderla fantasma, il nucleo centrale rimane perfettamente solido (255)
    result = np.where(inner_core > 0, 255.0, blurred_alpha)
    
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


def add_subtle_shadow(target, varroa_mask, insert_x, insert_y, offset=(2, 3), blur_size=7, opacity=0.3):
    """
    Aggiunge un'ombra sottile sotto la varroa per dare profondità.
    OTTIMIZZATA: calcola il blur solo sul rettangolo dell'ombra,
    evitando di sfocare milioni di pixel inutili.
    """
    H, W = target.shape[:2]
    rv_h, rv_w = varroa_mask.shape[:2]
    dx, dy = offset

    # Calcoliamo lo spazio extra per non tagliare l'ombra (blur radius + offset)
    pad = blur_size * 2 + max(dx, dy)
    
    # Creiamo un piccolo canvas temporaneo grande quanto la varroa + padding
    mini_canvas_h = rv_h + pad * 2
    mini_canvas_w = rv_w + pad * 2
    mini_canvas = np.zeros((mini_canvas_h, mini_canvas_w), dtype=np.uint8)
    
    # Incolliamo la maschera della varroa al centro del canvas temporaneo
    mini_canvas[pad:pad+rv_h, pad:pad+rv_w] = varroa_mask

    # Spostiamo l'ombra (offset) sul canvas temporaneo
    M_shift = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(mini_canvas.astype(np.float32), M_shift, (mini_canvas_w, mini_canvas_h))

    # Sfocatura veloce SOLO sulla piccola porzione
    shadow_mask = cv2.GaussianBlur(shifted, (blur_size, blur_size), 0)
    shadow_mask = (shadow_mask / 255.0) * opacity

    # Calcoliamo dove incollare l'ombra sull'immagine globale
    sy1 = insert_y - pad
    sy2 = insert_y - pad + mini_canvas_h
    sx1 = insert_x - pad
    sx2 = insert_x - pad + mini_canvas_w

    # Clamp alle dimensioni dell'immagine target
    tgt_y1 = max(0, sy1)
    tgt_y2 = min(H, sy2)
    tgt_x1 = max(0, sx1)
    tgt_x2 = min(W, sx2)

    if tgt_y2 <= tgt_y1 or tgt_x2 <= tgt_x1:
        return target

    # Mappiamo le coordinate sul canvas temporaneo dell'ombra
    src_y1 = tgt_y1 - sy1
    src_y2 = src_y1 + (tgt_y2 - tgt_y1)
    src_x1 = tgt_x1 - sx1
    src_x2 = src_x1 + (tgt_x2 - tgt_x1)

    shadow_roi = shadow_mask[src_y1:src_y2, src_x1:src_x2]

    # Applica l'ombra scurendo il target locale
    for c in range(3):
        target[tgt_y1:tgt_y2, tgt_x1:tgt_x2, c] = np.clip(
            target[tgt_y1:tgt_y2, tgt_x1:tgt_x2, c].astype(np.float32) * (1.0 - shadow_roi), 0, 255
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


def find_body_position(favo, det, rv_w, rv_h, gt_mask=None, v_mask_binary=None, num_candidates=30):
    """
    Trova la posizione migliore sul corpo dell'ape (non sulle ali).
    Campiona più posizioni candidate dentro il poligono e sceglie quella
    con la saturazione più alta (il corpo dell'ape è più colorato/scuro,
    le ali sono chiare e poco sature).
    Scarta i candidati che si sovrappongono a varroa già inserite.
    """
    bx, by, bw, bh = det["bbox"]
    polygon = det["polygon"].astype(np.int32)
    H, W = favo.shape[:2]

    # Margini ridotti (25%) per lasciare spazio a più varroa sulla stessa ape
    margin_w = int(bw * 0.25)
    margin_h = int(bh * 0.25)

    x_min = bx + margin_w
    y_min = by + margin_h
    x_max = bx + bw - margin_w - rv_w
    y_max = by + bh - margin_h - rv_h

    if x_max <= x_min or y_max <= y_min:
        # Fallback: usa il centroide
        cx, cy = det["centroid"]
        return int(cx - rv_w / 2), int(cy - rv_h / 2)

    # OTTIMIZZAZIONE: Ritaglia l'area in cui si trova l'ape e converti solo quella in HSV.
    # Convertire 100x100 pixel è istantaneo rispetto ai 12 Megapixel dell'intera immagine.
    roi_y1, roi_y2 = max(0, by), min(H, by + bh)
    roi_x1, roi_x2 = max(0, bx), min(W, bx + bw)
    roi_bgr = favo[roi_y1:roi_y2, roi_x1:roi_x2]
    
    if roi_bgr.size == 0:
        return None, None
        
    roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

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

        # Check overlap
        if gt_mask is not None and v_mask_binary is not None:
            overlap_region = gt_mask[ry1:ry2, rx1:rx2]
            vy1_check = ry1 - cy
            vy2_check = vy1_check + (ry2 - ry1)
            vx1_check = rx1 - cx
            vx2_check = vx1_check + (rx2 - rx1)
            new_varroa_region = v_mask_binary[vy1_check:vy2_check, vx1_check:vx2_check]
            overlap_pixels = cv2.bitwise_and(overlap_region, new_varroa_region)
            if np.count_nonzero(overlap_pixels) > 0:
                continue # Sovrapposizione, scarta candidato

        # Trasforma le coordinate globali (cx, cy) in coordinate locali alla ROI
        local_rx1 = rx1 - roi_x1
        local_rx2 = rx2 - roi_x1
        local_ry1 = ry1 - roi_y1
        local_ry2 = ry2 - roi_y1

        # Leggi la patch in formato HSV ma solo dalla ROI appena creata
        local_hsv = roi_hsv[local_ry1:local_ry2, local_rx1:local_rx2]
        
        if local_hsv.size == 0:
            continue
            
        sat_mean = local_hsv[:, :, 1].mean()
        val_mean = local_hsv[:, :, 2].mean()
        
        # Score = saturazione media + (255 - luminosità media)
        score = sat_mean + (255 - val_mean) * 0.5

        if score > best_score:
            best_score = score
            best_x, best_y = cx, cy

    if best_score == -1:
        return None, None

    # Clamp ai bordi immagine
    best_x = max(0, min(best_x, W - rv_w))
    best_y = max(0, min(best_y, H - rv_h))

    return best_x, best_y

def generate_synthetic_anomaly(favo, varroa_sources, detections):
    """
    Pipeline principale per l'inserimento realistico della varroa sulle api.
    Usa alpha compositing diretto (no Poisson blending) per evitare artefatti rettangolari.

    varroa_sources: lista di immagini RGBA delle varroa (normalizzate alla stessa scala).
    Per ogni inserimento viene scelta casualmente una delle varroa disponibili.
    """
    H, W = favo.shape[:2]
    synthetic_img = favo.copy()
    gt_mask = np.zeros((H, W), dtype=np.uint8)

    # Scegli quante api infettare
    num_bees = random.randint(NUM_BEES_TO_INFECT_MIN, NUM_BEES_TO_INFECT_MAX)

    # --- FILTRO API CANDIDATE: solo api ben visibili, isolate e non ai bordi ---
    BORDER_MARGIN = 40  # pixel dal bordo immagine
    MIN_BEE_SIDE = 60   # lato minimo bbox (più grande = ape più visibile)
    MAX_OVERLAP_RATIO = 0.15  # massima sovrapposizione con altre api (15%)

    valid_indices = []
    bboxes = [det["bbox"] for det in detections]

    for i, det in enumerate(detections):
        bx, by, bw, bh = det["bbox"]

        # 1. Scarta api troppo piccole
        if bw < MIN_BEE_SIDE or bh < MIN_BEE_SIDE:
            continue

        # 2. Scarta api il cui bounding box tocca i bordi dell'immagine
        if bx < BORDER_MARGIN or by < BORDER_MARGIN:
            continue
        if (bx + bw) > (W - BORDER_MARGIN) or (by + bh) > (H - BORDER_MARGIN):
            continue

        # 3. OTTIMIZZAZIONE: Usa la matematica dei Bounding Box per l'intersezione
        # invece di matrici (immagini) da 12 Megapixel. Calcolo fulmineo.
        overlaps_with_others = False
        this_area = bw * bh
        for j, other_bbox in enumerate(bboxes):
            if i == j: continue
            
            ox, oy, ow, oh = other_bbox
            
            # Calcola l'intersezione dei rettangoli
            ix = max(bx, ox)
            iy = max(by, oy)
            iw = min(bx + bw, ox + ow) - ix
            ih = min(by + bh, oy + oh) - iy
            
            if iw > 0 and ih > 0:
                # Se l'area sovrapposta supera la soglia, scarta l'ape
                overlap_ratio = (iw * ih) / this_area
                if overlap_ratio > MAX_OVERLAP_RATIO:
                    overlaps_with_others = True
                    break

        if overlaps_with_others:
            continue

        valid_indices.append(i)

    if not valid_indices:
        return None, None

    selected_indices = random.sample(valid_indices, min(num_bees, len(valid_indices)))

    print(f"    Infezione di {len(selected_indices)} api su {len(valid_indices)} valide (filtrate da {len(detections)} totali)...")
    inserted_count = 0
    for idx in selected_indices:
        det = detections[idx]

        # Scegli quante varroa mettere su questa ape (da 1 a 3)
        varroa_count = random.randint(VARROA_PER_BEE_MIN, VARROA_PER_BEE_MAX)
        print(f"      Ape {idx}: tento di inserire {varroa_count} varroa...")

        for _v in range(varroa_count):
            bx, by, bw, bh = det["bbox"]

            # Scegli casualmente una delle varroa disponibili
            varroa_src = random.choice(varroa_sources)

            # --- 1. Dimensionamento della Varroa ---
            reference_dim = min(bw, bh)
            ratio = random.uniform(VARROA_BEE_RATIO_MIN, VARROA_BEE_RATIO_MAX)
            target_v_w = int(reference_dim * ratio)

            # Manteniamo l'aspect ratio della Varroa scelta
            v_h_orig, v_w_orig = varroa_src.shape[:2]
            aspect_ratio = v_h_orig / v_w_orig
            target_v_h = int(target_v_w * aspect_ratio)

            # Ridimensioniamo l'intera immagine RGBA della varroa
            varroa_resized = cv2.resize(varroa_src, (target_v_w, target_v_h), interpolation=cv2.INTER_AREA)

            # --- FIX "EFFETTO RIQUADRO" ---
            # Aggiungiamo un generoso margine trasparente attorno all'immagine prima
            # di fare qualsiasi ombra o sfumatura. In questo modo le sfocature
            # sfumano morbidamente a zero e non si "schiantano" contro il bordo dell'immagine.
            pad_v = int(max(target_v_w, target_v_h) * 0.60) + 15
            varroa_padded = cv2.copyMakeBorder(
                varroa_resized, pad_v, pad_v, pad_v, pad_v, 
                cv2.BORDER_CONSTANT, value=[0, 0, 0, 0]
            )

            # --- 2. Rotazione casuale con espansione ---
            angle = random.uniform(0, 360)
            varroa_rotated = rotate_image_rgba(varroa_padded, angle)

            # Estraiamo i canali dopo la rotazione
            v_rgb = varroa_rotated[:, :, :3]
            v_alpha_raw = varroa_rotated[:, :, 3]
            rv_h, rv_w = v_rgb.shape[:2]

            # Crea maschera binaria netta per la GT mask
            _, v_mask_binary = cv2.threshold(v_alpha_raw, 128, 255, cv2.THRESH_BINARY)

            # Crea alpha sfumato proporzionale alla grandezza della varroa
            # Abbassato all'8% per renderla più marcata e definita (meno "sbiadita" sui bordi)
            dynamic_feather = max(2, int(rv_w * 0.08))
            v_alpha_feathered = feather_alpha(v_mask_binary, radius=dynamic_feather)
            v_alpha_float = v_alpha_feathered.astype(np.float32) / 255.0

            # --- 3. Posizionamento intelligente sul corpo dell'ape ---
            # Più tentativi (50) per trovare posizioni valide anche con varroa già presenti
            insert_x, insert_y = None, None
            for _attempt in range(50):
                cand_x, cand_y = find_body_position(favo, det, rv_w, rv_h, gt_mask=gt_mask, v_mask_binary=v_mask_binary, num_candidates=30)
                
                if cand_x is None or cand_y is None:
                    continue

                # Controlla sovrapposizione con varroa già inserite nella gt_mask
                oy1 = max(cand_y, 0)
                oy2 = min(cand_y + rv_h, H)
                ox1 = max(cand_x, 0)
                ox2 = min(cand_x + rv_w, W)
                if oy2 <= oy1 or ox2 <= ox1:
                    continue

                vy1_check = oy1 - cand_y
                vy2_check = vy1_check + (oy2 - oy1)
                vx1_check = ox1 - cand_x
                vx2_check = vx1_check + (ox2 - ox1)

                # Controlla se la nuova varroa (parte opaca) si sovrappone a una già presente
                overlap_region = gt_mask[oy1:oy2, ox1:ox2]
                new_varroa_region = v_mask_binary[vy1_check:vy2_check, vx1_check:vx2_check]
                overlap_pixels = cv2.bitwise_and(overlap_region, new_varroa_region)

                if np.count_nonzero(overlap_pixels) == 0:
                    insert_x, insert_y = cand_x, cand_y
                    break

            # Se non si è trovata una posizione senza overlap, salta questa varroa
            if insert_x is None:
                continue

            # --- 4. Adattamento colore leggero ---
            roi_y1 = max(insert_y, 0)
            roi_y2 = min(insert_y + rv_h, H)
            roi_x1 = max(insert_x, 0)
            roi_x2 = min(insert_x + rv_w, W)
            if roi_y2 <= roi_y1 or roi_x2 <= roi_x1:
                continue

            target_region = synthetic_img[roi_y1:roi_y2, roi_x1:roi_x2]
            v_rgb_adapted = adapt_color_to_region(v_rgb, target_region, v_mask_binary)

            # --- 5. Alpha compositing diretto ---
            synthetic_img = alpha_composite(synthetic_img, v_rgb_adapted, v_alpha_float, insert_x, insert_y)

            # --- 5b. Ombra di contatto ---
            # Ridotto l'alone e leggermente scurita l'ombra per "piantarla" meglio
            contact_ring_w = max(3, int(min(rv_w, rv_h) * 0.10))
            synthetic_img = add_contact_shadow(
                synthetic_img, v_mask_binary, insert_x, insert_y,
                ring_width=contact_ring_w, opacity=0.25
            )

            # --- 6. Ombra sottile spostata ---
            shadow_offset = max(1, int(min(rv_w, rv_h) * 0.08))
            shadow_blur = max(3, int(min(rv_w, rv_h) * 0.20)) | 1
            synthetic_img = add_subtle_shadow(
                synthetic_img, v_mask_binary, insert_x, insert_y,
                offset=(shadow_offset, shadow_offset),
                blur_size=shadow_blur,
                opacity=0.30
            )

            # --- 7. Aggiorna la Ground Truth Mask ---
            sy1 = max(insert_y, 0)
            sy2 = min(insert_y + rv_h, H)
            sx1 = max(insert_x, 0)
            sx2 = min(insert_x + rv_w, W)
            vy1 = sy1 - insert_y
            vy2 = vy1 + (sy2 - sy1)
            vx1 = sx1 - insert_x
            vx2 = vx1 + (sx2 - sx1)
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
        image_files = glob.glob(os.path.join(IMAGES_DIR, "*.[jJ][pP][gG]")) + glob.glob(os.path.join(IMAGES_DIR, "*.[pP][nN][gG]"))
        print(f"Trovate {len(image_files)} immagini in {IMAGES_DIR}")

        # Crea sottocartelle per immagini e maschere
        OUT_IMG_DIR = os.path.join(OUTPUT_DIR, "images")
        OUT_MASK_DIR = os.path.join(OUTPUT_DIR, "masks")
        os.makedirs(OUT_IMG_DIR, exist_ok=True)
        os.makedirs(OUT_MASK_DIR, exist_ok=True)

        for path_favo in image_files:
            base_name = os.path.splitext(os.path.basename(path_favo))[0]
            path_labels = os.path.join(LABELS_DIR, base_name + ".txt")

            if not os.path.exists(path_labels):
                print(f"Skipping {base_name}: label non trovata.")
                continue

            print(f"\nElaborazione di: {base_name}")
            favo_img, varroa_list = load_images(path_favo)
            H, W = favo_img.shape[:2]

            print(f"  Immagine caricata: {W}x{H}, {len(varroa_list)} varianti varroa caricate")

            # 1. Leggi le detection YOLO segmentation (poligoni)
            detections = parse_yolo_segmentation_labels(path_labels, W, H)

            print(f"  Rilevate {len(detections)} api.")

            if not detections:
                print(f"Nessuna ape rilevata in {base_name}. Skipping.")
                continue

            num_variants = 1
            for i in range(num_variants):
                print(f"  Generazione variante {i}...")
                syn_img, gt_mask = generate_synthetic_anomaly(favo_img, varroa_list, detections)
                print(f"  Generazione completata per variante {i}.")

                if syn_img is not None:
                    out_img_path = os.path.join(OUT_IMG_DIR, f"{base_name}_syn.jpg")
                    out_mask_path = os.path.join(OUT_MASK_DIR, f"{base_name}_mask.png")
                    cv2.imwrite(out_img_path, syn_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    cv2.imwrite(out_mask_path, gt_mask * 255)
                    print(f"  ✓ Salvata: {out_img_path}")
                else:
                    print(f"  ✗ Fallito inserimento su {base_name}.")

        print(f"\nPipeline completata. Controlla la cartella: {OUTPUT_DIR}")

    except Exception as e:
        import traceback
        print(f"Errore fatale: {e}")
        traceback.print_exc()