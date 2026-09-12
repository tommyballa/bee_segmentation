import os
import cv2
import glob
import json
import numpy as np

# ============================================================
# PARAMETRI DI PULIZIA (sanificazione)
# ============================================================
MIN_BLUR_SCORE = 50.0       # Sotto questa soglia l'immagine è troppo sfocata
MIN_AREA_PIXELS = 1500      # Area minima della maschera (in pixel) per essere un'ape vera
MIN_FILL_RATIO = 0.30       # Un'ape riempie almeno il 30% del bounding box
MAX_ASPECT_RATIO = 3.0      # Rapporto d'aspetto massimo (evita frammenti tipo ali/zampe)
MIN_BBOX_SIDE = 25          # Dimensione minima di un lato del bbox (in pixel prima del resize a 224)
MIN_CONFIDENCE = 0.35       # Confidenza minima di YOLO (letta dal nome file)

# ============================================================
# PARAMETRI DI NORMALIZZAZIONE PER ANOMALY DETECTION
# ============================================================
TARGET_SIZE = 512           # Dimensione finale dell'immagine quadrata
TARGET_FILL_RATIO = 0.60    # L'ape deve occupare ~60% del frame dopo la normalizzazione
EDGE_FEATHER_PX = 12        # Pixel di sfumatura graduale ai bordi della maschera

# Nome del file manifesto che traccia i file già processati (accettati e scartati)
CLEANING_MANIFEST_FILENAME = "_cleaning_manifest.json"


def parse_confidence(filename):
    """Estrae la confidenza dal nome file (es. bee_00042_conf0.87.png -> 0.87)."""
    try:
        conf_part = filename.split("_conf")[1].replace(".png", "")
        return float(conf_part)
    except (IndexError, ValueError):
        return 1.0  # Se il nome non ha il formato, accetta il file


def load_cleaning_manifest(output_dir):
    """Carica il manifesto della pulizia (file già processati: accettati e scartati)."""
    manifest_path = os.path.join(output_dir, CLEANING_MANIFEST_FILENAME)
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            return json.load(f)
    return {"processed_files": {}}


def save_cleaning_manifest(output_dir, manifest):
    """Salva il manifesto della pulizia aggiornato."""
    manifest_path = os.path.join(output_dir, CLEANING_MANIFEST_FILENAME)
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)


def load_image_and_mask(path):
    """
    Carica un'immagine e la relativa maschera.
    
    Supporta due formati:
    - RGBA PNG (nuovo): il canale Alpha contiene la maschera grezza del poligono YOLO,
      preservata senza alcun processing dall'estrazione. Questo evita la perdita
      di informazione causata dalla ri-derivazione della maschera dai pixel.
    - RGB PNG/JPG (legacy): la maschera viene derivata dal threshold dei pixel non-neri.
      Compatibilità con immagini estratte dalla versione precedente dello script.
    
    Returns:
        img_bgr (H x W x 3), mask (H x W), is_raw_polygon (bool)
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, None, False
    
    if img.ndim == 3 and img.shape[2] == 4:
        # RGBA: estrai maschera dal canale alpha (dati grezzi dal poligono YOLO)
        mask = img[:, :, 3]
        img_bgr = img[:, :, :3]
        return img_bgr, mask, True
    else:
        # RGB legacy: deriva maschera dal threshold (fallback per vecchie estrazioni)
        img_bgr = img
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        return img_bgr, mask, False


def refine_raw_polygon_mask(mask):
    """
    Raffina la maschera grezza del poligono YOLO.
    
    Applica le operazioni morfologiche che migliorano la continuità
    e la copertura della maschera. Queste operazioni venivano prima
    eseguite sia in extract_anomaly_dataset.py che qui, causando
    doppia elaborazione. Ora sono centralizzate solo qui.
    
    Pipeline:
    1. Chiusura morfologica → riempie buchi interni e fratture nel poligono
    2. Dilatazione → espande la maschera per coprire peli e zampe
    
    Questa funzione viene chiamata SOLO per maschere grezze (RGBA).
    Le immagini legacy (RGB) hanno già subìto queste operazioni
    durante l'estrazione e NON vengono ri-processate.
    """
    # 1. Chiusura morfologica: riempie buchi interni e unisce rientranze (fratture)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    
    # 2. Espandiamo la maschera per non tagliare peli e zampe
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.dilate(mask, kernel_dilate, iterations=1)
    
    return mask


def normalize_orientation(img_bgr, mask):
    """
    Ruota l'ape in modo che il suo asse maggiore sia sempre verticale.
    Usa cv2.minAreaRect per trovare l'angolo dell'ellisse che meglio
    approssima la sagoma e la raddrizza.
    
    Returns:
        img_rotated, mask_rotated
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_bgr, mask
    
    largest = max(contours, key=cv2.contourArea)
    
    # Serve un minimo di 5 punti per fitEllipse
    if len(largest) < 5:
        return img_bgr, mask
    
    # minAreaRect restituisce (centro, (larghezza, altezza), angolo)
    rect = cv2.minAreaRect(largest)
    center = rect[0]
    w_rect, h_rect = rect[1]
    angle = rect[2]
    
    # minAreaRect restituisce angoli tra -90 e 0.
    # Vogliamo che l'asse MAGGIORE sia verticale (0°).
    # Se la larghezza > altezza, l'asse maggiore è "orizzontale" per OpenCV,
    # quindi dobbiamo ruotare di (angle + 90) per metterlo in verticale.
    # Altrimenti basta ruotare di (angle) gradi.
    if w_rect > h_rect:
        rotation_angle = angle + 90
    else:
        rotation_angle = angle
    
    h_img, w_img = img_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w_img / 2, h_img / 2), rotation_angle, 1.0)
    
    # Calcoliamo la nuova dimensione del canvas per non tagliare nulla dopo la rotazione
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h_img * sin_a + w_img * cos_a)
    new_h = int(h_img * cos_a + w_img * sin_a)
    
    # Aggiustiamo la matrice di rotazione per centrare l'immagine nel nuovo canvas
    M[0, 2] += (new_w - w_img) / 2
    M[1, 2] += (new_h - h_img) / 2
    
    img_rotated = cv2.warpAffine(img_bgr, M, (new_w, new_h), 
                                  flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    mask_rotated = cv2.warpAffine(mask, M, (new_w, new_h), 
                                   flags=cv2.INTER_NEAREST, borderValue=0)
    
    return img_rotated, mask_rotated


def normalize_scale(img_bgr, mask, target_size=512, target_fill=0.60):
    """
    Riscala l'ape in modo che occupi sempre ~target_fill% del frame finale.
    Poi la centra in un quadrato target_size x target_size.
    
    Returns:
        img_final (target_size x target_size x 3), mask_final (target_size x target_size)
    """
    # Trova il bounding box stretto dell'ape
    coords = cv2.findNonZero(mask)
    if coords is None:
        return img_bgr, mask
    
    x, y, bw, bh = cv2.boundingRect(coords)
    
    # Ritaglia stretto attorno all'ape
    img_tight = img_bgr[y:y+bh, x:x+bw]
    mask_tight = mask[y:y+bh, x:x+bw]
    
    # Calcola il fattore di scala: vogliamo che il lato maggiore dell'ape
    # occupi target_fill * target_size pixel
    bee_max_side = max(bw, bh)
    desired_side = int(target_size * target_fill)
    scale = desired_side / bee_max_side if bee_max_side > 0 else 1.0
    
    new_w = int(bw * scale)
    new_h = int(bh * scale)
    
    # Riscala
    img_scaled = cv2.resize(img_tight, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    mask_scaled = cv2.resize(mask_tight, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    
    # Centra nel quadrato finale
    img_final = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    mask_final = np.zeros((target_size, target_size), dtype=np.uint8)
    
    off_x = (target_size - new_w) // 2
    off_y = (target_size - new_h) // 2
    
    # Clamp nel caso in cui l'ape scalata sia più grande del target
    paste_w = min(new_w, target_size - off_x)
    paste_h = min(new_h, target_size - off_y)
    off_x = max(0, off_x)
    off_y = max(0, off_y)
    
    img_final[off_y:off_y+paste_h, off_x:off_x+paste_w] = img_scaled[:paste_h, :paste_w]
    mask_final[off_y:off_y+paste_h, off_x:off_x+paste_w] = mask_scaled[:paste_h, :paste_w]
    
    return img_final, mask_final


def feather_edges(img_bgr, mask, feather_px=12):
    """
    Sfuma gradualmente i bordi della maschera per creare una transizione
    morbida tra l'ape e lo sfondo nero, eliminando il taglio netto.
    
    Il risultato è un'immagine dove i pixel ai bordi dell'ape "sfumano"
    dolcemente verso il nero, riducendo gli artefatti di bordo per PatchCore.
    
    Returns:
        img_feathered (stesse dimensioni)
    """
    # Creiamo una versione "erosa" della maschera (il nucleo sicuramente dentro l'ape)
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (feather_px * 2 + 1, feather_px * 2 + 1))
    inner_mask = cv2.erode(mask, erode_kernel, iterations=1)
    
    # Creiamo un gradiente morbido: 
    # - Dentro (inner_mask) = 1.0 (completamente visibile)
    # - Bordo = sfumatura graduale da 1.0 a 0.0
    # - Fuori = 0.0 (completamente nero)
    # Per farlo, sfochiamo la maschera originale con un kernel grande
    alpha = mask.astype(np.float32) / 255.0
    alpha = cv2.GaussianBlur(alpha, (feather_px * 4 + 1, feather_px * 4 + 1), 0)
    
    # Forziamo l'interno a essere pieno (1.0) per non sfocare l'ape stessa
    alpha[inner_mask > 0] = 1.0
    
    # Applichiamo l'alpha all'immagine
    alpha_3c = np.stack([alpha] * 3, axis=-1)
    img_feathered = (img_bgr.astype(np.float32) * alpha_3c).astype(np.uint8)
    
    return img_feathered


def clean_and_filter(img_bgr, mask, is_raw_polygon, filename):
    """
    Applica tutti i filtri di sanificazione su una singola immagine.
    Ritorna l'immagine pulita, normalizzata e pronta per AD, oppure None se scartata.
    
    Args:
        img_bgr: immagine BGR (risoluzione originale del crop)
        mask: maschera binaria (dal canale alpha RGBA o da threshold legacy)
        is_raw_polygon: True se la maschera è il poligono grezzo YOLO (RGBA),
                        False se è già stata pre-processata (legacy RGB)
        filename: nome del file per estrarre la confidenza
    """
    # --- FILTRO 1: Confidenza YOLO ---
    conf = parse_confidence(filename)
    if conf < MIN_CONFIDENCE: return None
    
    # --- RAFFINAMENTO MASCHERA (solo per poligoni grezzi RGBA) ---
    # Per le immagini RGBA (nuove), applica chiusura + dilatazione sulla maschera
    # grezza del poligono. Per le immagini RGB (legacy), queste operazioni erano
    # già state fatte in extract_anomaly_dataset.py, quindi le saltiamo.
    if is_raw_polygon:
        mask = refine_raw_polygon_mask(mask)
    
    # --- FILTRO 2: Ape tagliata ai bordi ---
    h_img, w_img = mask.shape[:2]
    border_margin = 3
    border_pixels = (cv2.countNonZero(mask[:border_margin, :]) + cv2.countNonZero(mask[h_img - border_margin:, :]) +
                     cv2.countNonZero(mask[:, :border_margin]) + cv2.countNonZero(mask[:, w_img - border_margin:]))
    if border_pixels > 15: return None
    
    # --- FILTRO 3: Spezza ponti sottili (thin bridges) ---
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    
    contours, _ = cv2.findContours(opened_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    
    # --- FILTRO 4: Rimozione frammenti ---
    largest_contour = max(contours, key=cv2.contourArea)
    clean_mask = np.zeros_like(mask)
    cv2.drawContours(clean_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    clean_mask = cv2.dilate(clean_mask, dilate_kernel, iterations=1)
    clean_mask = cv2.bitwise_and(clean_mask, mask)
    
    total_mask_area = cv2.countNonZero(mask)
    main_area = cv2.countNonZero(clean_mask)
    if total_mask_area > 0:
        if (1.0 - (main_area / total_mask_area)) > 0.15: return None
    
    # Chiudiamo i buchi interni
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)
    
    # --- FILTRO 5: Area minima ---
    area = cv2.countNonZero(clean_mask)
    if area < MIN_AREA_PIXELS: return None
    
    # --- FILTRO 6: Aspect ratio ---
    x, y, bw, bh = cv2.boundingRect(largest_contour)
    if bw < MIN_BBOX_SIDE or bh < MIN_BBOX_SIDE: return None
    if max(bw, bh) / max(min(bw, bh), 1) > MAX_ASPECT_RATIO: return None
    
    # --- FILTRO 7: Fill ratio ---
    bbox_area = bw * bh
    if (area / bbox_area if bbox_area > 0 else 0) < MIN_FILL_RATIO: return None
    
    # --- FILTRO 8: Blur score ---
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if cv2.Laplacian(gray, cv2.CV_64F).var() < MIN_BLUR_SCORE: return None

    # --- SMOOTHING DEI CONTORNI ---
    # Un singolo passaggio di blur+threshold per arrotondare tutti i bordi
    # (prima veniva fatto sia in extract che qui — ora solo qui, una volta sola)
    clean_mask = cv2.GaussianBlur(clean_mask, (31, 31), 0)
    _, clean_mask = cv2.threshold(clean_mask, 127, 255, cv2.THRESH_BINARY)

    # --- PULIZIA: applica maschera + CLAHE ---
    cleaned_img = cv2.bitwise_and(img_bgr, img_bgr, mask=clean_mask)
    
    lab = cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    cleaned_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    cleaned_img = cv2.bitwise_and(cleaned_img, cleaned_img, mask=clean_mask)
    
    # ============================================================
    # NORMALIZZAZIONI PER ANOMALY DETECTION
    # ============================================================
    
    # 1. ORIENTAMENTO: ruota l'ape in modo che l'asse maggiore sia verticale
    cleaned_img, clean_mask = normalize_orientation(cleaned_img, clean_mask)
    
    # 2. SCALA: riscala l'ape in modo che occupi sempre ~60% del frame 512x512
    #    Questo è l'UNICO resize della pipeline (prima veniva fatto anche in extract,
    #    causando un doppio resize che degradava la qualità dell'immagine)
    cleaned_img, clean_mask = normalize_scale(cleaned_img, clean_mask, 
                                               target_size=TARGET_SIZE, 
                                               target_fill=TARGET_FILL_RATIO)
    
    # 3. SFUMATURA BORDI: transizione morbida ape → sfondo nero
    cleaned_img = feather_edges(cleaned_img, clean_mask, feather_px=EDGE_FEATHER_PX)
    
    return cleaned_img


def run_cleaning(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    img_paths = glob.glob(os.path.join(input_dir, "*.png")) + glob.glob(os.path.join(input_dir, "*.jpg"))
    
    # Ignora il file manifesto dell'estrazione se presente nella cartella input
    img_paths = [p for p in img_paths if not os.path.basename(p).startswith("_")]
    
    # Carica il manifesto della pulizia per sapere quali file sono già stati processati
    manifest = load_cleaning_manifest(output_dir)
    
    total = len(img_paths)
    discarded = 0
    saved = 0
    skipped = 0
    print(f"Trovate {total} immagini in '{input_dir}'.")
    print("Inizio pulizia e normalizzazione per Anomaly Detection...")
    
    for idx, path in enumerate(img_paths):
        filename = os.path.basename(path)
        
        # Skip se il file è già stato processato (accettato o scartato) in una run precedente
        if filename in manifest["processed_files"]:
            skipped += 1
            continue
        
        img_bgr, mask, is_raw_polygon = load_image_and_mask(path)
        if img_bgr is None:
            discarded += 1
            manifest["processed_files"][filename] = "error"
            continue
            
        processed_img = clean_and_filter(img_bgr, mask, is_raw_polygon, filename)
        if processed_img is None:
            discarded += 1
            manifest["processed_files"][filename] = "discarded"
        else:
            cv2.imwrite(os.path.join(output_dir, filename), processed_img)
            saved += 1
            manifest["processed_files"][filename] = "accepted"
        
        # Salva il manifesto periodicamente per robustezza a interruzioni
        if (idx + 1) % 100 == 0:
            save_cleaning_manifest(output_dir, manifest)
            print(f"  Processate {idx + 1}/{total} immagini...")
    
    # Salvataggio finale del manifesto
    save_cleaning_manifest(output_dir, manifest)
            
    print(f"\n📊 Risultato pulizia: {saved} api accettate, {discarded} scartate, {skipped} già processate su {total} totali.")
    print(f"Api pulite e normalizzate salvate in '{output_dir}'")


if __name__ == "__main__":
    raw_dir = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/train/anomaly_bees"
    clean_dir = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/processed/train_single_bee/cleaned_bees"
    run_cleaning(raw_dir, clean_dir)
