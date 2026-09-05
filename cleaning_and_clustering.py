import os
import cv2
import glob
import shutil
import numpy as np
import torch
import torchvision.transforms as T
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

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
# PARAMETRI DI CLUSTERING
# ============================================================
FEATURE_EXTRACTOR = "dinov2"  # "dinov2" (consigliato per clustering semantico) oppure "resnet18"
NUM_CLUSTERS = 5            # Numero di cluster per KMeans
PCA_COMPONENTS = 50         # Componenti PCA per ridurre le feature
BATCH_SIZE = 64             # Dimensione del batch per estrazione feature su GPU

# ============================================================
# FUNZIONI DI PULIZIA
# ============================================================

def parse_confidence(filename):
    """Estrae la confidenza dal nome file (es. bee_00042_conf0.87.png -> 0.87)."""
    try:
        # Formato atteso: bee_XXXXX_confY.YY.png
        conf_part = filename.split("_conf")[1].replace(".png", "")
        return float(conf_part)
    except (IndexError, ValueError):
        return 1.0  # Se il nome non ha il formato, accetta il file


def clean_and_filter(img_bgr, filename):
    """
    Applica tutti i filtri di sanificazione su una singola immagine.
    Ritorna l'immagine pulita e migliorata, oppure None se scartata.
    """
    # --- FILTRO 1: Confidenza YOLO ---
    conf = parse_confidence(filename)
    if conf < MIN_CONFIDENCE:
        return None
    
    # Convertiamo in scala di grigi e creiamo la maschera binaria
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    # --- FILTRO 2: Ape tagliata ai bordi ---
    # Se la maschera tocca il bordo dell'immagine 224x224, l'ape era ai margini
    # della foto originale ed è stata tagliata → la scartiamo
    h_img, w_img = mask.shape[:2]
    border_margin = 3  # pixel dal bordo
    top_strip    = mask[:border_margin, :]
    bottom_strip = mask[h_img - border_margin:, :]
    left_strip   = mask[:, :border_margin]
    right_strip  = mask[:, w_img - border_margin:]
    border_pixels = (cv2.countNonZero(top_strip) + cv2.countNonZero(bottom_strip) +
                     cv2.countNonZero(left_strip) + cv2.countNonZero(right_strip))
    # Se ci sono più di pochi pixel bianchi sul bordo, l'ape è tagliata
    if border_pixels > 15:
        return None
    
    # --- FILTRO 3: Spezza ponti sottili (thin bridges) ---
    # Alcune maschere hanno l'ape principale collegata a frammenti di altre api
    # tramite strisce di pochi pixel. Un'erosione seguita da dilatazione (opening)
    # spezza quei ponti, separando i blob.
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    
    # Troviamo i contorni sulla maschera "aperta" (ponti spezzati)
    contours, _ = cv2.findContours(opened_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    # --- FILTRO 4: Rimozione frammenti di altre api ---
    # Teniamo SOLO il contorno più grande (l'ape principale).
    # Tutti gli altri blob (pezzetti di api vicine) vengono rimossi.
    largest_contour = max(contours, key=cv2.contourArea)
    clean_mask = np.zeros_like(mask)
    cv2.drawContours(clean_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    
    # Ridilattiamo leggermente per recuperare i pixel persi dall'erosione ai bordi dell'ape
    clean_mask = cv2.dilate(clean_mask, open_kernel, iterations=1)
    # Ma limitiamo alla maschera originale per non inventare pixel
    clean_mask = cv2.bitwise_and(clean_mask, mask)
    
    # Se c'erano frammenti significativi, il crop è sporco
    total_mask_area = cv2.countNonZero(mask)
    main_area = cv2.countNonZero(clean_mask)
    if total_mask_area > 0:
        fragment_ratio = 1.0 - (main_area / total_mask_area)
        # Se più del 15% della maschera era composta da frammenti, è un crop problematico
        if fragment_ratio > 0.15:
            return None
    
    # Chiudiamo i buchi interni nella maschera dell'ape
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)
    
    # --- FILTRO 5: Area minima della maschera ---
    area = cv2.countNonZero(clean_mask)
    if area < MIN_AREA_PIXELS:
        return None
    
    # --- FILTRO 6: Aspect ratio del bounding box ---
    x, y, bw, bh = cv2.boundingRect(largest_contour)
    if bw < MIN_BBOX_SIDE or bh < MIN_BBOX_SIDE:
        return None
    aspect_ratio = max(bw, bh) / max(min(bw, bh), 1)
    if aspect_ratio > MAX_ASPECT_RATIO:
        return None
    
    # --- FILTRO 7: Fill ratio (rapporto di riempimento) ---
    bbox_area = bw * bh
    fill_ratio = area / bbox_area if bbox_area > 0 else 0
    if fill_ratio < MIN_FILL_RATIO:
        return None
    
    # --- FILTRO 8: Blur score (nitidezza) ---
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_score < MIN_BLUR_SCORE:
        return None

    # --- PULIZIA FINALE: Maschera e migliora il contrasto ---
    # Applichiamo la clean_mask per rimuovere i frammenti dall'immagine stessa
    cleaned_img = cv2.bitwise_and(img_bgr, img_bgr, mask=clean_mask)
    
    # Boost del contrasto con CLAHE per rendere visibili eventuali varroa
    lab = cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    enhanced_lab = cv2.merge((cl, a, b))
    final_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # Assicuriamoci che lo sfondo sia perfettamente nero
    final_img = cv2.bitwise_and(final_img, final_img, mask=clean_mask)
    
    return final_img


# ============================================================
# PIPELINE PRINCIPALE
# ============================================================

def run_pipeline(input_dir, output_dir):
    """
    Pipeline completa: pulizia + estrazione feature + clustering.
    
    Args:
        input_dir:  cartella con le api grezze (output di extract_anomaly_dataset.py)
        output_dir: cartella dove salvare i cluster puliti (es. train/good per MVTec)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Carichiamo DINOv2 (ViT-S/14) per estrarre feature visive
    # DINOv2 è addestrato in modo self-supervised: cattura forma, texture e struttura
    # molto meglio di ResNet18 ImageNet per il clustering di oggetti simili
    print("Caricamento modello DINOv2 (ViT-S/14)...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model.to(device)
    model.eval()
    
    # DINOv2 usa la stessa normalizzazione ImageNet ma a 224x224
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    valid_data = []
    features_list = []
    
    # Cerchiamo sia PNG che JPG
    img_paths = (
        glob.glob(os.path.join(input_dir, "*.png")) +
        glob.glob(os.path.join(input_dir, "*.jpg"))
    )
    
    total = len(img_paths)
    discarded = 0
    print(f"Trovate {total} immagini grezze in '{input_dir}'.")
    print("Inizio pulizia e filtraggio...")
    
    for idx, path in enumerate(img_paths):
        filename = os.path.basename(path)
        img = cv2.imread(path)
        if img is None:
            discarded += 1
            continue
        
        # Applica tutti i filtri di sanificazione
        processed_img = clean_and_filter(img, filename)
        if processed_img is None:
            discarded += 1
            continue
            
        # Estrai feature deep sull'immagine pulita
        img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        tensor = transform(img_rgb).unsqueeze(0).to(device)
        
        with torch.no_grad():
            feat = model(tensor).cpu().numpy().flatten()
            
        valid_data.append((filename, processed_img))
        features_list.append(feat)
        
        # Stampa progresso ogni 500 immagini
        if (idx + 1) % 500 == 0:
            print(f"  Processate {idx + 1}/{total} immagini...")

    print(f"\n📊 Risultato pulizia: {len(valid_data)} api accettate, {discarded} scartate su {total} totali.")
    
    if not valid_data:
        print("❌ Nessuna immagine ha passato i filtri. Prova ad abbassare le soglie.")
        return
    
    # ============================================================
    # CLUSTERING
    # ============================================================
    print(f"\nAvvio clustering con KMeans (k={NUM_CLUSTERS})...")
    
    # Riduciamo la dimensionalità per aiutare KMeans
    feat_matrix = np.array(features_list)
    n_components = min(PCA_COMPONENTS, len(valid_data), feat_matrix.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    reduced_feat = pca.fit_transform(feat_matrix)
    
    # Adattiamo il numero di cluster se abbiamo poche immagini
    n_clusters = min(NUM_CLUSTERS, len(valid_data))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(reduced_feat)
    
    # Salviamo le immagini pulite nelle cartelle dei cluster
    cluster_counts = {}
    for (filename, img), label in zip(valid_data, labels):
        cluster_folder = os.path.join(output_dir, f"cluster_{label}")
        os.makedirs(cluster_folder, exist_ok=True)
        cv2.imwrite(os.path.join(cluster_folder, filename), img)
        cluster_counts[label] = cluster_counts.get(label, 0) + 1
    
    # Stampa il riepilogo dei cluster
    print(f"\n✅ Clustering completato! Risultati salvati in '{output_dir}'.")
    print("Distribuzione dei cluster:")
    for label in sorted(cluster_counts.keys()):
        print(f"  cluster_{label}: {cluster_counts[label]} immagini")


if __name__ == "__main__":
    # Input: la cartella con le api grezze prodotte da extract_anomaly_dataset.py
    raw_dir = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/anomaly_dataset/raw_bees"
    # Output: le api pulite e clusterizzate
    output_dir = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/anomaly_dataset/clustered_bees"
    
    run_pipeline(raw_dir, output_dir)