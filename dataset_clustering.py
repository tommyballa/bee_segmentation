import os
import cv2
import glob
import numpy as np
import torch
import torchvision.transforms as T
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import random
from sklearn.svm import SVC

PCA_COMPONENTS = 50

def run_clustering(input_dir, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Caricamento modello DINOv2 (ViT-S/14)...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model.to(device)
    model.eval()
    
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_paths = glob.glob(os.path.join(input_dir, "*.png")) + glob.glob(os.path.join(input_dir, "*.jpg"))
    if not img_paths:
        print(f"Nessuna immagine trovata in '{input_dir}'. Esegui prima la pulizia!")
        return

    valid_data = []
    features_list = []
    total = len(img_paths)
    print(f"Estrazione feature per {total} immagini...")
    
    for idx, path in enumerate(img_paths):
        filename = os.path.basename(path)
        img = cv2.imread(path)
        if img is None: continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = transform(img_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model(tensor).cpu().numpy().flatten()
            
        valid_data.append((filename, img))
        features_list.append(feat)
        if (idx + 1) % 500 == 0:
            print(f"  Elaborate {idx + 1}/{total} immagini...")

    # ============================================================
    # ACTIVE LEARNING CLASSIFICATION
    # ============================================================
    feat_matrix = np.array(features_list)
    n_components = min(PCA_COMPONENTS, len(valid_data), feat_matrix.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    reduced_feat = pca.fit_transform(feat_matrix)
    
    print("\n" + "="*60)
    print("FASE DI ACTIVE LEARNING (MODALITÀ TERMINALE / SERVER)")
    print("="*60)
    
    tmp_label_dir = os.path.join(output_dir, "tmp_labeling_images")
    os.makedirs(tmp_label_dir, exist_ok=True)
    
    print(f"ATTENZIONE: Ho salvato le immagini da etichettare in:\n  -> {tmp_label_dir}")
    print("Apri questa cartella (es. con VSCode, WinSCP o visualizzatore) per guardare le immagini.")
    print("\nIstruzioni:")
    print(" - Digita un NUMERO da 1 a 9 per assegnare l'immagine alla categoria.")
    print(" - Digita 's' per saltare l'immagine se è brutta/incerta.")
    print(" - Digita 'q' per finire l'etichettatura e addestrare il modello.")
    
    # Usiamo KMeans solo per selezionare campioni diversificati da mostrarti
    n_clusters_init = min(20, len(valid_data))
    kmeans_init = KMeans(n_clusters=n_clusters_init, random_state=42, n_init=10)
    kmeans_init.fit(reduced_feat)
    
    sample_indices = []
    for i in range(kmeans_init.n_clusters):
        cluster_idx = np.where(kmeans_init.labels_ == i)[0]
        if len(cluster_idx) > 0:
            sample_indices.extend(random.sample(list(cluster_idx), min(5, len(cluster_idx))))
    random.shuffle(sample_indices)
    
    labeled_indices = []
    labels_y = []
    category_names = {}
    
    for idx_in_list, idx in enumerate(sample_indices):
        filename, img = valid_data[idx]
        
        # Salviamo l'immagine temporanea
        tmp_filename = f"da_etichettare_{idx_in_list:03d}.png"
        tmp_path = os.path.join(tmp_label_dir, tmp_filename)
        cv2.imwrite(tmp_path, img)
        
        # Chiediamo l'input nel terminale
        risposta = input(f"\nGuarda '{tmp_filename}'. Categoria (1-9, s=salta, q=fine)? ").strip().lower()
        
        if risposta == 'q':
            break
        elif risposta == 's':
            print("-> Saltata.")
            continue
        elif risposta.isdigit() and 1 <= int(risposta) <= 9:
            class_id = int(risposta)
            if class_id not in category_names:
                category_names[class_id] = f"categoria_{class_id}"
                print(f"Creata nuova categoria: {class_id}")
            
            labeled_indices.append(idx)
            labels_y.append(class_id)
            print(f"-> Assegnata categoria {class_id}")
        else:
            print("-> Input non valido, la salto.")
            
    import shutil
    shutil.rmtree(tmp_label_dir, ignore_errors=True)
    
    if len(set(labels_y)) < 2:
        print("\n❌ Hai etichettato immagini di meno di 2 categorie.")
        print("Il classificatore SVM richiede almeno 2 categorie per distinguere.")
        print("Nessun salvataggio effettuato.")
        return
        
    print(f"\nHai etichettato {len(labels_y)} immagini. Addestramento del modello SVM in corso...")
    
    X_train = reduced_feat[labeled_indices]
    y_train = np.array(labels_y)
    
    # Support Vector Machine
    svm = SVC(kernel='rbf', class_weight='balanced')
    svm.fit(X_train, y_train)
    
    print("Classificazione automatica di tutte le restanti immagini...")
    predicted_labels = svm.predict(reduced_feat)
    
    # Salvataggio nelle cartelle finali
    cluster_counts = {}
    for (filename, img), pred_label in zip(valid_data, predicted_labels):
        cat_name = category_names[pred_label]
        cat_folder = os.path.join(output_dir, cat_name)
        os.makedirs(cat_folder, exist_ok=True)
        cv2.imwrite(os.path.join(cat_folder, filename), img)
        cluster_counts[cat_name] = cluster_counts.get(cat_name, 0) + 1
    
    print(f"\n Active Learning completato! Risultati salvati in '{output_dir}'.")
    print("Distribuzione finale:")
    for name in sorted(cluster_counts.keys()):
        print(f"  {name}: {cluster_counts[name]} immagini")


if __name__ == "__main__":
    clean_dir = "/mnt/disk1/borsattifr/datasets/bees_datasets/DatasetApi_Ceschi/processed/train_single_bee/cleaned_bees"
    output_dir = "/mnt/disk1/borsattifr/datasets/bees_datasets/DatasetApi_Ceschi/processed/train_single_bee/train_single_bee_clustered"
    run_clustering(clean_dir, output_dir)
    