import os
import cv2
import glob

import shutil

def resize_images_in_folder(folder_path, output_folder, max_dimension=1920):
    os.makedirs(output_folder, exist_ok=True)
    
    # Trova tutte le immagini jpg/png
    images = glob.glob(os.path.join(folder_path, "*.[jJ][pP][gG]")) + \
             glob.glob(os.path.join(folder_path, "*.[pP][nN][gG]"))
             
    print(f"Trovate {len(images)} immagini in {folder_path}.")
    
    for img_path in images:
        img_name = os.path.basename(img_path)
        out_path = os.path.join(output_folder, img_name)
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"Errore nella lettura di {img_path}")
            continue
            
        h, w = img.shape[:2]
        
        # Se l'immagine è già più piccola della dimensione massima, la copiamo e basta
        if max(h, w) <= max_dimension:
            shutil.copy2(img_path, out_path)
            continue
            
        # Calcoliamo il fattore di scala per mantenere le proporzioni
        scale = max_dimension / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Ridimensiona
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Salva nella nuova cartella
        cv2.imwrite(out_path, resized_img)
        print(f"Salvata: {img_name} ({w}x{h} -> {new_w}x{new_h})")

def main():
    print("Inizio il ridimensionamento a risoluzione '1080p' (lato lungo max 1920px)...")
    print("Creerò una copia dell'intero dataset nella nuova cartella 'bee_dataset_1080p'")
    
    # Cartelle di origine
    train_dir = "bee_dataset/images/train"
    val_dir = "bee_dataset/images/val"
    train_lbl_dir = "bee_dataset/labels/train"
    val_lbl_dir = "bee_dataset/labels/val"
    
    # Cartelle di destinazione
    out_train_dir = "bee_dataset_1080p/images/train"
    out_val_dir = "bee_dataset_1080p/images/val"
    out_train_lbl = "bee_dataset_1080p/labels/train"
    out_val_lbl = "bee_dataset_1080p/labels/val"
    
    # Ridimensiona immagini
    if os.path.exists(train_dir):
        resize_images_in_folder(train_dir, out_train_dir, max_dimension=1920)
        # Copia le etichette
        if os.path.exists(train_lbl_dir):
            shutil.copytree(train_lbl_dir, out_train_lbl, dirs_exist_ok=True)
            
    if os.path.exists(val_dir):
        resize_images_in_folder(val_dir, out_val_dir, max_dimension=1920)
        # Copia le etichette
        if os.path.exists(val_lbl_dir):
            shutil.copytree(val_lbl_dir, out_val_lbl, dirs_exist_ok=True)
            
    # Crea il nuovo data.yaml
    with open("bee_dataset/data.yaml", "r") as f:
        yaml_content = f.read()
        
    # Sostituisci il vecchio percorso con il nuovo
    yaml_content = yaml_content.replace("bee_dataset", "bee_dataset_1080p")
    
    with open("bee_dataset_1080p/data.yaml", "w") as f:
        f.write(yaml_content)
        
    print("\n✅ Dataset clonato e ridimensionato con successo in 'bee_dataset_1080p'!")
    print("Le etichette (file .txt) rimarranno valide al 100% perché YOLO usa coordinate in percentuale (normalizzate) e non in pixel assoluti!")

if __name__ == "__main__":
    main()
