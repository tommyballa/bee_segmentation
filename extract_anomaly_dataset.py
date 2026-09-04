import os
import cv2
import glob
import numpy as np
from ultralytics import YOLO

def main():
    print(" Caricamento del modello YOLO...")
    model = YOLO("runs/segment/runs/segment/bee_model_finetuned_yolo26s/weights/best.pt")
    
    # Scegli la cartella da cui estrarre le api (qui usiamo il train set remoto)
    source_dir = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/train/images"
    
    # Output: cartella con TUTTE le api estratte grezze (senza filtri)
    out_dir = "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/anomaly_dataset/raw_bees"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Eseguendo l'estrazione delle api da: {source_dir}")
    
    image_paths = (
        glob.glob(os.path.join(source_dir, "*.[jJ][pP][gG]")) +
        glob.glob(os.path.join(source_dir, "*.[pP][nN][gG]"))
    )
    bee_counter = 0
    
    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Facciamo inferenza alla massima risoluzione per prendere bene i bordi
        # Soglia di confidenza BASSA: estraiamo tutto, sarà cleaning_and_clustering a filtrare
        results = model.predict(source=img, save=False, show=False, conf=0.05, imgsz=1920)
        
        if len(results) == 0 or results[0].masks is None:
            continue
            
        res = results[0]
        boxes = res.boxes.xyxy.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy()
        
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
            
            # Aggiungiamo un po' di margine (padding) per non tagliare al limite esatto del bordo
            padding = 10
            x1_p = max(0, x1 - padding)
            y1_p = max(0, y1 - padding)
            x2_p = min(w, x2 + padding)
            y2_p = min(h, y2 + padding)
            
            # Ritagliamo sia la foto che la maschera usando le coordinate maggiorate
            img_crop = img[y1_p:y2_p, x1_p:x2_p].copy()
            mask_crop = blank_mask[y1_p:y2_p, x1_p:x2_p]
            
            # Convertiamo la maschera a 3 canali per poterla applicare alla foto a colori
            mask_crop_3c = cv2.cvtColor(mask_crop, cv2.COLOR_GRAY2BGR)
            
            # Applica la maschera: mantiene l'ape intatta e rende tutto il resto nero (0,0,0)
            final_crop = cv2.bitwise_and(img_crop, mask_crop_3c)
            
            # RENDIAMOLA QUADRATA E CENTRATA (come i dataset MVTec)
            crop_w = x2_p - x1_p
            crop_h = y2_p - y1_p
            sq_size = max(crop_w, crop_h)
            
            # Creiamo il quadrato nero puro
            square_img = np.zeros((sq_size, sq_size, 3), dtype=np.uint8)
            
            # Calcoliamo dove incollare l'ape per farla stare perfettamente al centro
            off_x = (sq_size - crop_w) // 2
            off_y = (sq_size - crop_h) // 2
            square_img[off_y:off_y+crop_h, off_x:off_x+crop_w] = final_crop
            
            # Ridimensiona a 224x224 (standard di PatchCore / ResNet)
            square_img = cv2.resize(square_img, (224, 224))
            
            # Salva il risultato con la confidenza nel nome per debug
            bee_filename = f"bee_{bee_counter:05d}_conf{confs[i]:.2f}.png"
            cv2.imwrite(os.path.join(out_dir, bee_filename), square_img)
            bee_counter += 1
            
    print(f"\n✅ Estrazione completata! Ho salvato {bee_counter} api grezze in '{out_dir}'.")
    print("Ora lancia cleaning_and_clustering.py per pulire e clusterizzare il dataset.")

if __name__ == "__main__":
    main()
