import cv2
import os
import numpy as np

image_dir = '/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/train/images'
label_dir = '/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/train/labels'
output_dir = 'visualizations'

os.makedirs(output_dir, exist_ok=True)
print(f"Salvataggio delle visualizzazioni nella cartella '{output_dir}'...")

# Prendi solo le prime 50 immagini per non riempire la cartella inutilmente, o togli il counter per farle tutte
count = 0
for img_name in os.listdir(image_dir):
    if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
        
    img_path = os.path.join(image_dir, img_name)
    label_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + '.txt')
    
    if not os.path.exists(label_path):
        continue
        
    # Leggi l'immagine con OpenCV
    image = cv2.imread(img_path)
    if image is None:
        continue
    h, w, _ = image.shape
    
    # Leggi le maschere dal file testuale
    with open(label_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
            
        # Salta l'ID della classe (parts[0]) ed estrai le coordinate (x, y) normalizzate
        coords = [float(p) for p in parts[1:]]
        points = []
        for i in range(0, len(coords), 2):
            # De-normalizza le coordinate moltiplicandole per larghezza (w) e altezza (h)
            x = int(coords[i] * w)
            y = int(coords[i+1] * h)
            points.append([x, y])
            
        # Converti i punti in un array numpy nel formato atteso da OpenCV
        points_np = np.array(points, np.int32).reshape((-1, 1, 2))
        
        # Disegna il poligono riempito con trasparenza
        overlay = image.copy()
        cv2.fillPoly(overlay, [points_np], color=(0, 255, 0))  # Verde
        cv2.addWeighted(overlay, 0.4, image, 0.6, 0, image)
        
        # Disegna il bordo del poligono
        cv2.polylines(image, [points_np], isClosed=True, color=(0, 255, 0), thickness=2)

    # Salva l'immagine
    out_path = os.path.join(output_dir, img_name)
    cv2.imwrite(out_path, image)
    
    count += 1
    if count >= 30: # Limite di 30 immagini per verificare che le labels siano giuste
        break

print(f"Completato! Ho salvato {count} immagini in {output_dir}/")
