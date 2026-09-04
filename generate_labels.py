import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import cv2
import numpy as np
from PIL import Image
import torchvision
from lang_sam import LangSAM
import gc

# init model
model = LangSAM()

datasets = [
    ("/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/train/images", "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/train/labels"),
    ("/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/test/images", "/home/tommaso_ballarin/bees_datasets/DatasetApi_Ceschi/test/labels")
]

text_prompt = "bee."
TARGET_BEE_SIZE = 120.0
BATCH_SIZE = 8  # Processa 8 ritagli alla volta sulla GPU (velocità 8x!)

for image_dir, label_dir in datasets:
    print(f"Processing dataset: {image_dir}")
    os.makedirs(label_dir, exist_ok=True)
    for img_name in os.listdir(image_dir):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        label_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + '.txt')
        
        if os.path.exists(label_path):
            print(f"Skipping {img_name} (already processed)")
            continue
            
        img_path = os.path.join(image_dir, img_name)
        original_pil = Image.open(img_path).convert("RGB")
        
        # ---------------------------------------------------------
        # PASS 1: QUICK SCAN TO ESTIMATE BEE SIZE
        # ---------------------------------------------------------
        scan_pil = original_pil.copy()
        scan_pil.thumbnail((800, 800), Image.Resampling.LANCZOS)
        
        ratio_w = original_pil.width / scan_pil.width
        ratio_h = original_pil.height / scan_pil.height
        scale_factor = 1.0
        
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            with torch.no_grad():
                try:
                    quick_results = model.predict(
                        [scan_pil], 
                        [text_prompt], 
                        box_threshold=0.25, 
                        text_threshold=0.25
                    )
                except Exception as e:
                    print(f"Errore nello scan veloce: {e}")
                    quick_results = []
                
        if len(quick_results) > 0 and len(quick_results[0].get("boxes", [])) > 0:
            boxes = quick_results[0].get("boxes", [])
            if hasattr(boxes, "cpu"):
                boxes = boxes.cpu().numpy()
                
            sizes = []
            for box in boxes:
                w = (box[2] - box[0]) * ratio_w
                h = (box[3] - box[1]) * ratio_h
                sizes.append(max(w, h)) 
                
            median_bee_size = float(np.median(sizes))
            
            if median_bee_size > 0:
                scale_factor = TARGET_BEE_SIZE / median_bee_size
                scale_factor = max(0.3, min(scale_factor, 5.0)) # Alzato a 5.0 per api piccole
                
        try: del quick_results, boxes
        except NameError: pass
        gc.collect()
        torch.cuda.empty_cache()

        # ---------------------------------------------------------
        # PASS 2: APPLY ZOOM E BATCH TILING
        # ---------------------------------------------------------
        new_w = int(original_pil.width * scale_factor)
        new_h = int(original_pil.height * scale_factor)
        image_pil = original_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        img_w, img_h = image_pil.size
        
        TILE_SIZE = 500
        OVERLAP = int(TILE_SIZE * 0.30)
        STRIDE = TILE_SIZE - OVERLAP
        
        crops = []
        for y in range(0, img_h, STRIDE):
            for x in range(0, img_w, STRIDE):
                x1 = x
                y1 = y
                x2 = min(img_w, x + TILE_SIZE)
                y2 = min(img_h, y + TILE_SIZE)
                
                if x2 - x1 < 50 or y2 - y1 < 50:
                    continue
                    
                crops.append((x1, y1, x2, y2))
                
        all_contours = []
        all_scores = []
        
        with open(label_path, 'w') as f:
            # PROCESSO IN BATCH!
            for i in range(0, len(crops), BATCH_SIZE):
                batch_crops = crops[i:i+BATCH_SIZE]
                batch_tiles = [image_pil.crop(cb) for cb in batch_crops]
                batch_prompts = [text_prompt] * len(batch_tiles)
                
                torch.cuda.empty_cache()
                
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    with torch.no_grad():
                        try:
                            batch_results = model.predict(
                                batch_tiles, 
                                batch_prompts, 
                                box_threshold=0.20, # Abbassato per api piccole
                                text_threshold=0.20
                            )
                        except Exception as e:
                            print(f"Ignoro questo batch a causa di un errore SAM2: {e}")
                            batch_results = []
                
                for batch_idx, result in enumerate(batch_results):
                    crop_box = batch_crops[batch_idx]
                    left, upper, right, lower = crop_box
                    
                    masks = result.get("masks", [])
                    scores = result.get("scores", [])
                    
                    for m_idx, mask_tensor in enumerate(masks):
                        score = float(scores[m_idx]) if len(scores) > m_idx else 1.0
                        
                        if hasattr(mask_tensor, "cpu"):
                            mask_np = (mask_tensor.cpu().numpy() * 255).astype(np.uint8)
                        else:
                            mask_np = (mask_tensor * 255).astype(np.uint8)
                            
                        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        for contour in contours:
                            # Abbassato a 200 per non scartare api piccole
                            if cv2.contourArea(contour) < 600: 
                                continue
                            
                            if cv2.contourArea(contour) > 0.25 * (right-left) * (lower-upper):
                                continue
                                
                            epsilon = 0.005 * cv2.arcLength(contour, True)
                            approx = cv2.approxPolyDP(contour, epsilon, True)
                            flat_contour = approx.flatten()
                            
                            xs = flat_contour[0::2]
                            ys = flat_contour[1::2]
                            
                            touches_artificial_border = False
                            if left > 0 and min(xs) <= 2: touches_artificial_border = True
                            if right < img_w and max(xs) >= (right-left) - 3: touches_artificial_border = True
                            if upper > 0 and min(ys) <= 2: touches_artificial_border = True
                            if lower < img_h and max(ys) >= (lower-upper) - 3: touches_artificial_border = True
                                
                            if touches_artificial_border:
                                continue
                                
                            global_contour = []
                            for p_i in range(len(flat_contour)):
                                if p_i % 2 == 0:
                                    global_contour.append(flat_contour[p_i] + left)
                                else:
                                    global_contour.append(flat_contour[p_i] + upper)
                                    
                            all_contours.append(global_contour)
                            all_scores.append(score)
                
                try: del batch_results, batch_tiles, batch_prompts
                except NameError: pass
                gc.collect()
            
            if len(all_contours) > 0:
                boxes_list = []
                for contour in all_contours:
                    xs = contour[0::2]
                    ys = contour[1::2]
                    boxes_list.append([min(xs), min(ys), max(xs), max(ys)])
                    
                boxes_tensor = torch.tensor(boxes_list, dtype=torch.float32)
                scores_tensor = torch.tensor(all_scores, dtype=torch.float32)
                keep_indices = torchvision.ops.nms(boxes_tensor, scores_tensor, iou_threshold=0.7)
                
                for idx in keep_indices:
                    global_contour = all_contours[idx]
                    normalized_points = []
                    for p_i in range(len(global_contour)):
                        if p_i % 2 == 0:
                            normalized_points.append(max(0.0, min(1.0, global_contour[p_i] / img_w)))
                        else:
                            normalized_points.append(max(0.0, min(1.0, global_contour[p_i] / img_h)))
                            
                    points_str = " ".join([f"{p:.6f}" for p in normalized_points])
                    f.write(f"0 {points_str}\n")
                    
        gc.collect()
        torch.cuda.empty_cache()