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

image_dir = 'bee_dataset/images/val'
label_dir = 'bee_dataset/labels/val'
os.makedirs(label_dir, exist_ok=True)
text_prompt = "bee."

# desired bee size in pixels for the segmentation model
TARGET_BEE_SIZE = 120.0 

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
    # downsample heavily for a fast detection pass
    scan_pil = original_pil.copy()
    scan_pil.thumbnail((800, 800), Image.Resampling.LANCZOS)
    
    # ratio to map scan coordinates back to original image
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
        
        # calculate width and height for all detected boxes mapping them to original resolution
        if hasattr(boxes, "cpu"):
            boxes = boxes.cpu().numpy()
            
        sizes = []
        for box in boxes:
            # box format usually [x1, y1, x2, y2]
            w = (box[2] - box[0]) * ratio_w
            h = (box[3] - box[1]) * ratio_h
            sizes.append(max(w, h)) 
            
        # use median to ignore giant false positives or tiny artifacts
        median_bee_size = float(np.median(sizes))
        
        if median_bee_size > 0:
            scale_factor = TARGET_BEE_SIZE / median_bee_size
            # clamp scale factor to avoid memory explosions or microscopic downscaling
            scale_factor = max(0.3, min(scale_factor, 3.0)) 
            
    # clean memory after scan
    try: del quick_results, boxes
    except NameError: pass
    gc.collect()
    torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # PASS 2: APPLY ZOOM AND DYNAMIC TILING
    # ---------------------------------------------------------
    new_w = int(original_pil.width * scale_factor)
    new_h = int(original_pil.height * scale_factor)
    image_pil = original_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    img_w, img_h = image_pil.size
    
    # generate dynamic crop grid (ridotto a 500x500 per non far esplodere SAM2 di scatole)
    # this prevents VRAM crashes if scale_factor makes the image huge
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
            
            # Se il ritaglio sul bordo finale è minuscolo (es. largo 1 pixel), ignoralo!
            # (Tanto è già coperto abbondantemente dal ritaglio precedente grazie all'overlap)
            if x2 - x1 < 50 or y2 - y1 < 50:
                continue
                
            crops.append((x1, y1, x2, y2))
            
    all_contours = []
    all_scores = []
    
    with open(label_path, 'w') as f:
        for crop_box in crops:
            left, upper, right, lower = crop_box
            tile_pil = image_pil.crop(crop_box)
            
            torch.cuda.empty_cache()
            
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                with torch.no_grad():
                    try:
                        results = model.predict(
                            [tile_pil], 
                            [text_prompt], 
                            box_threshold=0.30,
                            text_threshold=0.30
                        )
                    except Exception as e:
                        print(f"Ignoro questo ritaglio a causa di un errore interno di SAM2: {e}")
                        results = []
            
            if len(results) == 0:
                try: del results
                except: pass
                gc.collect()
                continue
                
            result = results[0]
            masks = result.get("masks", [])
            scores = result.get("scores", [])
            
            if len(masks) == 0:
                try: del results, result, masks, scores
                except: pass
                gc.collect()
                continue
                
            for idx, mask_tensor in enumerate(masks):
                score = float(scores[idx]) if len(scores) > idx else 1.0
                
                if hasattr(mask_tensor, "cpu"):
                    mask_np = (mask_tensor.cpu().numpy() * 255).astype(np.uint8)
                else:
                    mask_np = (mask_tensor * 255).astype(np.uint8)
                    
                contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    # adjusted noise filter to account for the normalized zoom
                    if cv2.contourArea(contour) < 1000: # Alzato a 3000 per polverizzare i falsi positivi molto piccoli
                        continue
                    
                    if cv2.contourArea(contour) > 0.25 * tile_pil.width * tile_pil.height:
                        continue
                        
                    epsilon = 0.005 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    flat_contour = approx.flatten()
                    
                    xs = flat_contour[0::2]
                    ys = flat_contour[1::2]
                    
                    # discard artificial cut borders
                    touches_artificial_border = False
                    if left > 0 and min(xs) <= 2: touches_artificial_border = True
                    if right < img_w and max(xs) >= tile_pil.width - 3: touches_artificial_border = True
                    if upper > 0 and min(ys) <= 2: touches_artificial_border = True
                    if lower < img_h and max(ys) >= tile_pil.height - 3: touches_artificial_border = True
                        
                    if touches_artificial_border:
                        continue
                        
                    # map local tile coordinates -> global scaled coordinates
                    global_contour = []
                    for i in range(len(flat_contour)):
                        if i % 2 == 0:
                            global_contour.append(flat_contour[i] + left)
                        else:
                            global_contour.append(flat_contour[i] + upper)
                            
                    all_contours.append(global_contour)
                    all_scores.append(score)
            
            # aggressive memory cleanup
            try: del results, result, masks, scores, mask_tensor, mask_np
            except NameError: pass
            gc.collect()
            torch.cuda.empty_cache()
            
        # NMS to merge tiles
        if len(all_contours) > 0:
            boxes_list = []
            for contour in all_contours:
                xs = contour[0::2]
                ys = contour[1::2]
                boxes_list.append([min(xs), min(ys), max(xs), max(ys)])
                
            boxes_tensor = torch.tensor(boxes_list, dtype=torch.float32)
            scores_tensor = torch.tensor(all_scores, dtype=torch.float32)
            keep_indices = torchvision.ops.nms(boxes_tensor, scores_tensor, iou_threshold=0.8)
            
            for idx in keep_indices:
                global_contour = all_contours[idx]
                normalized_points = []
                for i in range(len(global_contour)):
                    # normalize using the zoomed image dimensions
                    if i % 2 == 0:
                        normalized_points.append(max(0.0, min(1.0, global_contour[i] / img_w)))
                    else:
                        normalized_points.append(max(0.0, min(1.0, global_contour[i] / img_h)))
                        
                points_str = " ".join([f"{p:.6f}" for p in normalized_points])
                f.write(f"0 {points_str}\n")
                
    gc.collect()
    torch.cuda.empty_cache()