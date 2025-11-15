from ultralytics import YOLO
import cv2
from tkinter import Tk, filedialog
import sys
import numpy as np

if __name__ == '__main__':
    model = YOLO("../runs/detect/mm_detector3/weights/best.pt")
    
    Tk().withdraw()
    image_path = filedialog.askopenfilename(
        title="pick an image file",
        filetypes=[("Images", "*.jpg *.jpeg *.png")]
    )
    
    if not image_path:
        print("No image selected. Exiting.")
        sys.exit(1)
    
    results = model.predict(image_path, conf=0.5)
    
    img = cv2.imread(image_path)
   
    # box colors for each M&M color
    colors = {
        'blue': (255, 0, 0),      
        'brown': (42, 42, 165),   
        'green': (0, 255, 0),     
        'orange': (0, 165, 255),  
        'red': (0, 0, 255),       
        'yellow': (0, 255, 255)   
    }
    
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        class_id = int(box.cls[0])
        class_name = results[0].names[class_id]
        color = colors.get(class_name, (255, 255, 255))
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        
        cv2.putText(img, class_name, (x1, y1-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    
    cv2.imshow('M&M Detection', img)
    
    print(f"\nFound {len(results[0].boxes)} M&Ms!")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
