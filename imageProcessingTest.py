import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

#Read both inpout and output directory paths here
input_dir = 'C:/Users/shita/OneDrive/Documents/Computer Vision-Python/Inputs'  
output_dir = 'C:/Users/shita/OneDrive/Documents/Computer Vision-Python/Outputs'   
os.makedirs(output_dir, exist_ok=True)

def process_image(file_path):
    image = cv2.imread(file_path)
    
    # Creating a binary mask where channels are above 200
    mask = np.all(image > 200, axis=-1).astype(np.uint8) * 255  

    # Create output path for the mask
    file_name = os.path.splitext(os.path.basename(file_path))[0] + '_mask.png'
    output_path = os.path.join(output_dir, file_name)
    
    cv2.imwrite(output_path, mask)
    
    # Count the number of pixels where the mask is max (255)
    count_max_pixels = np.sum(mask == 255)
    
    return count_max_pixels

def main():
    # Get list of all jpg files in the input directory
    image_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.jpg')]
    
    total_max_pixels = 0
     
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_image, file) for file in image_files]
        
        for future in as_completed(futures):
            try:
                total_max_pixels += future.result()
            except Exception as exc:
                print(f"Error occurred: {exc}")
    
    print(f"Total number of max pixels across all masks: {total_max_pixels}")

if __name__ == "__main__":
    main()
