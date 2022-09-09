import cv2
import math
import glob
import time
import numpy as np
import random as rnd
import matplotlib.pyplot as plt


def display_images(images, **kwargs):
    plt.figure(figsize=(20, 20))
    plt.axis('off')
    num_images = len(images)
    for i in range(num_images):
        ax = plt.subplot(math.ceil(num_images/2), 2, i+1)
        # ax.axis('off')
        ax.imshow(images[i], cmap=plt.gray(), **kwargs)
    plt.show()

def count_pixels(patch, hue_margin=10, saturation_th=100, lightness_margin=10):
    total = patch.shape[0] * patch.shape[1]
    patch_hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    patch_hsl = cv2.cvtColor(patch, cv2.COLOR_RGB2HLS)

    whites = (np.logical_and(patch_hsv[:, :, 1] <= 40, patch_hsv[:, :, 2] >= 10)).sum()
    reds = (np.logical_and(np.logical_or(patch_hsv[:, :, 0] <= 10, patch_hsv[:, :, 0] >= 170), patch_hsv[:, :, 1] >= 150)).sum()
    blues = (np.logical_and(np.logical_and(patch_hsv[:, :, 0] <= 115+hue_margin, patch_hsv[:, :, 0] >= 115-hue_margin), patch_hsv[:, :, 1] >= saturation_th)).sum()

    return total, whites, reds, blues

def detect_triangular_signs(image, edge, rb, iamge_num, save_path):
    
    ROI_number = 0
    cnts = cv2.findContours(rb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    final_signs = []

    height, width = image.shape[:2]

    for cnt in cnts: 
      approx = cv2.approxPolyDP(cnt,0.25*cv2.arcLength(cnt,True),True)
      if len(approx) == 3:
        if cv2.contourArea(cnt) >= 500:
          x,y,w,h = cv2.boundingRect(cnt)
          margin = int(w)
          ymin = max(0, y-margin)
          ymax = min(height, y+h+margin)
          xmin = max(0, x-margin)
          xmax = min(width, x+w+margin)
          final_signs.append(image[ymin:ymax, xmin:xmax])

    for i, s in enumerate(final_signs):
        cv2.imwrite(f'{save_path}/{iamge_num}-tri-{i}.png', s[:, :, ::-1])
    return final_signs

def detect_circular_signs(image, edge, rb, iamge_num, save_path):

    circles = cv2.HoughCircles(edge,
                                cv2.HOUGH_GRADIENT,
                                1,
                                20,
                                param1 = 10,
                                param2 = 20,
                                minRadius = 0, # TODO: find the right parameters
                                maxRadius = 0)
    
    h, w = image.shape[:2]
    cop = image.copy()
    final_signs = []
    if circles is not None:
        
        circles = np.round(circles[0, :]).astype("int")

        for (x, y, r) in circles:
            sub_area = cop[y-r:y+r, x-r:x+r, :]
            
            if sub_area.shape[0] * sub_area.shape[1] == 0:
                continue

            total, whites, reds, blues = count_pixels(sub_area)

            area = 3.141592 * r * r
            margin = int(r)

            if (blues/total > 0.3) and (whites/total > 0.15) and (area > 500) and (area < 6000):
                cv2.circle(cop, (x, y), r, (0, 255, 0), 2) 
                ymin = max(0, y-r-margin)
                ymax = min(h, y+r+margin)
                xmin = max(0, x-r-margin)
                xmax = min(w, x+r+margin)
                x = image[ymin:ymax, xmin:xmax]
                final_signs.append(x)

            elif (reds/total > 0.2) and (whites/total > 0.3) and (area > 500) and (area < 6000):
                cv2.circle(cop, (x, y), r, (0, 255, 0), 2)
                ymin = max(0, y-r-margin)
                ymax = min(h, y+r+margin)
                xmin = max(0, x-r-margin)
                xmax = min(w, x+r+margin)
                x = image[ymin:ymax, xmin:xmax]
                final_signs.append(x)

    for i, s in enumerate(final_signs):
        cv2.imwrite(f'{save_path}/{iamge_num}-circ-{i}.png', s[:, :, ::-1])
    return final_signs

def detect_traffic_signs(images, save_path):
    
    triangles = []
    circles = []
    
    for i, image in enumerate(images):
        
        tick = time.time()

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        r = np.logical_and(np.logical_or(hsv[:, :, 0] <= 10, hsv[:, :, 0] >= 170), hsv[:, :, 1] >= 100)
        b = np.logical_and(np.logical_and(hsv[:, :, 0] <= 135, hsv[:, :, 0] >= 105), hsv[:, :, 1] >= 100)
        rb = np.logical_or(r, b)
        rb = (rb * 1.0).astype('uint8') 

        """closing"""
        rb = cv2.morphologyEx(rb, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))


        edge = cv2.Canny(rb, 0, 1)
        triangles.extend(detect_triangular_signs(image, edge, rb, i, save_path))
        circles.extend(detect_circular_signs(image, edge, rb, i, save_path))

        tock = time.time()
        print(f'Image {i} Processed in {tock-tick:.2f} seconds')

    signs = []
    signs.extend(triangles)
    signs.extend(circles)
    return signs


def main():

    images = []
    paths = sorted(glob.glob('TestIJCNN2013Download/*.ppm'))
    for p in paths[:]:
        image = cv2.imread(p)[:, :, ::-1]
        images.append(image)

    signs = detect_traffic_signs(images, './Detected4')

if __name__ == '__main__':
    main()