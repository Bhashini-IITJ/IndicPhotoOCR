import sys
import os
import json
import cv2
import numpy as np

imageName = sys.argv[1]
jsonPath = sys.argv[2]

with open(jsonPath) as f:
    data = json.load(f)

key_name = imageName.split("/")[-2] + "_" + imageName.split("/")[-1].split(".")[0]
value = data[key_name]['annotations']
print(value)

# read the image and plot the polygons
image = cv2.imread(imageName)

for poly_id, coords in value.items():
    coords = coords['coordinates']
    coords = [tuple(coord) for coord in coords]
    # print(coords)
    cv2.polylines(image, [np.array(coords)], True, (0, 255, 0), 2)

cv2.imshow("image", image)
cv2.waitKey(0)