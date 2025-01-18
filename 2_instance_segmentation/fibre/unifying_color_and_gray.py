import cv2
import os
import numpy as np
from PIL import Image # (pip install Pillow)


#unifying color and gray
# for filename in os.listdir("labels"):
#     if filename.endswith('.png'):
#         img = cv2.imread(os.path.join('labels', filename), cv2.IMREAD_GRAYSCALE)
#         print(type(img))
#         #img[np.all(img == (255, 255, 255), axis=-1)] = (0, 0, 0)
#         cv2.imwrite(os.path.join('labels2', os.path.splitext(filename)[0]+'.png'), img)


#2 label images
# tab 20 color dict
color_dict={
            0: (0, 0, 0),
            1: (31, 119, 180),
            2: (174, 199, 232),
            3: (255, 127, 14),
            4: (255, 187, 120),
            5: (44, 160, 44),
            6: (152, 223, 138),
            7: (214, 39, 40),
            8: (255, 152, 150),
            9: (148, 103, 189),
            10: (197, 176, 213),
            11: (140, 86, 75),
            12: (196, 156, 148),
            13: (227, 119, 194),
            14: (247, 182, 210),
            15: (127, 127, 127),
            16: (199, 199, 199),
}
print(list(color_dict.values())[0])

img_label = np.zeros((512,512))
for filename in os.listdir("labels"):
    if filename.endswith('.png'):
        img = cv2.imread(os.path.join('labels', filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for i in range(17):
            # print(np.all(img == list(color_dict.values())[i], axis=-1).tolist())
            img_label[np.all(img == list(color_dict.values())[i], axis=-1)] = i
            # print(i)
        img_label = img_label.astype(np.uint16)
        cv2.imwrite(os.path.join('labels3', os.path.splitext(filename)[0] + '.png'), img_label)
print(img_label.tolist())
# print(filename)
