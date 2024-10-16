import os
import shutil

ls = os.listdir('stage1_train')
imgs = []
for fol in ls:
    try:
        imgs.extend([os.path.join(os.path.join('stage1_train', fol, 'images', img)) for img in os.listdir(os.path.join('stage1_train', fol, 'images'))])
    except:
        continue
print(imgs)
for img in imgs:
    shutil.move(img, "img_n")
print("DONE")