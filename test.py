from PIL import Image
import os

path = []
for img in os.listdir("datasets/selfie2anime/testA"):
    #img_tensor = Image.open(os.path.join("datasets/selfie2anime/testA",img)).convert('RGB')
    #print(img_tensor.shape)
    path.append(os.path.join("datasets/selfie2anime/testA",img))
print("num_image",len(path))
img_tensor = Image.open(path).convert('RGB')