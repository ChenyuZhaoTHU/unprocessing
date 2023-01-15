import unprocess
import matplotlib.pyplot as plt
from PIL import  Image
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
import matplotlib.image as mpimg
import cv2
import imageio



# rgb = mpimg.imread("C:/Users/纸/Downloads/unprocessing/1.jpg")
rgb = np.array(Image.open('1.jpg',).convert('RGB'))    # 打开图片转换为numpy数组
# print(np.shape(rgb))
# plt.imshow(rgb)
# plt.show()


# min_max_scaler = preprocessing.MinMaxScaler()
# for dim in range(3):
#     rgb[:,:,dim] = min_max_scaler.fit_transform(rgb[:,:,dim])
# print(rgb)

rgb = rgb/255

print(type(rgb))
rgb = tf.convert_to_tensor(rgb, dtype = tf.float32)
print(type(rgb))
image, metadata, out_raw = unprocess.unprocess(rgb)

# array = tf.Session().run(image)

# print(type(array))
out_raw = out_raw
# plt.imshow(out_raw)
# plt.show()

print(np.shape(out_raw))

print(out_raw)
np.save('image.npy', out_raw)

# img = cv2.convertScaleAbs(out_raw)
# # img = img *255
# print(np.shape(img))
# plt.imshow('img', img)
# 注意到这个函数只能显示uint8类型的数据，如果是uint16的数据请先转成uint8。否则图片显示会出现问题。**
# plt.show()
out_raw = out_raw * 1024


out_raw = out_raw.astype(np.float32)
out_raw.tofile('b.raw')
# cv2.imwrite("b.", out_raw)
# cv2.imwrite("b.jpg", out_raw)
f = np.array(Image.open('b.jpg',).convert('RGB'))    # 打开图片转换为numpy数组
# f=f*255

print(np.shape(f))
print(np.min(f))
print(f)
np.savetxt('f.txt',f[:,:,1])
plt.imshow(f)
plt.show()