import tensorflow as tf
import cv2
import numpy as np
import os
import random

#file location
input_folder = "/Users/shadow/Desktop/frog"
output_folder = "/Users/shadow/Desktop/augmentedfrogs/"

os.makedirs(output_folder, exist_ok=True)

def augment_and_save(img_array, file_name, num_augments=5):

    sharpen_kernel = np.array([
        [0, -0.9, 0],
        [-0.9, 5, -0.9],
        [0, -0.9, 0]
    ])

    (h, w) = img_array.shape[:2]
    center = (w // 2, h // 2)

    def hflip(x): return tf.image.flip_left_right(x).numpy()
    def vflip(x): return tf.image.flip_up_down(x).numpy()
    def blur(x): return cv2.GaussianBlur(x, (5, 5), 0)
    def rot_plus15(x):
        M = cv2.getRotationMatrix2D(center, 15, 1.0)
        return cv2.warpAffine(x, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    def rot_minus15(x):
        M = cv2.getRotationMatrix2D(center, -15, 1.0)
        return cv2.warpAffine(x, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    def noisy(x):
        mean, stddev = 0, 11
        noise = np.random.normal(mean, stddev, x.shape).astype("int16")
        return np.clip(x.astype("int16") + noise, 0, 255).astype("uint8")
    def high_contrast(x): return cv2.convertScaleAbs(x, alpha=1.2, beta=0)
    def low_contrast(x): return cv2.convertScaleAbs(x, alpha=0.8, beta=0)

    transforms = {
        "hflip": hflip,
        "vflip": vflip,
        "blur": blur,
        "rot+15": rot_plus15,
        "rot-15": rot_minus15,
        "noisy": noisy,
        "contrast+": high_contrast,
        "contrast-": low_contrast,
    }

    chosen = random.sample(list(transforms.items()), num_augments)

    base_name = os.path.splitext(file_name)[0]
    for aug_name, aug_func in chosen:
        aug_img = aug_func(img_array)
        save_path = os.path.join(output_folder, f"{base_name}_{aug_name}.jpg")
        cv2.imwrite(save_path, cv2.cvtColor(aug_img.astype("uint8"), cv2.COLOR_RGB2BGR))

for file in os.listdir(input_folder):
    if file.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(input_folder, file)
        
        img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
        img_array = tf.keras.utils.img_to_array(img)

        #Number of augmentation you want
        augment_and_save(img_array, file, num_augments=8)

print(f"Augmented images saved in: {output_folder}")
