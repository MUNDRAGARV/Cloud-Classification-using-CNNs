import os
import tensorflow as tf 
import cv2
import matplotlib.pyplot as plt
from bing_image_downloader import downloader

##helps prevent OOM errors
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
# reduce memory growth
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True) 
    

## downloading cloud images from the internet
downloader.download("clouds", limit=1000, output_dir="cloud_images", adult_filter_off=True, force_replace=False, timeout=60)
    
    
