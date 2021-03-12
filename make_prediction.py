import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2

#### these are images I personally took myself, I also made it to the output image looks similar to
#### what we have in the dataset

# gray_img = cv2.imread('images/myimg.jpg', 0)
# gray_img[np.where(gray_img < [200])] = [255]
# num_img = gray_img

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

EXAMPLE = 30  # CHANGE THIS
num_img = x_test[EXAMPLE]  # comment this, and uncomment code above to use the images I took
plt.imshow(num_img)
plt.show()

loaded_model = tf.keras.models.load_model('my_model.h5')
prediction = loaded_model.predict(np.reshape(num_img, (1, 28 * 28)))
print(f"predicted number: {np.argmax(prediction)}")
