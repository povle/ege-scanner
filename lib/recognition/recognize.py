import tensorflow as tf
import cv2
import os
import numpy as np

package_directory = os.path.dirname(os.path.abspath(__file__))

class_indices = {'cyrillic': ['1', '2', '3', '4', '5', '6', '7', '8', '9',
                              'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ё', 'Ж', 'З',
                              'И', 'Й', 'К', 'Л', 'М', 'Н', 'О', 'П', 'Р',
                              'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ',
                              'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я'],
                }

class Recognizer:
    def __init__(self, model_path, indices):
        self.model = tf.keras.models.load_model(model_path)
        self.indices = indices
        self.image_size = self.model.get_input_shape_at(0)[1:3]

    def recognize(self, img):
        data = self.convert_cv2_to_tf(img)
        result = self.model.predict(data)[0]
        best = np.argmax(result)
        return self.indices[best]

    def convert_cv2_to_tf(self, img):
        resized = cv2.resize(img, self.image_size)
        processed = resized / 255
        tensor = tf.convert_to_tensor(processed, dtype=tf.float32)
        return tf.reshape(tensor, (1, *self.image_size, 1))

class CyrillicRecognizer(Recognizer):
    def __init__(self):
        super().__init__(os.path.join(package_directory, 'models/cyrillic'),
                         class_indices['cyrillic'])
