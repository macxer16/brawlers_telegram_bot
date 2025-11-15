from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

def get_class(model_path, labels_path, image_path):
    np.set_printoptions(suppress=True)  # отключаем научную нотацию
    
    # Загружаем модель
    model = load_model(model_path, compile=False)
    
    # Загружаем метки классов с указанием кодировки
    class_names = open(labels_path, "r", encoding="utf-8").readlines()
    
    # Создаем массив нужной формы для подачи в модель
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    
    # Загружаем и преобразуем изображение
    image = Image.open(image_path).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    
    # Нормализуем изображение
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array
    
    # Делаем предсказание
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    
    # Возвращаем имя класса и уверенность
    return (class_name[2:], confidence_score)