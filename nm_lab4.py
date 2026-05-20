!pip install ultralytics
import ultralytics
ultralytics.checks()

import yaml
import os

# Створення директорії для розміщення датасету
os.makedirs('/content/datasets', exist_ok=True)

# Формування конфігураційного словника для 10 базових класів COCO
data_config = {
    'path': '/content/datasets/coco128',
    'train': 'images/train2017',
    'val': 'images/train2017',
    'download': 'https://ultralytics.com/assets/coco128.zip',
    'names': {
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
        5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light'
    }
}

# Експорт конфігурації у формат YAML
with open('/content/coco10.yaml', 'w') as f:
    yaml.dump(data_config, f, sort_keys=False)

print("Конфігураційний файл coco10.yaml успішно згенеровано!")

from ultralytics import YOLO

# Ініціалізація базової архітектури YOLOv8n (nano) "з нуля"
model_n = YOLO('yolov8n.yaml')

# Запуск процесу навчання моделі
results_n = model_n.train(
    data='/content/coco10.yaml',
    epochs=15,
    imgsz=640,
    classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    device=0
)

from ultralytics import YOLO

# Завантажуємо архітектуру YOLOv8s (small) з нуля
model_s = YOLO('yolov8s.yaml')

# Запуск навчання з абсолютно ідентичними параметрами для чесного порівняння
results_s = model_s.train(
    data='/content/coco10.yaml',
    epochs=15,
    imgsz=640,
    classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    device=0
)

# Завантаження надійних тестових файлів 
!wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/dog.jpg -O photo1.jpg
!wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/kite.jpg -O photo2.jpg
!wget https://github.com/intel-iot-devkit/sample-videos/raw/master/people-detection.mp4 -O video1.mp4

# Детекція
from ultralytics import YOLO

# Завантажуємо вже натреновану "розумну" модель
smart_model = YOLO('yolov8s.pt')

print("\n--- Починаємо детекцію ---")
res_photo1 = smart_model.predict(source='/content/photo1.jpg', save=True, conf=0.25)
res_photo2 = smart_model.predict(source='/content/photo2.jpg', save=True, conf=0.25)
res_video = smart_model.predict(source='/content/video1.mp4', save=True, conf=0.25)

from ultralytics import YOLO

# Завантажуємо вже натреновану "розумну" модель
smart_model = YOLO('yolov8s.pt')

print("починаємо детекцію")
# Детектуємо ТІЛЬКИ ті файли, що вже лежать у папці
smart_model.predict(source=['test1.jpg', 'test2.jpg'], save=True, conf=0.25)

print("детекцію завершено, файли збережено у runs/detect/predict")