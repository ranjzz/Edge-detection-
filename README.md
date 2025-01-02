# Edge-detection-
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

!pip install ultralytics
from ultralytics import YOLO
Collecting ultralytics
  Downloading ultralytics-8.2.92-py3-none-any.whl.metadata (41 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 41.9/41.9 kB 637.0 kB/s eta 0:00:00
Requirement already satisfied: numpy<2.0.0,>=1.23.0 in /opt/conda/lib/python3.10/site-packages (from ultralytics) (1.26.4)
Requirement already satisfied: matplotlib>=3.3.0 in /opt/conda/lib/python3.10/site-packages (from ultralytics) (3.7.5)
Requirement already satisfied: opencv-python>=4.6.0 in /opt/conda/lib/python3.10/site-packages (from ultralytics) (4.10.0.84)
Requirement already satisfied: pillow>=7.1.2 in /opt/conda/lib/python3.10/site-packages (from ultralytics) (9.5.0)
Requirement already satisfied: pyyaml>=5.3.1 in /opt/conda/lib/python3.10/site-packages (from ultralytics) (6.0.2)
Requirement already satisfied: requests>=2.23.0 in /opt/conda/lib/python3.10/site-packages (from ultralytics) (2.32.3)
Requirement already satisfied: scipy>=1.4.1 in /opt/conda/lib/python3.10/site-packages (from ultralytics) (1.14.0)
Requirement already satisfied: torch>=1.8.0 in /opt/conda/lib/python3.10/site-packages (from ultralytics) (2.4.0)
Requirement already satisfied: torchvision>=0.9.0 in /opt/conda/lib/python3.10/site-packages (from ultralytics) (0.19.0)
Requirement already satisfied: tqdm>=4.64.0 in /opt/conda/lib/python3.10/site-packages (from ultralytics) (4.66.4)
Requirement already satisfied: psutil in /opt/conda/lib/python3.10/site-packages (from ultralytics) (5.9.3)
Requirement already satisfied: py-cpuinfo in /opt/conda/lib/python3.10/site-packages (from ultralytics) (9.0.0)
Requirement already satisfied: pandas>=1.1.4 in /opt/conda/lib/python3.10/site-packages (from ultralytics) (2.2.2)
Requirement already satisfied: seaborn>=0.11.0 in /opt/conda/lib/python3.10/site-packages (from ultralytics) (0.12.2)
Collecting ultralytics-thop>=2.0.0 (from ultralytics)
  Downloading ultralytics_thop-2.0.6-py3-none-any.whl.metadata (9.1 kB)
Requirement already satisfied: contourpy>=1.0.1 in /opt/conda/lib/python3.10/site-packages (from matplotlib>=3.3.0->ultralytics) (1.2.1)
Requirement already satisfied: cycler>=0.10 in /opt/conda/lib/python3.10/site-packages (from matplotlib>=3.3.0->ultralytics) (0.12.1)
Requirement already satisfied: fonttools>=4.22.0 in /opt/conda/lib/python3.10/site-packages (from matplotlib>=3.3.0->ultralytics) (4.53.0)
Requirement already satisfied: kiwisolver>=1.0.1 in /opt/conda/lib/python3.10/site-packages (from matplotlib>=3.3.0->ultralytics) (1.4.5)
Requirement already satisfied: packaging>=20.0 in /opt/conda/lib/python3.10/site-packages (from matplotlib>=3.3.0->ultralytics) (21.3)
Requirement already satisfied: pyparsing>=2.3.1 in /opt/conda/lib/python3.10/site-packages (from matplotlib>=3.3.0->ultralytics) (3.1.2)
Requirement already satisfied: python-dateutil>=2.7 in /opt/conda/lib/python3.10/site-packages (from matplotlib>=3.3.0->ultralytics) (2.9.0.post0)
Requirement already satisfied: pytz>=2020.1 in /opt/conda/lib/python3.10/site-packages (from pandas>=1.1.4->ultralytics) (2024.1)
Requirement already satisfied: tzdata>=2022.7 in /opt/conda/lib/python3.10/site-packages (from pandas>=1.1.4->ultralytics) (2024.1)
Requirement already satisfied: charset-normalizer<4,>=2 in /opt/conda/lib/python3.10/site-packages (from requests>=2.23.0->ultralytics) (3.3.2)
Requirement already satisfied: idna<4,>=2.5 in /opt/conda/lib/python3.10/site-packages (from requests>=2.23.0->ultralytics) (3.7)
Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/conda/lib/python3.10/site-packages (from requests>=2.23.0->ultralytics) (1.26.18)
Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/lib/python3.10/site-packages (from requests>=2.23.0->ultralytics) (2024.7.4)
Requirement already satisfied: filelock in /opt/conda/lib/python3.10/site-packages (from torch>=1.8.0->ultralytics) (3.15.1)
Requirement already satisfied: typing-extensions>=4.8.0 in /opt/conda/lib/python3.10/site-packages (from torch>=1.8.0->ultralytics) (4.12.2)
Requirement already satisfied: sympy in /opt/conda/lib/python3.10/site-packages (from torch>=1.8.0->ultralytics) (1.13.2)
Requirement already satisfied: networkx in /opt/conda/lib/python3.10/site-packages (from torch>=1.8.0->ultralytics) (3.3)
Requirement already satisfied: jinja2 in /opt/conda/lib/python3.10/site-packages (from torch>=1.8.0->ultralytics) (3.1.4)
Requirement already satisfied: fsspec in /opt/conda/lib/python3.10/site-packages (from torch>=1.8.0->ultralytics) (2024.6.1)
Requirement already satisfied: six>=1.5 in /opt/conda/lib/python3.10/site-packages (from python-dateutil>=2.7->matplotlib>=3.3.0->ultralytics) (1.16.0)
Requirement already satisfied: MarkupSafe>=2.0 in /opt/conda/lib/python3.10/site-packages (from jinja2->torch>=1.8.0->ultralytics) (2.1.5)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in /opt/conda/lib/python3.10/site-packages (from sympy->torch>=1.8.0->ultralytics) (1.3.0)
Downloading ultralytics-8.2.92-py3-none-any.whl (871 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 871.9/871.9 kB 6.1 MB/s eta 0:00:00
Downloading ultralytics_thop-2.0.6-py3-none-any.whl (26 kB)
Installing collected packages: ultralytics-thop, ultralytics
Successfully installed ultralytics-8.2.92 ultralytics-thop-2.0.6
# Set the device to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device is on: {device}')
Device is on: cuda
model = YOLO("/kaggle/input/yolo-v8/pytorch/yolov8s/1/yolov8s.pt").to(device)
def load_image(image_path):
    image = cv2.imread(image_path)
    # image = cv2.resize(image, (500, 500))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image, gray_image

image, gray = load_image('/kaggle/input/pothole-detection-dataset/normal/102.jpg')
results = model(image)
img = results[0].plot()  
plt.imshow(img)
plt.show()
0: 448x640 9 cars, 3 trucks, 44.8ms
Speed: 11.1ms preprocess, 44.8ms inference, 435.6ms postprocess per image at shape (1, 3, 448, 640)

def plot_histograms(results, gray, num_cols=4):
    num_objects = len(results[0].boxes.xyxy)
    num_rows = int(np.ceil(num_objects / num_cols))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 3 * num_rows))
    axes = axes.flatten()

    for i, box in enumerate(results[0].boxes.xyxy):
        x1, y1, x2, y2 = map(int, box)
        object_pixels = gray[y1:y2, x1:x2]
        hist = cv2.calcHist([object_pixels], [0], None, [256], [0, 256])
        axes[i].plot(hist)
        axes[i].set_title(f'Object {i + 1}', fontweight='bold')
        axes[i].set_xlabel('Pixel Value')
        axes[i].set_ylabel('Frequency')
        axes[i].set_xlim([0, 256])

    for j in range(num_objects, len(axes)):
        axes[j].remove()
    
    fig.suptitle('Gray Scales of Objects', fontsize=32, fontweight='bold')
    plt.tight_layout()

plot_histograms(results, gray)
