# Deploy YOLOv8 Pre-Trained Model using ONNX Runtime for Object Detection

![! ONNX YOLOv8 Object Detection](https://github.com/tirta-ir/KDU-CV-YOLOv8-ONNX-Object-Detection/raw/main/img/example-image.jpg)
*Original image: [https://images.pexels.com/photos/2422290/pexels-photo-2422290.jpeg?cs=srgb&dl=pexels-jopwell-2422290.jpg&fm=jpg](https://images.pexels.com/photos/2422290/pexels-photo-2422290.jpeg?cs=srgb&dl=pexels-jopwell-2422290.jpg&fm=jpg)*

# Creating ONNX File
You can use this code to convert the model. The author used pre-trained model from ultralytics and convert it into ONNX file using this code:

```python
from ultralytics import YOLO
model_name = 'yolov8m' #@param ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]
input_width = 640 #@param {type:"slider", min:32, max:4096, step:32}
input_height = 480 #@param {type:"slider", min:32, max:4096, step:32}
optimize_cpu = False

model = YOLO(f"{model_name}.pt")
model.export(format="onnx", imgsz=[input_height,input_width], optimize=optimize_cpu)
```

# Requirements

 * Check the **requirements.txt** file.
 * For ONNX runtime, if you have a NVIDIA GPU, then install the **onnxruntime-gpu**, otherwise use the **onnxruntime** library.

# Installation
```shell
git clone https://github.com/tirta-ir/KDU-CV-YOLOv8-ONNX-Object-Detection.git
cd KDU-CV-YOLOv8-ONNX-Object-Detection
pip install -r requirements.txt
```

For Nvidia GPU user:
`pip install onnxruntime-gpu`

Otherwise:
`pip install onnxruntime`

# Examples Video

 * **Image inference**: Look at the first picture on this readme file.

 * **Video inference**: https://youtu.be/wsrlBjgGsFQ

 * **Webcam inference**: https://youtu.be/8JdKoYaRMVQ

# References:
* YOLOv8 model: [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
* ONNX Runtime Example: [https://github.com/ibaiGorordo/ONNX-YOLOv8-Object-Detection](https://github.com/ibaiGorordo/ONNX-YOLOv8-Object-Detection)
