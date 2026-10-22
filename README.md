智慧电力设备巡检-电力行业仪表数据集，包括sf6继电器，电流表，电压表3类，共3231张，标注为xml格式，可用于目标检测训练![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3a8aeb7297024facbc3545cd150b0833.png)
1![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/dff870f1280442b68a1c843b02e94d1f.png)
1![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/00d51138a2e24e8f88976cd2b44281cb.png)
# 电力行业仪表检测数据集（SF6继电器+电流表+电压表｜3231张｜XML标注）
## 一、数据集信息表格
| 项目 | 详细内容 |
| :--- | :--- |
| 数据集名称 | 电力行业仪表检测数据集 |
| 数据总量 | **3231张** |
| 检测类别 | 3类：SF6继电器、电流表、电压表 |
| 标注格式 | **XML（VOC标准格式）** |
| 任务类型 | 目标检测 |
| 应用场景 | 电力巡检、仪表识别、自动读数、变电站智能监测 |
| 适配模型 | YOLOv8 / YOLOv5 / Faster R-CNN |

---

## 二、环境准备
```bash
conda create -n power_meter python=3.9
conda activate power_meter
pip install torch==1.9.0 torchvision==0.10.0
pip install ultralytics opencv-python pillow
```

---

## 三、数据集标准目录结构
```
power_meter/
├── Annotations/    # XML标注文件
├── JPEGImages/     # 原始图片
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

---

## 四、XML → YOLO TXT 格式转换代码
```python
import os
import xml.etree.ElementTree as ET
import cv2

# 3类仪表定义
classes = ["sf6_relay", "ammeter", "voltmeter"]

def xml_to_yolo(xml_path, img_w, img_h):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    labels = []
    for obj in root.iter("object"):
        cls_name = obj.find("name").text
        if cls_name not in classes:
            continue
        cls_id = classes.index(cls_name)
        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        # 归一化坐标
        cx = (xmin + xmax) / 2.0 / img_w
        cy = (ymin + ymax) / 2.0 / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h
        labels.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return labels

if __name__ == "__main__":
    xml_dir = "./power_meter/Annotations"
    img_dir = "./power_meter/JPEGImages"
    save_dir = "./power_meter/labels/train"
    os.makedirs(save_dir, exist_ok=True)

    for xml_file in os.listdir(xml_dir):
        if xml_file.endswith(".xml"):
            img_path = os.path.join(img_dir, xml_file[:-4] + ".jpg")
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            label = xml_to_yolo(os.path.join(xml_dir, xml_file), w, h)
            with open(os.path.join(save_dir, xml_file[:-4] + ".txt"), "w") as f:
                f.write("\n".join(label))
    print("✅ XML转YOLO格式完成！")
```

---

## 五、数据集配置文件 `power_meter.yaml`
```yaml
train: ./power_meter/images/train
val: ./power_meter/images/val

nc: 3
names:
  0: sf6_relay
  1: ammeter
  2: voltmeter
```

---

## 六、模型训练代码 `train.py`
```python
from ultralytics import YOLO

# 加载YOLOv8模型
model = YOLO("yolov8s.pt")

# 开始训练
model.train(
    data="power_meter.yaml",
    epochs=100,
    batch=16,
    imgsz=640,
    patience=10,
    device=0,
    project="power_meter_result",
    name="power_meter_detect"
)
```

---

## 七、推理检测代码 `detect.py`
```python
from ultralytics import YOLO
import cv2

# 加载训练好的模型
model = YOLO("power_meter_result/power_meter_detect/weights/best.pt")

# 单张图片测试
img = cv2.imread("test.jpg")
results = model(img)

# 保存结果
cv2.imwrite("detect_result.jpg", results[0].plot())
print("✅ 电力仪表检测完成！")
```

---

## 八、模型评估代码 `val.py`
```python
from ultralytics import YOLO

model = YOLO("power_meter_result/power_meter_detect/weights/best.pt")
metrics = model.val(data="power_meter.yaml", imgsz=640)

print("="*50)
print(f"mAP@0.5: {metrics.box.map50:.4f}")
print(f"精确率:  {metrics.box.precision.mean():.4f}")
print(f"召回率:  {metrics.box.recall.mean():.4f}")
print("="*50)
```

---

