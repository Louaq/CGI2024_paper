import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('D:/Downloads/YOLOv8/result/result_15_slimNeck/train/exp/weights/best.pt')
    model.val(data='D:/Downloads/YOLOv8/datasets/data.yaml',
              split='val',
              imgsz=640,
              batch=4,
              # rect=False,
              # save_json=True, # 这个保存coco精度指标的开关
              project='runs/val',
              name='exp',
              )