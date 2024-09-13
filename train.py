from multiprocessing import freeze_support

from ultralytics import YOLO

def train():
     # Load a model
     model = YOLO("yolov8n-cls.yaml")  # build a new model from YAML
     model = YOLO("yolov8n-cls.pt")  # load a pretrained model (recommended for training)
     model = YOLO("yolov8n-cls.yaml").load("yolov8n-cls.pt")  # build from YAML and transfer weights

     # Train the model
     results = model.train(data="./dataset/", epochs=15, imgsz=128)


if __name__ == '__main__':
     train()
     freeze_support()