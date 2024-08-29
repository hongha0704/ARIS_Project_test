from ultralytics import YOLO

model = YOLO("yolov8s-seg.pt")

result = model.train(data = "/home/beakhongha/YOLO_ARIS/data25/IceCream Cup_Segmentation.v33i.yolov8/data.yaml", epochs = 500)
