from ultralytics import YOLO

model = YOLO("yolov8s-seg.pt")

result = model.train(data = "/home/beakhongha/YOLO_ARIS/data20_hand1400/IceCream Cup_Segmentation.v29i.yolov8/data.yaml", epochs = 1000)
