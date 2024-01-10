from ultralytics import YOLO as yolo

# model = yolo('C:/Users/bcw60/OneDrive/Desktop/mydrive/runs/segment/train45/weights/last.pt')
# results = model.train(resume=True)



# 예측
# model = yolo('C:/Users/bcw60/OneDrive/Desktop/mydrive/runs/segment/train45/weights/best.pt')
# results = model.predict(source ='C:/Users/bcw60/OneDrive/Desktop/test사진/walker', save = True)


from ultralyticsplus import YOLO, render_result

# load model
model = YOLO('keremberke/yolov8n-pcb-defect-segmentation')

# set model parameters
model.overrides['conf'] = 0.25  # NMS confidence threshold
model.overrides['iou'] = 0.45  # NMS IoU threshold
model.overrides['agnostic_nms'] = False  # NMS class-agnostic
model.overrides['max_det'] = 1000  # maximum number of detections per image

# set image
image = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'

# perform inference
results = model.predict(image)

# observe results
print(results[0].boxes)
print(results[0].masks)
render = render_result(model=model, image=image, result=results[0])
render.show()


from ultralyticsplus import YOLO, render_result

# 모델 로드
model = YOLO('HF_USERNAME/MODELNAME')

# 모델 매개변수 설정
model.overrides['conf'] = 0.25  # NMS confidence threshold
model.overrides['iou'] = 0.45  # NMS IoU threshold
model.overrides['agnostic_nms'] = False  # NMS class-agnostic
model.overrides['max_det'] = 1000  # maximum number of detections per image

# 이미지 설정
image = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'

# 추론 수행
results = model.predict(image, imgsz=640)

# 결과 분석
result = results[0]
boxes = result.boxes.xyxy  # x1, y1, x2, y2
scores = result.boxes.conf
categories = result.boxes.cls
scores = result.probs  # for classification models
masks = result.masks  # for segmentation models

# 이미지에 결과 표시
render = render_result(model=model, image=image, result=result)
render.show()

