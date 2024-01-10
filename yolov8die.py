from ultralytics import YOLO as yolo

# model = yolo('C:/Users/bcw60/OneDrive/Desktop/mydrive/runs/segment/train45/weights/last.pt')
# results = model.train(resume=True)



# 예측
model = yolo('C:/Users/bcw60/OneDrive/Desktop/mydrive/runs/segment/train45/weights/best.pt')
results = model.predict(source ='C:/Users/bcw60/OneDrive/Desktop/test사진/walker', save = True)

