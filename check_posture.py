import cv2
import os
import pickle
import pandas as pd

from landmark_plotting import plot_pose

#  de-serializing the model using pickle

model=pickle.load(open("KNN_POSTURE_MODEL.pkl","rb"))
le=pickle.load(open("label_encoder.pkl","rb"))



def predict_pose(img):
    data={}
    for i in range(1,34):
        data[f"x{i}"]=[]
        data[f"y{i}"]=[]
        data[f"z{i}"]=[]
        data[f"visibility{i}"]=[]
    try:
        img_copy=img.copy()
        img_copy,landmarks=plot_pose(img_copy)
        if landmarks is not None:
            for index,a in enumerate(landmarks,1):
                data[f"x{index}"].append(a.x)
                data[f"y{index}"].append(a.y)
                data[f"z{index}"].append(a.z)
                data[f"visibility{index}"].append(a.visibility)
        data=pd.DataFrame(data).values
        y_pred=model.predict(data)
        predicted_class=le.inverse_transform(y_pred)[0]
        return img_copy,predicted_class
        
    except Exception as e:
        print(e)
        return img,"NO ONE IN FRAME"
        
cap=cv2.VideoCapture(0)

while True:
    _,frame=cap.read()
    frame,prediction=predict_pose(frame)
    cv2.putText(frame,prediction,(30,30),cv2.FONT_ITALIC,2,(0,0,0),2)
    cv2.imshow("frame",frame)
    if cv2.waitKey(1) == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
    