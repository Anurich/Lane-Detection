import torch
from model import *
import pickle
import matplotlib.pyplot as plt
import numpy as np
import cv2

def test():
    with torch.no_grad():
        cap = cv2.VideoCapture('IMG_0310.MOV')
        net = model()
        net.eval()
        load_weight = torch.load("weights/model.pth")
        net.load_state_dict(load_weight["model_state"])
        while(cap.isOpened()):
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resize_img = cv2.resize(img, (160, 80), interpolation=cv2.INTER_AREA)
            img_check = torch.Tensor(resize_img).permute(2,0,1).unsqueeze(0)
            prediction = net(img_check).squeeze(0).permute(1,2,0)
            prediction = cv2.cvtColor(prediction.numpy(), cv2.COLOR_GRAY2RGB)*255
            oldImage = img.copy()
            prediction = cv2.resize(prediction, (1280, 720), interpolation=cv2.INTER_AREA)
            newImage = cv2.addWeighted(oldImage, 1, np.asarray(prediction).astype(np.uint8), 1, 0)
            plt.imsave("outputs/semantic_seg.jpg", newImage)
            cv2.imshow('frame',newImage)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

test()
