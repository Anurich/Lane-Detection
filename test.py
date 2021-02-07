import torch
import torch.nn as  nn
from model import *
from dataset import *
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
def convertdata(lane):
    lane  = np.array(lane).reshape(-1,2)
    lane[:,0] = lane[:,0] * 480
    lane[:,1] = lane[:,1] * 256

    return lane

def drawLane(img_og, lane1, lane2, color):
        cv2.polylines(img_og,np.int32([lane1]),isClosed=False, color=color, thickness=5)
        cv2.polylines(img_og, np.int32([lane2]),isClosed=False, color=color, thickness=5)



def test():
    # load the weight

    transform_train = transforms.Compose([
        #transforms.Resize((256,480)),
        transforms.ToTensor(),                           # convert the PIL Image to a tensor
        transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))
    ])
    network = model()
    network.eval()
    weight = torch.load("weights/weight50.pth")
    network.load_state_dict(weight["model_state"])
    test_loader = data_loader(transform_train, train=False, batch_size=1)
    img,lane1_gt,lane2_gt,original_img = next(iter(test_loader))
    max_lane1 = lane1_gt.max()
    min_lane1 = lane1_gt.min()
    max_lane2 = lane2_gt.max()
    min_lane2 = lane2_gt.min()
    lane1, lane2  = network(img)
    lane1 = lane1*(max_lane1 - min_lane1)+min_lane1
    lane2 = lane2*(max_lane2 - min_lane2) + min_lane2
    img_og = np.squeeze(original_img.numpy().copy())
    with torch.no_grad():
        lane1 = np.array(lane1).reshape(-1,2)
        lane2 = np.array(lane2).reshape(-1,2)
        lane1_gt = np.array(lane1_gt).reshape(-1,2)
        lane2_gt = np.array(lane2_gt).reshape(-1,2)
        drawLane(img_og, lane1, lane2,(0,255,0))
        drawLane(img_og, lane1_gt, lane2_gt, (255,0,0))
    plt.imshow(img_og)
    plt.show()
test()
