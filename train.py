import torch
from dataset import *
from model import *
from torchvision import transforms
from model import *
import sys

def train():
    transform_train = transforms.Compose([
        #transforms.Resize((256,480)),
        #transforms.Grayscale(),
        transforms.ToTensor(),                                                # convert the PIL Image to a tensor
        transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))
    ])
    batch_size= 16
    train_loader= data_loader(transform_train, True, batch_size)
    network = model()
    iteration = 100
    #loss
    criteria = nn.L1Loss()
    optimizer = torch.optim.SGD(network.parameters(), lr=0.001, momentum=0.9)
    step_per_batch = int(np.ceil(train_loader.dataset.df.shape[0]/batch_size))
    for i in range(iteration):
        for step in range(step_per_batch):
            img, lane1, lane2 = next(iter(train_loader))
            # now we send the img through model
            lane1_prediction, lane2_prediction = network(img)
            lane1 = lane1.view(-1,2)
            lane2 = lane2.view(-1,2)
            lane1_prediction  = lane1_prediction.view(-1,2)
            lane2_prediction  = lane2_prediction.view(-1,2)
            #print(lane1_prediction)
            # now we calculate the loss separtely
            loss = criteria(lane1, lane1_prediction) + criteria(lane2, lane2_prediction)
            loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1)
            stats = "[%d/%d] LOSS:%.4f"%(step, iteration, loss.item())
            print("\r "+stats, end="")
            sys.stdout.flush()
            if step%50 == 0 and step!=0:
                torch.save({
                    "model_state":network.state_dict()
                }, "weights/weight"+str(step)+".pth")
                print("\r"+stats)

train()

