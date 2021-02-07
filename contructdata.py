import numpy as np
import pandas as pd
import json
import os
class constructdataset:
    def __init__(self, filename, train=True):
        self.filename = filename
        self.df = pd.DataFrame()
        self.train_file = "train.csv"
        self.test_file  = "test.csv"
        if train:
            if not os.path.exists(self.train_file):
                self.read(self.train_file)
            else:
                print("train file is already constructed")
        else:
            if not os.path.exists(self.test_file):
                self.read(self.test_file)
            else:
                print("test file is already constructed")

    def laneExtraction(self,height, lane):
        points = []
        for h, l in zip(height, lane):
            if l > 0:
                points.append(l)
                points.append(h)
        return points
    def read(self, train_or_test):
        for files in self.filename:
            with open(files,'r') as fp:
                for line in fp:
                    data = json.loads(line)
                    imageName = data["raw_file"]
                    height    = data["h_samples"]
                    lanes     = data["lanes"]
                    lane1 = self.laneExtraction(height, lanes[0])
                    lane2 = self.laneExtraction(height, lanes[1])

                    # check if the size of the lane 1 is 30 and lane 2 is 30
                    size_lane1 = np.array(lane1).reshape(-1,2)
                    size_lane2 = np.array(lane2).reshape(-1,2)

                    if size_lane1.shape[0] >= 30 and size_lane2.shape[0] >= 30:
                        store = {"imagename":imageName,
                             "lane1":lane1,
                             "lane2":lane2
                             }
                        self.df = self.df.append(store, ignore_index=True)
            # saving the file
            self.df.to_csv(train_or_test, index=False)


