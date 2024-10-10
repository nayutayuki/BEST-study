import os
import pickle
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
import json
import cv2

class Invasive(Dataset):
    """Breast cancer dataset from DaZhou."""

    def __init__(self, root_dir, txt_file, mode="gray", transforms=None):
        with open(txt_file, "r") as fp:
            samples = fp.readlines()
        samples = [i[:-1].split("\t") for i in samples]
        self.data_list = samples
        self.transforms = transforms
        self.root_dir = root_dir
        self.mode = mode

    def __len__(self):
        return len(self.data_list)
    
    def _parse_loc(self, json_path):
        json_file = json.load(open(json_path, "r"))
        mask = []
        for shape in json_file["shapes"]:
            if shape["label"]=="tumor":
                mask.append(int(shape["points"][0][0]))
                mask.append(int(shape["points"][0][1]))
                mask.append(int(shape["points"][1][0]))
                mask.append(int(shape["points"][1][1]))
        return mask
    
    def __getitem__(self, idx):
        record = self.data_list[idx]
        sample = {}
        if self.mode=="gray":
            img = cv2.imread(os.path.join(self.root_dir, record[0]))
            mask = self._parse_loc(os.path.join(self.root_dir, record[1]))
        elif self.mode=="cdfi":
            img = cv2.imread(os.path.join(self.root_dir, record[2]))
            mask = self._parse_loc(os.path.join(self.root_dir, record[3]))
        else:
            raise NotImplemented
        
        sample["image"] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        sample["startX"] = mask[0]
        sample["startY"] = mask[1]
        sample["endX"] = mask[2]
        sample["endY"] = mask[3]
        sample["Label"] = int(record[4])
        
        if self.transforms is not None:
            if isinstance(self.transforms, list):
                for trans in self.transforms:
                    sample = trans(sample)
            else:
                sample = trans(sample)

        return sample["image"], sample["Label"]