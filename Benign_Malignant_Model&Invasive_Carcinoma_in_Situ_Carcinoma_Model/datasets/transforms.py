import numpy as np
from torchvision import transforms
import torch
import random
import cv2
from PIL import Image

class MaskRandomPad(object):
    def __init__(self, rate=0.2, p=0.5):
        self.rate = rate
        self.p = p

    def __call__(self, sample):
        if random.random()>self.p:
            return sample
        img = sample['image']
        startX = int(sample["startX"])
        startY = int( sample["startY"])
        endX = int(sample["endX"])
        endY = int(sample["endY"])

        rx = random.randint(0, int(abs(endX-startX)/2 * self.rate))
        ry = random.randint(0, int(abs(endY-startY)/2 * self.rate))
        sample["startX"] = max(0,startX-rx)
        sample["endX"] = min(img.shape[1],endX+rx)
        sample["startY"] = max(0,startY-ry)
        sample["endY"] = min(img.shape[0],endY+ry)
        return sample

class RandomErasing(object):
    def __init__(self, p=0.9, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False):
        self.p = p
        self.tf = transforms.RandomErasing(1,scale,ratio,value)

    def __call__(self, sample):
        if random.random()>self.p:
            return sample
        img = sample['image']
        startX = int(sample["startX"])
        startY = int( sample["startY"])
        endX = int(sample["endX"])
        endY = int(sample["endY"])    
        mask = np.zeros(img.shape).astype(np.uint8)
        # save lesion area
        mask[startY:endY,startX:endX,:]=img[startY:endY,startX:endX,:]
        img = self.tf(torch.Tensor(img))
        img = np.array(img)
        img[startY:endY,startX:endX,:]=0
        img+=mask
        sample['image'] = img
        return sample

class ColorJitter(object):
    def __init__(self, p=0.9, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5):
        self.p = p
        self.tf = transforms.ColorJitter(brightness,contrast,saturation,hue)

    def __call__(self, sample):
        if random.random()>self.p:
            return sample
        sample['image'] = np.array(self.tf(Image.fromarray(sample['image'])))
        return sample

class AddMask(object):
    def __init__(self, mode=0):
        self.mode = mode

    def __call__(self, sample):
        img = sample['image']

        startX = int(sample["startX"])
        startY = int(sample["startY"])
        endX = int(sample["endX"])
        endY = int(sample["endY"])
        mask = np.zeros(img.shape[:2])
        if self.mode==0:
            mask[startY:endY,startX:endX]=255
        elif self.mode==1:
            mask[startY:endY,startX:endX]=img[startY:endY,startX:endX,:].mean(axis=2)
        else:
            raise NotImplementedError("mask mode not implemented!")
        sample['mask'] = mask
        return sample

class ToTensor(object):
    """Convert cv2 ndarrays in sample to Tensors."""

    def __init__(self):
        pass

    def __call__(self, sample):
        # swap color axis because
        # cv2 numpy image: H x W x C
        # torch image: C X H X W
        img = sample["image"]
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        img = torch.from_numpy(img).float()
        if "mask" in sample:
            mask = sample["mask"]
            mask = np.expand_dims(np.array(mask).astype(np.float32), -1).transpose((2, 0, 1))
            mask = torch.from_numpy(mask).float()
            img = torch.concat([img,mask],dim=0)
        sample["image"] = img/255
        return sample

class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, sample):
        if random.random() < self.prob:
            sample["image"] = cv2.flip(sample["image"],1)
            if "mask" in sample:
                sample["mask"] = cv2.flip(sample["mask"],1)
        return sample

class FixedResize(object):
    def __init__(self, size):
        self.size = tuple(size)

    def __call__(self, sample):
        sample["image"] = cv2.resize(sample["image"], dsize=self.size, interpolation = cv2.INTER_LINEAR)
        if "mask" in sample:
            sample["mask"] = cv2.resize(sample["mask"], dsize=self.size, interpolation = cv2.INTER_NEAREST)
        return sample

class RandomTranslate(object):
    def __init__(self, prob,range=20):
        self.prob = prob
        self.range = range

    def __call__(self, sample):
        if random.random() < self.prob:
            img = np.array(sample["image"])
            l_u, l_d = np.random.randint(self.range), img.shape[0]-np.random.randint(self.range)
            r_u, r_d = np.random.randint(self.range), img.shape[1]-np.random.randint(self.range)
            img = img[l_u:l_d,r_u:r_d,:]
            sample["image"] = img
            if "mask" in sample:
                mask = np.array(sample["mask"])
                mask = mask[l_u:l_d,r_u:r_d]
                sample["mask"] = mask
        return sample