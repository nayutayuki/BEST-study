import os
import numpy as np
from torch.utils.data import Dataset
import pdb
import time
import cv2
import scipy
from shutil import copyfile

class BCData(Dataset):
    def __init__(self, transform=None, phase='Train', parent_dir=None, over_sample=None):
        self.phase = phase     
        data_files = []
        boundingbox = []
        labels = []
        label_dist = np.zeros((3,))
        
        if phase=="Train":
            cur_labeldir = os.path.join(parent_dir, 'label/GT_train.txt')
        else:
            cur_labeldir = os.path.join(parent_dir, 'label/GT_test.txt')

        count = 0
        with open(cur_labeldir, "r") as f:
            xs = f.readlines()
            for x in xs:
                anchor, cur_label, cur_path = self.general_process(x)
                if cur_label is None:
                    continue
                labels.append(cur_label)
                label_dist[cur_label] += 1
                data_files.append(os.path.join(parent_dir, "data", cur_path))
                boundingbox.append(anchor)
                count += 1
        print("Dazhou have ", count, " ", phase, " data")                  

        ###################### OVER SAMPLING HERE ############################
        self.data_files = data_files
        self.boundingbox = boundingbox
        self.labels = labels

        if(over_sample is not None):
            for idx in range(len(data_files)):
                for i in range(over_sample[labels[idx]]):
                    self.data_files.append(data_files[idx])
                    self.boundingbox.append(boundingbox[idx])
                    self.labels.append(labels[idx])
                    label_dist[labels[idx]] += 1          

        self.transform = transform
        print('the data length is %d, for %s' % (len(self.data_files), phase))
        print("class 1/2->%d" % (label_dist[0]))
        print("class 0/3->%d" % (label_dist[1]))
        print("class 4/5->%d" % (label_dist[2]))


    def __len__(self):
        L = len(self.data_files)
        return L
    
    def general_process(self, string):
        x = string.split('\t')
        if(len(x)<2):
            x = string.split(' ')
        label_arr = x[1].split('\n')[0].split('_T_')
        
        if int(label_arr[0]) == 0: # 如果是1类，没有检测框，剔除掉
            return None, None, None

        if int(label_arr[0]) <2:
            cur_label = 0
        elif int(label_arr[0]) > 2:
            cur_label = 2
        else:
            cur_label = 1
        anchor = np.array([min(int(label_arr[1]), int(label_arr[2])),
                    min(int(label_arr[3]), int(label_arr[4])),
                    max(int(label_arr[1]), int(label_arr[2])),
                    max(int(label_arr[3]), int(label_arr[4]))])
        return anchor, cur_label, x[0]


    def __getitem__(self, index):
        _img = cv2.imread(self.data_files[index])
        _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
        _label = self.labels[index]
        _boundingbox = self.boundingbox[index]
        
        sample = { 'image': _img, 'label': _label, 'boundingbox':_boundingbox}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
    def __str__(self):
        pass


class GradeData(Dataset):
    def __init__(self, transform=None, phase='Train', 
                 label_dict=None, 
                 parent_dir=None,
                 over_sample=None,
                 data_name = "Dazhou"):
        self.phase = phase
        self.label_dict = label_dict      
        self.data_files = []
        self.boundingbox = []
        self.labels = []

        data_files = []
        boundingbox = []
        labels = []
        
        data_names = data_name.split("_")
        label_dist = np.zeros((4,))
        
        if "Dazhou" in data_names:
            if phase=="Train":
                cur_labeldir = os.path.join(parent_dir, 'developmentSet/label/GT_train_val.txt')
            else:
                cur_labeldir = os.path.join(parent_dir, 'developmentSet/label/GT_test.txt')

            count = 0
            with open(cur_labeldir, "r") as f:
                xs = f.readlines()
                for x in xs:
                    anchor, cur_label, cur_path = self.general_process(x)
                    if cur_label is None:
                        continue
                    labels.append(cur_label)
                    label_dist[cur_label] += 1
                    data_files.append(os.path.join(parent_dir, "developmentSet/data", cur_path))
                    boundingbox.append(anchor)
                    count += 1
            print("Dazhou have ", count, " ", phase, " data")                  


        if "Jdf" in data_names and phase=="Train":
            parent_path = "/root/workspace/Python3/data/BIData/auxiliary_dataset/JunQuZongData/"
            cur_labeldir = os.path.join(parent_dir, 'jdf.txt')
            count = 0
            with open(cur_labeldir, "r") as f:
                xs = f.readlines()
                for x in xs:
                    anchor, cur_label, cur_path = self.general_process(x)
                    if cur_label is None:
                        continue
                    labels.append(cur_label)
                    label_dist[cur_label] += 1
                    data_files.append(os.path.join(parent_path, "data", cur_path))
                    boundingbox.append(anchor)
                    count += 1
            print("Jdf have ", count, " ", phase, " data")


        if "Center" in data_names and phase=="Train":
            parent_path = os.path.join(parent_dir,"centerSet/data")
            if phase=="Train":
                cur_labeldir = os.path.join(parent_dir, 'centerSet/label/GT_all.txt')
            else:
                cur_labeldir = os.path.join(parent_dir, 'centerSet/label/GT_test.txt')
            count = 0
            with open(cur_labeldir, "r") as f:
                xs = f.readlines()
                for x in xs:
                    anchor, cur_label, cur_path = self.general_process(x)
                    if cur_label is None:
                        continue
                    labels.append(cur_label)
                    label_dist[cur_label] += 1
                    data_files.append(os.path.join(parent_path, cur_path))
                    boundingbox.append(anchor)
                    count += 1
            print("Center have ", count, " ", phase, " data")
            
        if "OtherOrgan" in data_names and phase=="Train":
            parent_path = os.path.join(parent_dir,"otherOrganSet/data")
            if phase=="Train":
                cur_labeldir = os.path.join(parent_dir, 'otherOrganSet/label/GT_train.txt')
            count = 0
            with open(cur_labeldir, "r") as f:
                xs = f.readlines()
                for x in xs:
                    anchor, cur_label, cur_path = self.general_process(x)
                    if cur_label is None:
                        continue
                    labels.append(cur_label)
                    label_dist[cur_label] += 1
                    data_files.append(os.path.join(parent_path, cur_path))
                    boundingbox.append(anchor)
                    count += 1
            print("OtherOrgan have ", count, " ", phase, " data")
            

        ###################### OVER SAMPLING HERE ############################
        self.data_files = data_files
        self.boundingbox = boundingbox
        self.labels = labels
        if(over_sample is not None and phase=="Train"):
            for idx in range(len(data_files)):
                for i in range(over_sample[labels[idx]]):
                    self.data_files.append(data_files[idx])
                    self.boundingbox.append(boundingbox[idx])
                    self.labels.append(labels[idx])
                    label_dist[labels[idx]] += 1          

        self.transform = transform
        print('the data length is %d, for %s' % (len(self.data_files), phase))
        print("class 1/2->%d" % (label_dist[0]))
        print("class 0/3->%d" % (label_dist[1]))
        print("class 4/5->%d" % (label_dist[2]))
        print("other organ ->%d" % (label_dist[3]))


    def __len__(self):
        L = len(self.data_files)
        return L
    
    def general_process(self, string):
        x = string.split('\t')
        if(len(x)<2):
            x = string.split(' ')
        if len(x) !=2:
            return None, None, None
        
        label_arr = x[1].split('\n')[0].split('_T_')

        if int(label_arr[0]) == -1:
            cur_label = 3
        elif int(label_arr[0]) <2:
            cur_label = 0
        elif int(label_arr[0]) > 2:
            cur_label = 2
        else:
            cur_label = 1
        anchor = np.array([min(int(label_arr[1]), int(label_arr[2])),
                    min(int(label_arr[3]), int(label_arr[4])),
                    max(int(label_arr[1]), int(label_arr[2])),
                    max(int(label_arr[3]), int(label_arr[4]))])
        if(anchor[2]-anchor[0]) / (anchor[3]-anchor[1]+0.0001) > 5 or (anchor[2]-anchor[0]) / (anchor[3]-anchor[1]+0.0001) < 0.2:
            return None, None, None
        
        if anchor[0] == 0 or anchor[1] == 0 or anchor[2] == 0 or anchor[3] == 0:
            return None, None, None

        return anchor, cur_label, x[0]


    def __getitem__(self, index):
        _img = cv2.imread(self.data_files[index])

        _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
        _label = self.labels[index]
        _boundingbox = self.boundingbox[index]
        
        sample = { 'image': _img, 'label': _label, 'boundingbox':_boundingbox}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
    def __str__(self):
        pass
    
    
class BIRADSData(Dataset):
    def __init__(self, transform=None, phase='Train', parent_dir=None, label_dict=None, data_name=None, over_sample=None):
        self.phase = phase     
        data_files = []
        boundingbox = []
        labels = []
        label_dist = np.zeros((7,))
        self.label_dict = label_dict
        
        if phase=="Train":
            cur_labeldir = os.path.join(parent_dir, 'label/GT_train.txt')
        elif phase == "Val":
            cur_labeldir = os.path.join(parent_dir, 'label/GT_val.txt')
        else:
            cur_labeldir = os.path.join(parent_dir, 'label/GT_test.txt')

        count = 0
        with open(cur_labeldir, "r") as f:
            xs = f.readlines()
            for x in xs:
                anchor, cur_label, cur_path = self.general_process(x)
                if cur_label is None:
                    continue
                labels.append(cur_label)
                label_dist[cur_label] += 1
                data_files.append(os.path.join(parent_dir, "data", cur_path))
                boundingbox.append(anchor)
                count += 1
        print("Dazhou have ", count, " ", phase, " data")                  

        ###################### OVER SAMPLING HERE ############################
        self.data_files = data_files
        self.boundingbox = boundingbox
        self.labels = labels

        if(over_sample is not None):
            for idx in range(len(data_files)):
                for i in range(over_sample[labels[idx]]):
                    self.data_files.append(data_files[idx])
                    self.boundingbox.append(boundingbox[idx])
                    self.labels.append(labels[idx])
                    label_dist[labels[idx]] += 1          

        self.transform = transform
        print('the data length is %d, for %s' % (len(self.data_files), phase))
        print("class 1->%d" % (label_dist[0]))
        print("class 2->%d" % (label_dist[1]))
        print("class 3->%d" % (label_dist[2]))
        print("class 4A->%d" % (label_dist[3]))
        print("class 4B->%d" % (label_dist[4]))
        print("class 4C->%d" % (label_dist[5]))
        print("class 5->%d" % (label_dist[6]))


    def __len__(self):
        L = len(self.data_files)
        return L
    
    def general_process(self, string):
        x = string.split('\t')
        if(len(x)<2):
            x = string.split(' ')
        label_arr = x[1].split('\n')[0].split('_T_')

        if label_arr[0] in self.label_dict.keys():
            cur_label = self.label_dict[label_arr[0]]
        else:
            return None, None, None
        anchor = np.array([min(int(label_arr[1]), int(label_arr[2])),
                    min(int(label_arr[3]), int(label_arr[4])),
                    max(int(label_arr[1]), int(label_arr[2])),
                    max(int(label_arr[3]), int(label_arr[4]))])
        return anchor, cur_label, x[0]


    def __getitem__(self, index):
        _img = cv2.imread(self.data_files[index])
        _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
        _label = self.labels[index]
        _boundingbox = self.boundingbox[index]
        
        sample = { 'image': _img, 'label': _label, 'boundingbox':_boundingbox}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
    def __str__(self):
        pass


class CenterBIData(Dataset):
    def __init__(self, transform=None, parent_dir=None, label_dict=None, data_name=None, over_sample=None):  
        data_files = []
        boundingbox = []
        labels = []
        label_dist = np.zeros((7,))
        self.label_dict = label_dict
        cur_labeldir = os.path.join(parent_dir, 'label/GT_all.txt')

        count = 0
        with open(cur_labeldir, "r") as f:
            xs = f.readlines()
            for x in xs:
                anchor, cur_label, cur_path = self.general_process(x)

                if data_name == "usgs":
                    if not (data_name in cur_path or "DaChuanZhongYi" in cur_path):
                        continue
                elif data_name == "resized_subline_breast_cases":
                    if not (data_name in cur_path or "QuXian" in cur_path):
                        continue
                elif data_name == "DaChuan":
                    if (data_name not in cur_path) or ("DaChuanZhongYi" in cur_path):
                        continue
                else:
                    if not data_name in cur_path:
                        continue
                if cur_label is None:
                    continue
                labels.append(cur_label)
                label_dist[cur_label] += 1
                data_files.append(os.path.join(parent_dir, "data", cur_path))
                boundingbox.append(anchor)
                count += 1
        print(data_name, " have ", count, " data")                  

        ###################### OVER SAMPLING HERE ############################
        self.data_files = data_files
        self.boundingbox = boundingbox
        self.labels = labels

        if(over_sample is not None):
            for idx in range(len(data_files)):
                for i in range(over_sample[labels[idx]]):
                    self.data_files.append(data_files[idx])
                    self.boundingbox.append(boundingbox[idx])
                    self.labels.append(labels[idx])
                    label_dist[labels[idx]] += 1          

        self.transform = transform
        print('the data length is %d ' % (len(self.data_files)))
        print("class 1->%d" % (label_dist[0]))
        print("class 2->%d" % (label_dist[1]))
        print("class 3->%d" % (label_dist[2]))
        print("class 4A->%d" % (label_dist[3]))
        print("class 4B->%d" % (label_dist[4]))
        print("class 4C->%d" % (label_dist[5]))
        print("class 5->%d" % (label_dist[6]))


    def __len__(self):
        L = len(self.data_files)
        return L
    
    def general_process(self, string):
        x = string.split('\t')
        if(len(x)<2):
            x = string.split(' ')
        label_arr = x[1].split('\n')[0].split('_T_')

        if label_arr[0] in self.label_dict.keys():
            cur_label = self.label_dict[label_arr[0]]
        else:
            return None, None, None
        anchor = np.array([min(int(label_arr[1]), int(label_arr[2])),
                    min(int(label_arr[3]), int(label_arr[4])),
                    max(int(label_arr[1]), int(label_arr[2])),
                    max(int(label_arr[3]), int(label_arr[4]))])
        return anchor, cur_label, x[0]


    def __getitem__(self, index):
        _img = cv2.imread(self.data_files[index])
        _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
        _label = self.labels[index]
        _boundingbox = self.boundingbox[index]
        
        sample = { 'image': _img, 'label': _label, 'boundingbox':_boundingbox}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
    def __str__(self):
        pass

    
class InternalBIData(Dataset):
    def __init__(self, transform=None, parent_dir=None, label_dict=None, data_name=None, over_sample=None):  
        data_files = []
        boundingbox = []
        labels = []
        label_dist = np.zeros((7,))
        self.label_dict = label_dict
        cur_labeldir = os.path.join(parent_dir, data_name+'_label.txt')

        count = 0
        with open(cur_labeldir, "r") as f:
            xs = f.readlines()
            for x in xs:
                anchor, cur_label, cur_path = self.general_process(x)
                
                if cur_label is None:
                    continue
                labels.append(cur_label)
                label_dist[cur_label] += 1
                data_files.append(os.path.join(parent_dir, data_name, cur_path))
                boundingbox.append(anchor)
                count += 1
        print(data_name, " have ", count, " data")                  

        ###################### OVER SAMPLING HERE ############################
        self.data_files = data_files
        self.boundingbox = boundingbox
        self.labels = labels

        if(over_sample is not None):
            for idx in range(len(data_files)):
                for i in range(over_sample[labels[idx]]):
                    self.data_files.append(data_files[idx])
                    self.boundingbox.append(boundingbox[idx])
                    self.labels.append(labels[idx])
                    label_dist[labels[idx]] += 1          

        self.transform = transform
        print('the data length is %d ' % (len(self.data_files)))
        print("class 1->%d" % (label_dist[0]))
        print("class 2->%d" % (label_dist[1]))
        print("class 3->%d" % (label_dist[2]))
        print("class 4A->%d" % (label_dist[3]))
        print("class 4B->%d" % (label_dist[4]))
        print("class 4C->%d" % (label_dist[5]))
        print("class 5->%d" % (label_dist[6]))


    def __len__(self):
        L = len(self.data_files)
        return L
    
    def general_process(self, string):
        x = string.split('\t')
        if(len(x)<2):
            x = string.split(' ')
        label_arr = x[1].split('\n')[0].split('_T_')

        if label_arr[0] in self.label_dict.keys():
            cur_label = self.label_dict[label_arr[0]]
        else:
            return None, None, None
        anchor = np.array([min(int(label_arr[1]), int(label_arr[2])),
                    min(int(label_arr[3]), int(label_arr[4])),
                    max(int(label_arr[1]), int(label_arr[2])),
                    max(int(label_arr[3]), int(label_arr[4]))])
        return anchor, cur_label, x[0]


    def __getitem__(self, index):
        _img = cv2.imread(self.data_files[index])
        _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
        _label = self.labels[index]
        _boundingbox = self.boundingbox[index]
        
        sample = { 'image': _img, 'label': _label, 'boundingbox':_boundingbox}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
    def __str__(self):
        pass
    
    
class RetroTestData(Dataset):
    def __init__(self, transform=None, parent_dir=None, label_dict=None, data_name=None, over_sample=None):     
        data_files = []
        boundingbox = []
        labels = []
        self.label_dict = label_dict
        cur_labeldir = os.path.join(parent_dir, 'label/label.txt')

        count = 0
        with open(cur_labeldir, "r") as f:
            xs = f.readlines()
            for x in xs:
                anchor, cur_label, cur_path = self.general_process(x)
                if cur_label is None:
                    continue
                labels.append(cur_label)
                data_files.append(os.path.join(parent_dir, "data", cur_path))
                boundingbox.append(anchor)
                count += 1
                
        ###################### OVER SAMPLING HERE ############################
        self.data_files = data_files
        self.boundingbox = boundingbox
        self.labels = labels
        self.transform = transform

    def __len__(self):
        L = len(self.data_files)
        return L
    
    def __getitem__(self, index):
        _img = cv2.imread(self.data_files[index])
        _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
        _label = self.labels[index]
        _boundingbox = self.boundingbox[index]
        
        sample = { 'image': _img, 'label': _label, 'boundingbox':_boundingbox}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
    def __str__(self):
        pass

    def general_process(self, string):
        x = string.split('\t')
        if(len(x)<2):
            x = string.split(' ')
        label_arr = x[1].split('\n')[0].split('_T_')

        if label_arr[0] in self.label_dict.keys():
            cur_label = self.label_dict[label_arr[0]]
        else:
            return None, None, None
        anchor = np.array([min(int(label_arr[1]), int(label_arr[2])),
                    min(int(label_arr[3]), int(label_arr[4])),
                    max(int(label_arr[1]), int(label_arr[2])),
                    max(int(label_arr[3]), int(label_arr[4]))])
        return anchor, cur_label, x[0]
    