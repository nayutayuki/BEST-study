import os
import cv2
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import timm
from loguru import logger
import numpy as np
import pandas as pd
from torch.nn import functional as F

class Config:
    log_name="only_marked.log"
    data_path=["unmarked/DZCH/","unmarked/HNPH/","unmarked/SCPH/","unmarked/STFH/"]
    anno_path=["anno/anno_DZCH/","anno/anno_HNPH/","anno/anno_SCPH/","anno/anno_STFH/"]
    label_dict={"Basal-like":0,"HER2+":1,"Luminal A":2,"Luminal B":3,"Luminal B-HER2+":4}
    train_over_sample=[[1,3,4,4],
                       [1,2,7,8],
                       [1,2,1,4],
                       [1,1,1,1],
                       [2,2,2,2]]
    train_num=[425,531,109,524]
    eval_num=[33,22,2,8]
    test_num=[33,22,2,8]
    net="resnet18"
    pretrained=True
    num_classes=5
    resize_size=128
    pad=2
    degrees=10
    batch_size=256
    lr=1e-3
    in_chans=3
    epoch=100
    device=torch.device("cuda:0"if torch.cuda.is_available()else "cpu")
    train_transform=transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(resize_size),
        transforms.Pad(pad),
        transforms.RandomCrop(resize_size),
        transforms.RandomAffine(degrees=degrees),
        transforms.ToTensor(),
    ])
    eval_transform=transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(resize_size),
        transforms.CenterCrop(resize_size),
        transforms.ToTensor(),
    ])
    test_transform=transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(resize_size),
        transforms.CenterCrop(resize_size),
        transforms.ToTensor(),
    ])
config=Config()

class Data(Dataset):
    def __init__(self,phase='train',transform=None):
        self.D=[]
        self.T=[]
        for i,hospital in enumerate(config.data_path):
            logger.info("%s"%(hospital))
            for key,value in config.label_dict.items():
                path=hospital+key+'/'
                dr=os.listdir(path)
                if phase=='test':dr=dr[0:config.test_num[i]]
                elif phase=='eval':dr=dr[config.test_num[i]:config.test_num[i]*2]
                else:
                    help=dr[config.test_num[i]*2:]
                    dr=[]
                    for _ in range(config.train_over_sample[value][i]):
                        dr.extend(help)
                    dr=dr[0:config.train_num[i]]
                for name in dr:
                    img=cv2.imread(path+name)
                    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                    csv=config.anno_path[i]+name[:-4]+".csv"
                    df=pd.read_csv(csv)
                    startX,endX,startY,endY=int(df.loc[0,'startX']),int(df.loc[0,'endX']),int(df.loc[0,'startY']),int(df.loc[0,'endY'])
                    mask=np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8)
                    mask[startY:endY,startX:endX,:]=img[startY:endY,startX:endX,:]
                    if transform is not None:
                        mask=transform(mask)
                    self.D.append(mask)
                    self.T.append(value)

    def __len__(self):
        return len(self.D)
    def __getitem__(self,idx):
        return self.D[idx],self.T[idx]
    def __str__(self):
        pass

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.net=timm.create_model(config.net,pretrained=config.pretrained,num_classes=config.num_classes,in_chans=config.in_chans)

    def forward(self,x):
        return self.net(x)

if __name__=='__main__':
    logger.add(config.log_name)
    trainLD=DataLoader(dataset=Data("train",config.train_transform),
                       batch_size=config.batch_size,
                       shuffle=True,
                       drop_last=False)
    evalLD=DataLoader(dataset=Data("eval",config.eval_transform),
                       batch_size=config.batch_size,
                       shuffle=False,
                       drop_last=False)
    testLD=DataLoader(dataset=Data("test",config.test_transform),
                      batch_size=config.batch_size,
                      shuffle=False,
                      drop_last=False)
    net=Net().to(config.device)
    criterion=torch.nn.CrossEntropyLoss(size_average=True).to(config.device)
    optimizer=torch.optim.AdamW(net.parameters(),lr=config.lr, weight_decay=1e-2)
    
    best_epoch,best_acc=-1,-1
    for epoch in range(1,config.epoch+1):
        #train
        net.train()
        loss_sum=0
        excel_data_all=None
        excel_format={
            'real_label':[],
            'pred_label':[],
            "0":[],
            "1":[],
            "2":[],
            "3":[],
            "4":[],
        }
        correct,tot=0,0
        for i,(x,y) in enumerate(trainLD):
            optimizer.zero_grad()
            x,y=x.to(config.device),y.to(config.device)
            y_pred=net(x)
            loss=criterion(y_pred,y)
            loss_sum+=loss.item()
            loss.backward()
            optimizer.step()
            _,yy=torch.max(y_pred.data,dim=1)
            correct+=(yy==y).sum().item()
            tot+=y.size(0)
            excel_data=torch.cat((y.data.view(-1,1),yy.data.view(-1,1),F.softmax(y_pred,dim=1).data),dim=1)
            if excel_data_all is None: excel_data_all=excel_data
            else: excel_data_all=torch.cat((excel_data_all,excel_data))
        excel_format['real_label']=excel_data_all[:,0].cpu().numpy().tolist()
        excel_format['pred_label']=excel_data_all[:,1].cpu().numpy().tolist()
        excel_format['0']=excel_data_all[:,2].cpu().numpy().tolist()
        excel_format['1']=excel_data_all[:,3].cpu().numpy().tolist()
        excel_format['2']=excel_data_all[:,4].cpu().numpy().tolist()
        excel_format['3']=excel_data_all[:,5].cpu().numpy().tolist()
        excel_format['4']=excel_data_all[:,6].cpu().numpy().tolist()
        excel=pd.DataFrame(excel_format)
        excel.to_excel('only_marked_train.xlsx',index=False)

        logger.info("Epoch%d loss: %.5f"%(epoch,loss_sum))  
        logger.info("Train Acc: %.5f%%"%(100.0*correct/tot))
        #eval
        excel_data_all=None
        excel_format={
            'real_label':[],
            'pred_label':[],
            "0":[],
            "1":[],
            "2":[],
            "3":[],
            "4":[],
        }
        net.eval()
        correct,tot=0,0
        with torch.no_grad():
            for x,y in evalLD:
                x,y=x.to(config.device),y.to(config.device)
                y_pred=net(x)
                _,yy=torch.max(y_pred.data,dim=1)
                correct+=(yy==y).sum().item()
                tot+=y.size(0)
                excel_data=torch.cat((y.data.view(-1,1),yy.data.view(-1,1),F.softmax(y_pred,dim=1).data),dim=1)
                if excel_data_all is None: excel_data_all=excel_data
                else: excel_data_all=torch.cat((excel_data_all,excel_data))
            excel_format['real_label']=excel_data_all[:,0].cpu().numpy().tolist()
            excel_format['pred_label']=excel_data_all[:,1].cpu().numpy().tolist()
            excel_format['0']=excel_data_all[:,2].cpu().numpy().tolist()
            excel_format['1']=excel_data_all[:,3].cpu().numpy().tolist()
            excel_format['2']=excel_data_all[:,4].cpu().numpy().tolist()
            excel_format['3']=excel_data_all[:,5].cpu().numpy().tolist()
            excel_format['4']=excel_data_all[:,6].cpu().numpy().tolist()
            excel=pd.DataFrame(excel_format)
            excel.to_excel('only_marked_eval.xlsx',index=False)
        logger.info("Eval Acc: %.5f%%"%(100.0*correct/tot))
        #test
        excel_data_all=None
        excel_format={
            'real_label':[],
            'pred_label':[],
            "0":[],
            "1":[],
            "2":[],
            "3":[],
            "4":[],
        }
        score_list=[]
        label_list=[]
        net.eval()
        correct,tot=0,0
        cf=np.zeros((config.num_classes,config.num_classes))
        with torch.no_grad():
            for x,y in testLD:
                x,y=x.to(config.device),y.to(config.device)
                y_pred=net(x)
                score_list.extend(y_pred.detach().cpu().numpy())
                label_list.extend(y.cpu().numpy())
                _,yy=torch.max(y_pred.data,dim=1)
                correct+=(yy==y).sum().item()
                tot+=y.size(0)
                for i in range(y.size(0)):
                    cf[y.data[i],yy[i]]+=1
                excel_data=torch.cat((y.data.view(-1,1),yy.data.view(-1,1),F.softmax(y_pred,dim=1).data),dim=1)
                if excel_data_all is None: excel_data_all=excel_data
                else: excel_data_all=torch.cat((excel_data_all,excel_data))
            excel_format['real_label']=excel_data_all[:,0].cpu().numpy().tolist()
            excel_format['pred_label']=excel_data_all[:,1].cpu().numpy().tolist()
            excel_format['0']=excel_data_all[:,2].cpu().numpy().tolist()
            excel_format['1']=excel_data_all[:,3].cpu().numpy().tolist()
            excel_format['2']=excel_data_all[:,4].cpu().numpy().tolist()
            excel_format['3']=excel_data_all[:,5].cpu().numpy().tolist()
            excel_format['4']=excel_data_all[:,6].cpu().numpy().tolist()
            excel=pd.DataFrame(excel_format)
            excel.to_excel('only_marked_test.xlsx',index=False)
        logger.info("Test Acc: %.5f%%"%(100.0*correct/tot))
        if best_acc<100.0*correct/tot:
            best_acc=100.0*correct/tot
            best_epoch=epoch
        logger.info("Best epoch: %d. Best acc: %.5f%%."%(best_epoch,best_acc))
        logger.info(cf)