#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as image
import torchvision
import torch
import sklearn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch import nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset 
import os

import os
import PIL.Image as Image
from torchvision import transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchinfo import summary

import time
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, auc
from tqdm.auto import tqdm
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import densenet121, DenseNet121_Weights
import wandb
print(f"Device type: {device}")


# In[2]:


config = {"image_size": 244,
          "batch_size": 64,
          "epochs": 100,
         "learning_rate":0.001,
         "data_fraction": 7,
         "freez_till":10}
# wandb.login()
# wandb.init(project="Retinal_fundus_dr", config = config)


# In[3]:


train_csv= '../data/kaggle_data/trainLabels.csv'
train_dir = '../data/kaggle_data/train/'
test_dir = '../data/kaggle_data/test/'
Saved_model_dir = '../trained model/RF_resnet_bin.pt'
label_dict = {"0": "No DR",
            "1": "Mild",
           "2":"Moderate",
           "3":"Severe",
           "4":"Proliferative DR"}
class2idx = {v:k for k,v in label_dict.items()}


# In[4]:



class Retinal_Data(Dataset):
    def __init__(self, annotation_file, img_dir, transform = None, binary =None):
        self.image_label = pd.read_csv(annotation_file)
        self.image_label['binary'] = self.image_label.level.apply(lambda x: 0 if x==0 else 1)
        
        self.image_label_1 = self.image_label[self.image_label['binary']==1]
        self.image_label_1 = self.image_label_1.sample(len(self.image_label_1)//config['data_fraction'],random_state=42)
        
        self.image_label_0 = self.image_label[self.image_label['binary']==0]
        self.image_label_0 = self.image_label_0.sample(len(self.image_label_1),random_state=42)
        
        self.image_label = pd.concat([self.image_label_1, self.image_label_0], axis = 0)
        self.image_label['img_path'] = self.image_label['image'].apply(lambda x: img_dir+x+'.jpeg')

        
        self.img_dir = img_dir
        self.transform = transform
        self.binary = binary
    
    
    def __len__(self):
        return len(self.image_label)
    
    def __getitem__(self, idx):
        "it should return transformed image arr and corusponding label"
        image_path = self.image_label.iloc[idx, 3]
        img = torchvision.io.read_image(image_path).float()
        
        if transform:
            img = transform(img)
        if self.binary:
            return img, torch.tensor(self.image_label.iloc[idx, 2]).float()
        else:
            return img, torch.tensor(self.image_label.iloc[idx, 1]).float()
        
        
        


# In[5]:


transform = transforms.Compose([transforms.Resize((config["image_size"],config["image_size"])),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomRotation(5)])


train_data = Retinal_Data(train_csv,train_dir, transform, binary = True)

## Splitting the datasets into train and valid 

train_ln = int(len(train_data)*.8)

valid_ln = len(train_data)-train_ln

train_data, valid_data = random_split(train_data, [train_ln, valid_ln])

print(f'Total training data: {len(train_data)+len(valid_data)}')
print(f'Training_split : {len(train_data)}')
print(f'Validation_split: {len(valid_data)}')


train_loader = DataLoader(train_data, batch_size = config["batch_size"], num_workers = 96, shuffle  = True)
valid_loader = DataLoader(valid_data,batch_size = config["batch_size"],num_workers = 96, shuffle  = True)


# In[6]:


class Rf_Model(nn.Module):
    
    def __init__(self):
        super(Rf_Model, self).__init__()
        self.densnet  = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        
        for count,(name,child) in enumerate(self.densnet.features.named_children()):
            if count<config["freez_till"]:
                print(f"freezing the following layer: {name}")
                for param in child.parameters():
                    param.requires_grad=False
                    
        self.densnet.classifier = nn.Linear(in_features=1024, out_features=64)
        self.fc =  nn.Linear(in_features=64, out_features=1)

        
    def forward(self, x):
        x = self.densnet(x)
        x = self.fc(x)  
        return x
        

    def train_model(self,model, train_loader, optimizer, loss_fn):
        train_loss, running_loss, train_acc = 0,0,0

        model.train()  

        for batch_num, (xa, yb) in tqdm(enumerate(train_loader)):
            if torch.cuda.is_available():
                print("pushing data into GPU")
                xa = xa.to(device)
                yb = yb.to(device)

            # zero the parameter gradients
            optimizer.zero_grad() 

            # forward + backward + optimize
            batch_preds = model(xa).squeeze()
            batch_loss = loss_fn(batch_preds,yb)
            batch_loss.backward()
            optimizer.step()

            # accuracy calculation
            running_loss += batch_loss.item()
            batch_preds_Bin = (batch_preds>0.5).float()
            batch_acc = accuracy_score(yb.cpu(),batch_preds_Bin.cpu())
            train_acc+=batch_acc
            # print("running_loss",running_loss/(batch_num+1))
            # print("batch_acc",batch_acc)
        train_loss = running_loss/len(train_loader)
        train_acc = train_acc/len(train_loader)
        return train_loss, train_acc
    
    
    def validate_model(self, model, valid_loader, loss_fn):
        with torch.no_grad():
            valid_loss, running_loss, valid_acc = 0,0,0
            model.eval()
            for batch_num, (xa, yb) in tqdm(enumerate(valid_loader)):
                if torch.cuda.is_available():
                    print("pushing data into GPU")
                    xa = xa.to(device)
                    yb = yb.to(device)

                # forward
                batch_preds = model(xa).squeeze()
                batch_loss = loss_fn(batch_preds,yb)

                # accuracy calculation
                running_loss += batch_loss.item()
                batch_preds_Bin = (batch_preds>0.5).float()
                batch_acc = accuracy_score(yb.cpu(),batch_preds_Bin.cpu())
                valid_acc+=batch_acc
            valid_loss = running_loss/len(valid_loader)
            valid_acc = valid_acc/len(valid_loader)
        return valid_loss, valid_acc


# In[7]:


model  = Rf_Model()
inp = torch.rand((config["batch_size"],3,config["image_size"], config["image_size"]))
out = model(inp)


# In[8]:


summary(model, input_size=(20,3, 224,224))


# In[ ]:





# In[ ]:


accuracy_stats = {"train": [],
                 "val":[]}
loss_stats = {"running":[],
              "train": [],
             "val": []}
model = Rf_Model()

if torch.cuda.is_available():
    model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = config['learning_rate'])
BCE_loss = nn.BCEWithLogitsLoss()

# print("idensnetize", config["image_size"])
# print("batch size", config["batch_size"])
print(f"Total number of batches: {len(train_loader)}") 

print("Model training started...")

start_epoch_time = time.time()
for epoch in tqdm(range(config['epochs'])):
    #training
    train_loss, train_acc = model.train_model(model, train_loader, optimizer, BCE_loss)

    # Validation    
    valid_loss, valid_acc = model.validate_model(model, valid_loader, BCE_loss)
    
    if epoch ==0:
        single_epoch_time = time.time()-start_epoch_time
        # wandb.log({"single_epoch_time":int(single_epoch_time)})
        
    print(single_epoch_time)
    loss_stats['train'].append(train_loss)
    accuracy_stats["train"].append(train_acc)
    loss_stats['val'].append(valid_loss)
    accuracy_stats["val"].append(valid_acc)

    log_dict = {"epoch":epoch+1,
               "train_loss":train_loss,
             "train_acc":train_acc,
              "valid_loss":valid_loss,
              "valid_acc":valid_acc,
              }  
    print(log_dict)
    
    if log_dict["train_loss"] <.05:
        print("Saving model....")
        torch.save(model.state_dict(), Saved_model_dir)
        print("model saved")
        
#     wandb.log(log_dict)

# wandb.finish()


# In[ ]:


#wandb.finish()


# In[ ]:


plt.plot(accuracy_stats["train"])
plt.plot(accuracy_stats["val"])
plt.show()


# In[ ]:


plt.plot(accuracy_stats["train"])
plt.plot(loss_stats["train"])
plt.show()


# In[ ]:


plt.plot(accuracy_stats["val"])
plt.plot(loss_stats["val"])
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


`


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




