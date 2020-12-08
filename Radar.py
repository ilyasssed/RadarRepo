import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("Radar_Traffic_Counts.csv")                        
df_new = df.copy()

#Sortby date to get logical predictions 
df_new = df_new.sort_values(by = ["Year", "Month", "Day", "Day of Week"])


#Transform object data to integer classes

mappings = {}

for i in range(df_new.shape[1]):
    if df_new.iloc[:,i].dtypes == 'O':
        labels_list=list(df_new.iloc[:,i].unique())
        mapping = dict(zip(labels_list,range(len(labels_list))))
        mappings[df_new.columns[i]] = (mapping)
        df_new.iloc[:,i] = df_new.iloc[:,i].map(mapping)


#divide data into features and labels        
X = df_new.drop("Volume", axis = 1)
y = df_new.Volume

scaler = MinMaxScaler()
scaler.fit(np.array(y).reshape(-1,1))
y_scaled = scaler.transform(np.array(y).reshape(-1,1))

#take a small part of data to accelerate computation
X = X.iloc[:10000, :]
y = y_scaled[:10000]

scaler = MinMaxScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
X_train, X_test = X_scaled[:8000], X_scaled[8000:]
y_train, y_test = y[:8000], y[8000:]

#create a dataset class
class SelectDataset(Dataset):
    
    def __init__(self,feature,target):
        self.feature = feature
        self.target = target
    
    def __len__(self):
        return len(self.feature)
    
    def __getitem__(self,idx):
        item = self.feature[idx]
        label = self.target[idx]
        
        return item,label
    
#transform datasets into Pytorch tensors
X_train, X_test = torch.tensor(X_train), torch.tensor(X_test)
y_train, y_test = torch.tensor(y_train), torch.tensor(y_test)
batch_size = 10
test = SelectDataset(X_test,y_test)
train = SelectDataset(X_train, y_train)
train_loader = DataLoader(train, batch_size = batch_size,shuffle = False)
test_loader = DataLoader(test, batch_size = batch_size, shuffle = False)

#Define a neural network
class NN(nn.Module):
    
    def __init__(self):
        super(NN,self).__init__()
        self.fc1 = nn.Linear(11,5)
        self.fc2 = nn.Linear(5,5)
        self.fc3 = nn.Linear(5,1)
        
    def forward(self,x):

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x
    
#Define the model and the loss function and the functions for training and testing
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = NN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.MSELoss()

train_losses = []


def Train():
    
    running_loss = .0
    
    model.train()
    
    for idx, (inputs,labels) in enumerate(train_loader):
        
        inputs = inputs.to(device)
        labels = labels.float().to(device)
        optimizer.zero_grad()
        preds = model(inputs.float())
        loss = criterion(preds,labels)
        loss.backward()
        optimizer.step()
        running_loss += loss
        
    train_loss = running_loss/len(train_loader)
    train_losses.append(train_loss.detach().cpu().numpy())
    
    print(f'train_loss {train_loss}')

test_losses = []

def Test():
    
    running_loss = .0
    
    model.eval()
    
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            preds = model(inputs.float())
            loss = criterion(preds,labels)
            running_loss += loss
            
        test_loss = running_loss/len(test_loader)
        test_losses.append(test_loss.detach().cpu().numpy())
        print(f'test_loss {test_loss}')


#Do 100 epochs and see how the loss changes

epochs = 100
for epoch in range(epochs):
    print('epochs {}/{}'.format(epoch+1,epochs))
    Train()
    Test()
    
#Plot the losses
fig, ax = plt.subplots()
color = "tab:blue"
ax.plot(range(len(train_losses)), train_losses, color = color, label = "train loss")

color = "tab:red"
ax.plot(range(len(test_losses)), test_losses, color = color, label = "test loss")

legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large')

plt.show


