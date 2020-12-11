import pandas as pd
import numpy as np
import torch
import torch.nn as nn
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

scaler = MinMaxScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
#take a small part of data to accelerate computation

X_scaled = X_scaled[:10000, :]
y = y_scaled[:10000]    


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


#Do 500 epochs and see how the loss changes

epochs = 300
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

#Now we're going to change the MSE loss to the Huber Loss

model1 = NN().to(device)
optimizer = torch.optim.Adam(model1.parameters(), lr=1e-5)
criterion = nn.SmoothL1Loss()

train_losses_huber = []


def Train_huber():
    
    running_loss = .0
    
    model1.train()
    
    for idx, (inputs,labels) in enumerate(train_loader):
        
        inputs = inputs.to(device)
        labels = labels.float().to(device)
        optimizer.zero_grad()
        preds = model1(inputs.float())
        loss = criterion(preds,labels)
        loss.backward()
        optimizer.step()
        running_loss += loss
        
    train_loss = running_loss/len(train_loader)
    train_losses_huber.append(train_loss.detach().cpu().numpy())
    
    print(f'train_loss {train_loss}')

test_losses_huber = []

def Test_huber():
    
    running_loss = .0
    
    model1.eval()
    
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            preds = model1(inputs.float())
            loss = criterion(preds,labels)
            running_loss += loss
            
        test_loss = running_loss/len(test_loader)
        test_losses_huber.append(test_loss.detach().cpu().numpy())
        print(f'test_loss {test_loss}')
    

#Do 100 epochs using this new loss
epochs = 300
for epoch in range(epochs):
    print('epochs {}/{}'.format(epoch+1,epochs))
    Train_huber()
    Test_huber()
    
#Now we define a convolutional neural network

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1d = nn.Conv1d(1,32,kernel_size= 3, stride = 1)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool1d(3, stride=1)
        self.fc1 = nn.Linear(7 * 32,50)
        self.fc2 = nn.Linear(50,1)
        
    def forward(self,x):
        x = self.conv1d(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(10,-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        
        return x

model_CNN = CNN().to(device)
optimizer = torch.optim.Adam(model_CNN.parameters(), lr=1e-5)
criterion = nn.SmoothL1Loss()

train_losses_CNN = []


def Train_CNN():
    
    running_loss = .0
    
    model_CNN.train()
    
    for idx, (inputs,labels) in enumerate(train_loader):
        inputs = inputs.view(-1,1,11)
        inputs = inputs.to(device)
        labels = labels.float().to(device)
        optimizer.zero_grad()
        preds = model_CNN(inputs.float())
        loss = criterion(preds,labels)
        loss.backward()
        optimizer.step()
        running_loss += loss
        
    train_loss = running_loss/len(train_loader)
    train_losses_CNN.append(train_loss.detach().cpu().numpy())
    
    print(f'train_loss {train_loss}')

test_losses_CNN = []

def Test_CNN():
    
    running_loss = .0
    
    model_CNN.eval()
    
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.view(-1,1,11)
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            preds = model_CNN(inputs.float())
            loss = criterion(preds,labels)
            running_loss += loss
            
        test_loss = running_loss/len(test_loader)
        test_losses_CNN.append(test_loss.detach().cpu().numpy())
        print(f'test_loss {test_loss}')

epochs = 300
for epoch in range(epochs):
    print('epochs {}/{}'.format(epoch+1,epochs))
    Train_CNN()
    Test_CNN()
    
#Plot the losses
fig, ax = plt.subplots(figsize = (10,8))
color = "tab:blue"
ax.plot(range(len(train_losses_CNN)), train_losses_CNN, color = color, label = "train loss")

color = "tab:red"
ax.plot(range(len(test_losses_CNN)), test_losses_CNN, color = color, label = "test loss")

ax.set_xlabel("Epochs")
ax.set_ylabel("Losses")
legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large')

plt.show