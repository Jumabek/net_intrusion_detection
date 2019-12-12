import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as utils
from torch.utils.tensorboard import SummaryWriter
import os
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import metrics

from os.path import join

SEED=2

class Softmax(nn.Module):
    def __init__(self,input_dim,num_classes,device):
        super(Softmax,self).__init__()
        self.classifier = nn.Linear(input_dim, num_classes).to(device)
        #self.classifier = self.classifier'

    def forward(self,x):
        output= self.classifier(x)         
        return output


class CNN2(nn.Module):
    def __init__(self,input_dim,num_classes,device):
        super(CNN2, self).__init__()
        # kernel
        self.input_dim = input_dim
        self.num_classes = num_classes

        conv_layers = []
        conv_layers.append(nn.Conv1d(in_channels=1,out_channels=64,kernel_size=3,padding=1)) # ;input_dim,64
        conv_layers.append(nn.BatchNorm1d(64))
        conv_layers.append(nn.ReLU(True))

        conv_layers.append(nn.Conv1d(in_channels=64,out_channels=128,kernel_size=3,padding=1)) #(input_dim,128)
        conv_layers.append(nn.BatchNorm1d(128))
        conv_layers.append(nn.ReLU(True))

        self.conv = nn.Sequential(*conv_layers).to(device)

        fc_layers = []
        fc_layers.append(nn.Linear(input_dim*128,num_classes))
        self.classifier = nn.Sequential(*fc_layers).to(device)

    def forward(self, x):
        batch_size, D = x.shape
        x = x.view(batch_size,1,D)

        x = self.conv(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x


class Net3(nn.Module):

    def __init__(self,input_dim,num_classes,device):
        super(Net3, self).__init__()
        # kernel
        print('building NN3')
        self.input_dim = input_dim
        self.num_classes = num_classes

        layers = []

        layers.append(nn.Dropout(p=0.1))
        layers.append(nn.Linear(self.input_dim, 128))
        layers.append(nn.BatchNorm1d(num_features=128)) 
        layers.append(nn.Dropout(p=0.3))
        layers.append(nn.Linear(128, 128))
        layers.append(nn.BatchNorm1d(num_features=128))
        layers.append(nn.Linear(128, self.num_classes))
        self.classifier = nn.Sequential(*layers).to(device)
        
    def forward(self, x):
        x = self.classifier(x)
        return x


class Classifier:
    def __init__(self,method,input_dim,num_classes,num_epochs,batch_size=100,lr=1e-3,reg=1e-5,runs_dir=None):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = lr
        self.reg= reg
        self.runs_dir = runs_dir
        #self.device = 'cuda'

        #self.model = nn.Linear(self.input_size, self.num_classes).to(self.device)
        if method=='softmax':
            self.device = torch.device('cuda:1')
            self.model = Softmax(input_dim,num_classes=num_classes, device=self.device)
        elif method=='cnn2':
            self.device = torch.device('cuda:2')
            self.model = CNN2(input_dim,num_classes=num_classes,device=self.device)        
        elif method=='nn3':
            self.device = torch.device('cuda:0')
            self.model = Net3(input_dim,num_classes=num_classes,device=self.device)        
        else:
            print('There is no such classifier')
            exit()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.learning_rate,betas=(0.9,0.99),eps=1e-08, weight_decay=self.reg, amsgrad=False)

    def fit(self,X,Y):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
        for dev_index, val_index in sss.split(X, Y): # runs only once
                X_dev = X[dev_index]
                Y_dev = Y[dev_index]
                X_val = X[val_index]
                Y_val = Y[val_index]  
        
        writer = SummaryWriter(self.runs_dir) 

        tensor_x = torch.stack([torch.Tensor(i) for i in X_dev]).to(self.device)
        tensor_y = torch.LongTensor(Y_dev).to(self.device) # checked working correctly

        dataset = utils.TensorDataset(tensor_x,tensor_y)        
        train_loader = utils.DataLoader(dataset,batch_size=self.batch_size) 
        N = tensor_x.shape[0]
        
        num_epochs = self.num_epochs

        model  = self.model
        best_acc = None
        best_epoch = None

        filepath = join(self.runs_dir,'checkpoint.pth')
        if os.path.isfile(filepath):
            checkpoint = self.load_checkpoint(filepath)
            best_epoch = checkpoint['epoch']
            best_batch = checkpoint['batch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            pred = self.predict(X_val)
            best_acc = metrics.balanced_accuracy_score(Y_val,pred)*100
            resume_epoch = best_epoch+1  
            resume_batch = best_batch+1
        else:
            resume_epoch = 0
            resume_batch = 0
            best_acc = -1
            best_epoch = 0

        no_improvement = 0
        print("best epoch {}, best batch {}".format(resume_epoch,resume_batch))
        print("bst acc ", best_acc)
        for epoch in range(resume_epoch,num_epochs):
            for i,(xi,yi) in enumerate(train_loader):
                if epoch==resume_epoch and i<resume_batch:
                    continue
                    
                outputs = model(xi)
                loss = self.criterion(outputs,yi)

                loss.requires_grad
                seen_so_far = self.batch_size*(epoch*len(train_loader)+i+1) # fixes issues with different batch size
                writer.add_scalar('Loss/train',loss.item(),seen_so_far)
                #batckward, optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                if (seen_so_far/self.batch_size)%50==0:
                    pred = self.predict(X_val)
                    balanced_acc = metrics.balanced_accuracy_score(Y_val,pred)*100
                    if balanced_acc > best_acc:
                        best_acc = balanced_acc
                        best_epoch = epoch
                        checkpoint = {
                        'state_dict': model.state_dict(),
                        'optimizer' : self.optimizer.state_dict(),
                        'epoch':epoch,
                        'batch': i,
                        'batch_size': self.batch_size
                        }
                        self.save(checkpoint)
                        no_improvement =0
                    else:
                        no_improvement+=1
                        if no_improvement>=10:
                            print("no improvement in accuracy for 10 iterations")
                            return
                        
                    writer.add_scalar('Accuracy/Balanced Val',balanced_acc,seen_so_far)

                    acc = metrics.accuracy_score(Y_val,pred)*100
                    writer.add_scalar('Accuracy/Val',acc,seen_so_far)
                        
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                                               .format(epoch+1, num_epochs, i+1, len(Y_dev)//self.batch_size, loss.item()))
        writer.close()

    def predict(self,x,eval_mode=False):
        tensor_x = torch.stack([torch.Tensor(i) for i in x]).to(self.device)
        bs = self.batch_size
        num_batch = x.shape[0]//bs +1*(x.shape[0]%bs!=0)

        pred = torch.zeros(0,dtype=torch.int64).to(self.device)
        
        if eval_mode:
            model = self.load_model()
        else:
            model = self.model
        model.eval()        
        
        with torch.no_grad():
            for i in range(num_batch):
                xi = tensor_x[i*bs:(i+1)*bs]
                outputs = model(xi)
                _, predi = torch.max(outputs.data,1)
                pred = torch.cat((pred,predi))

        return pred.cpu().numpy()


    def save(self,checkpoint):
        path = join(self.runs_dir,'checkpoint.pth')
        torch.save(checkpoint,path)

    
    def load_checkpoint(self,filepath):
        if os.path.isfile(filepath):
            checkpoint = torch.load(filepath)
            print("Loaded {} model trained with batch_size = {}, seen {} epochs and {} mini batches".
                format(self.runs_dir,checkpoint['batch_size'],checkpoint['epoch'],checkpoint['batch'])) 
            return checkpoint
        else:
            return None
        
            
    def load_model(self,inference_mode=True):
        filepath = join(self.runs_dir,'checkpoint.pth')
        checkpoint = self.load_checkpoint(filepath)
        
        model = self.model
        model.load_state_dict(checkpoint['state_dict'])
        
        if inference_mode:
            for parameter in model.parameters():
                parameter.requires_grad = False
            model.eval()
        return model

