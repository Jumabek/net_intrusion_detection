import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.utils.data as utils
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics
from os.path import join

class Net3(nn.Module):

    def __init__(self,input_size,num_classes,use_batchnorm=False):
        super(Net3, self).__init__()
        # kernel
        print('building NN3')
        self.input_size = input_size
        self.num_classes = num_classes
        self.use_batchnorm = use_batchnorm
        self.build()

    def build(self):

        self.drop1 = nn.Dropout(p=0.1)
        self.fc1 = nn.Linear(self.input_size, 128)
        self.bn1 = nn.BatchNorm1d(num_features=128) 
        self.drop2 = nn.Dropout(p=0.3) 
        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(num_features=128)
        self.fc3 = nn.Linear(128, self.num_classes)

        
    def forward(self, x):
        if self.use_batchnorm:
            x = F.relu(self.drop1(self.bn1(self.fc1(x))))
            x = F.relu(self.drop2(self.bn2(self.fc2(x))))
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Net5(nn.Module):

    def __init__(self,input_size,num_classes,use_batchnorm=False):
        super(Net5, self).__init__()
        # kernel
        self.input_size = input_size
        self.num_classes = num_classes
        self.use_batchnorm = use_batchnorm
        
        layers = []
        layers.append(nn.Linear(input_size,128))

        layers.append(nn.BatchNorm1d(128))
        layers.append(nn.ReLU(True))
        layers.append(nn.Linear(128,256))
        
        layers.append(nn.BatchNorm1d(256))
        layers.append(nn.Dropout(p=0.3))
        layers.append(nn.ReLU(True))
        layers.append(nn.Linear(256,256))
        
        layers.append(nn.BatchNorm1d(256))
        layers.append(nn.Dropout(p=0.4))
        layers.append(nn.ReLU(True))
        layers.append(nn.Linear(256,128))

        layers.append(nn.BatchNorm1d(128))
        layers.append(nn.Dropout(p=0.5))
        layers.append(nn.ReLU(True))        
        layers.append(nn.Linear(128,num_classes))

        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

class NetClassifier():
    def __init__(self, input_size,num_classes,num_epochs=5,batch_size=100,lr=5e-3,reg=2.5e-2,runs_dir=None,use_batchnorm=False):
        self.model = Net5(input_size,num_classes,use_batchnorm)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

        self.num_epochs = num_epochs
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.lr = lr
        self.reg = reg
        self.runs_dir = runs_dir

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=lr,betas=(0.9,0.99),eps=1e-08, weight_decay=reg, amsgrad=False)

  
    def fit(self,x,y,x_val,y_val,verbose=True):
        writer = SummaryWriter(self.runs_dir) 

        tensor_x = torch.stack([torch.Tensor(i) for i in x]).to(self.device)
        tensor_y = torch.LongTensor(y).to(self.device) # checked working correctly

        dataset = utils.TensorDataset(tensor_x,tensor_y)
        train_loader = utils.DataLoader(dataset,batch_size=self.batch_size) 
        model = self.model
        best_acc = -1

        for epoch in range(self.num_epochs):
            for i,(xi,yi) in enumerate(train_loader):
                outputs = model(xi)
                loss = self.criterion(outputs,yi)
                seen_so_far = self.batch_size*(epoch*len(train_loader)+i+1) # fixes issues with different batch size
                writer.add_scalar('Loss/train',loss.item(),seen_so_far)
                #batckward, optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if i%50==0 :
                    raw_pred,pred = self.predict(x_val)
                    balanced_acc = metrics.balanced_accuracy_score(y_val,pred)*100
                    if balanced_acc>best_acc:
                        best_acc = balanced_acc
                        
                        checkpoint = {
                        'state_dict': model.state_dict(),
                        'optimizer' : self.optimizer.state_dict(),
                        'epoch':epoch,
                        'batch': i,
                        'batch_size': self.batch_size
                        }
                        self.save(checkpoint) 
                    writer.add_scalar('Accuracy/Balanced Val',balanced_acc,seen_so_far)

                    acc = metrics.accuracy_score(y_val,pred)*100
                    writer.add_scalar('Accuracy/Val',acc,seen_so_far)

                    if verbose:
                        print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                                               .format(epoch+1, self.num_epochs, i+1, len(y)//self.batch_size, loss.item()))
        writer.close()

    def predict(self,x,eval_mode=False):
        tensor_x = torch.stack([torch.Tensor(i) for i in x]).to(self.device)
        bs = self.batch_size
        num_batch = x.shape[0]//bs +1*(x.shape[0]%bs!=0)

        pred = torch.zeros(0,dtype=torch.int64).to(self.device)
        raw_pred = torch.zeros((0,self.num_classes)).to(self.device)
        if eval_mode:
            model = self.load_checkpoint()
        else:
            model = self.model
        model.eval()

        with torch.no_grad():
            for i in range(num_batch):
                xi = tensor_x[i*bs:(i+1)*bs]
                outputs = model(xi) # N,C
                _, predi = torch.max(outputs.data,1)
                pred = torch.cat((pred,predi))
                raw_pred = torch.cat((raw_pred,outputs.data))
        return raw_pred.cpu().numpy(),pred.cpu().numpy()

    def save(self,checkpoint):
        path = join(self.runs_dir,'checkpoint.pth')
        torch.save(checkpoint,path)


    def load_checkpoint(self,inference_mode=True):
        filepath = join(self.runs_dir,'checkpoint.pth')
        checkpoint = torch.load(filepath)
        model = self.model
        model.load_state_dict(checkpoint['state_dict'])
        
        print("Loaded {} model trained with batch_size = {}, seen {} epochs and {} mini batches".
            format(self.runs_dir,checkpoint['batch_size'],checkpoint['epoch'],checkpoint['batch']))
    
        if inference_mode:
            for parameter in model.parameters():
                parameter.requires_grad = False
            model.eval()
        return model