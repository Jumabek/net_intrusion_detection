import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.utils.data as utils
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics
from os.path import join


class CNN5(nn.Module):

    def __init__(self,input_size,num_classes,use_batchnorm=False):
        super(CNN5, self).__init__()
        # kernel
        self.input_size = input_size 
        self.num_classes = num_classes
        self.use_batchnorm = use_batchnorm
        
        conv_layers = []
        conv_layers.append(nn.Conv1d(in_channels=1,out_channels=64,kernel_size=3,padding=1))
        conv_layers.append(nn.ReLU(True))
        conv_layers.append(nn.Conv1d(in_channels=64,out_channels=128,kernel_size=3,padding=1)) #(input_size,128)
        conv_layers.append(nn.ReLU(True))
        self.conv = nn.Sequential(*conv_layers)

        layers = []
        layers.append(nn.Linear(input_size*128,num_classes))
        self.classifier = nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x

class CNNClassifier():
    def __init__(self, input_size,num_classes,num_epochs=5,batch_size=100,lr=5e-3,reg=2.5e-2,runs_dir=None,use_batchnorm=False):
        self.model = CNN5(input_size,num_classes,use_batchnorm)
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
                # xi (bs,D)
                batch_size, D = xi.shape
                xi = xi.view(batch_size,1,D)
                
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
                batch_size, D = xi.shape
                xi = xi.view(batch_size,1,D)
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
        
        print("Loaded model with has batch_size = {}, seen {} epoch and {} batch".
            format(checkpoint['batch_size'],checkpoint['epoch'],checkpoint['batch']))
    
        if inference_mode:
            for parameter in model.parameters():
                parameter.requires_grad = False
            model.eval()
        return model