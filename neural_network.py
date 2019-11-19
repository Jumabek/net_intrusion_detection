import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.utils.data as utils
from torch.utils.tensorboard import SummaryWriter

class Net(nn.Module):

    def __init__(self,input_size,num_classes):
        super(Net, self).__init__()
        # kernel
        self.input_size = input_size
        self.num_classes = num_classes
        self.build()

    def build(self):

        self.fc1 = nn.Linear(self.input_size, 128)  
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.num_classes)
        

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class NetClassifier():
    def __init__(self, input_size,num_classes,num_epochs=5,batch_size=100,lr=5e-3,reg=2.5e-2,runs_dir=None):
        self.model = Net(input_size,num_classes)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

        self.num_epochs = num_epochs
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
                if i%10==0 and verbose:
                    pred = self.predict(x_val)
                    acc = np.mean(pred==y_val)*100
                    writer.add_scalar('Accuracy/Val',acc,seen_so_far)
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                                               .format(epoch+1, self.num_epochs, i+1, len(y)//self.batch_size, loss.item()))
        writer.close()

    def predict(self,x):
        tensor_x = torch.stack([torch.Tensor(i) for i in x]).to(self.device)
        bs = self.batch_size
        num_batch = x.shape[0]//bs +1*(x.shape[0]%bs!=0)

        pred = torch.zeros(0,dtype=torch.int64).to(self.device)
        with torch.no_grad():
            for i in range(num_batch):
                xi = tensor_x[i*bs:(i+1)*bs]
                outputs = self.model(xi)
                _, predi = torch.max(outputs.data,1)
                pred = torch.cat((pred,predi))
        
        return pred.cpu().numpy()

