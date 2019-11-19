import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as utils
from torch.utils.tensorboard import SummaryWriter

class LinearClassifier:
    def __init__(self,lossfunction,input_size,num_classes,num_epochs=5,batch_size=100,lr=5e-3,reg=2.5e-2,runs_dir=None):
        self.input_size = input_size
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = lr
        self.reg= reg
        self.lossfunction = lossfunction
        self.runs_dir = runs_dir
        self.build()

    def build(self):
        self.model = nn.Linear(self.input_size, self.num_classes)
        print("Model is created")
        if self.lossfunction=='softmax':
            self.criterion = nn.CrossEntropyLoss()
        elif self.lossfunction=='svm':
            pass
        else:
            print('Please specify loss function')
            exit()
            
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.learning_rate,betas=(0.9,0.99),eps=1e-08, weight_decay=self.reg, amsgrad=False)

    def fit(self,x,y,verbose=True):
        writer = SummaryWriter(self.runs_dir) 

        tensor_x = torch.stack([torch.Tensor(i) for i in x])
        tensor_y = torch.LongTensor(y) # checked working correctly
        dataset = utils.TensorDataset(tensor_x,tensor_y)
        train_loader = utils.DataLoader(dataset,batch_size=self.batch_size) 
        for epoch in range(self.num_epochs):
            for i,(xi,yi) in enumerate(train_loader):
                outputs = self.model(xi)
                loss = self.criterion(outputs,yi)
                seen_so_far = self.batch_size*(epoch*len(train_loader)+i+1) # fixes issues with different batch size
                writer.add_scalar('Loss/train',loss.item(),seen_so_far)
                #batckward, optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if i%100==0 and verbose:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                                               .format(epoch+1, self.num_epochs, i+1, len(y)//self.batch_size, loss.item()))
        writer.close()

    def predict(self,x):
        tensor_x = torch.stack([torch.Tensor(i) for i in x])
        bs = self.batch_size
        num_batch = x.shape[0]//bs +1*(x.shape[0]%bs!=0)

        pred = torch.zeros(0,dtype=torch.int64)
        with torch.no_grad():
            for i in range(num_batch):
                xi = tensor_x[i*bs:(i+1)*bs]
                outputs = self.model(xi)
                _, predi = torch.max(outputs.data,1)
                pred = torch.cat((pred,predi))
        
        return pred.cpu().numpy()


