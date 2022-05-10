import torch
from agent.networks import CNN

class BCAgent:
    
    def __init__(self, n_classes=5, history_length=1, lr=1e-4):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.device = torch.device('cpu')

        # TODO: Define network, loss function, optimizer
        self.net = CNN(n_classes=n_classes, history_length=history_length)
        self.net.to(device=self.device)
        print("Model in device : ",self.device)

        self.criterion = torch.nn.CrossEntropyLoss().to(device=self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        

    def update(self, X_batch, y_batch):

        
        self.net.train()
        # TODO: transform input to tensors
        # X_batch = torch.from_numpy(X_batch).to(self.device)
        y_batch = torch.from_numpy(y_batch).to(self.device)

        # TODO: forward + backward + optimize
        # zero the parameter_gradients
        self.optimizer.zero_grad()

        outputs = self.predict(X_batch)
        loss = self.criterion(outputs, y_batch)
        
        loss.backward()
        self.optimizer.step()

        predictions = torch.argmax(outputs, 1)
        accuracy = 100*(predictions == y_batch).sum()/len(y_batch)

        return loss, accuracy

    def validate(self, X_batch, y_batch):

        self.net.eval()
        # TODO: transform input to tensors
        # X_batch = torch.from_numpy(X_batch).to(self.device)
        y_batch = torch.from_numpy(y_batch).to(self.device)

        # TODO: forward + backward + optimize
        # zero the parameter_gradients
        
        outputs = self.predict(X_batch)
        loss = self.criterion(outputs, y_batch)

        predictions = torch.argmax(outputs, 1)
        accuracy = 100*(predictions == y_batch).sum()/len(y_batch)

        return loss, accuracy

    def predict(self, X):
        X = torch.from_numpy(X).to(self.device)

        return self.net(X)

    def load(self, file_name):
        self.net.load_state_dict(torch.load(file_name))


    def save(self, file_name):
        torch.save(self.net.state_dict(), file_name)

