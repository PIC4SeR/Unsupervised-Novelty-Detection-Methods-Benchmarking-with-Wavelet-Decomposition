import torch.nn as nn
from torch.optim import Adam
import torch, numpy as np, os
from torch.utils.data import DataLoader
from tqdm import tqdm , trange
from .printlog import print
import pathlib


class Autoencoder(nn.Module):
    def __init__(self,midlay=50,latent=25):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(70, midlay),
            nn.ReLU(),
            nn.Linear(midlay, latent),
            nn.LeakyReLU())
        
        self.decoder = nn.Sequential(
            nn.Linear(latent, midlay),
            nn.ReLU(),
            nn.Linear(midlay, 70),
            nn.LeakyReLU())
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train_nn(model,data,epochs=100,batch_size=32,lr=0.001,onnx=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()
    dataloader = DataLoader(torch.tensor(data, dtype=torch.float32), batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    for epoch in trange(epochs, desc=f'Training the AE', position=2, leave=False):
        for data in dataloader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)

            loss = criterion(output, data)            
            loss.backward()
            optimizer.step()

        #print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, epochs, loss.data), show=False)
    torch.save(model.state_dict(), 'model.pth')

    if onnx:
        onnx_program = torch.onnx.dynamo_export(model, data)
        onnx_program.save('model.onnx')

    return model

def predict_nn(model,data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input = torch.tensor(data, dtype=torch.float32).to(device)
    with torch.no_grad(): 
        model.to(device)
        model.eval()
        output = model(input)
        enc_out = model.encoder(input)
    error = torch.abs(output - input)
    return output,enc_out, error



def best_aemodel(model,X_train,X_test,ae_model,lr,batch_size,n_samp_variance,AE_Type, savepath):
    best_variance = 1e12
    for _ in trange(20, desc=f'Trying different initializations', position=1, leave=False):
        ae_model = train_nn(ae_model,X_train,epochs=150,batch_size=batch_size,lr=lr)    
        _, train_latent_representation, _ = predict_nn(ae_model,X_train)
        _, test_latent_representation, _ = predict_nn(ae_model,X_test)

        model.train(train_latent_representation.cpu().numpy())
        model.predict(train_latent_representation.cpu().numpy())
        NovMetric = model.evaluate(test_latent_representation.cpu().numpy())

        Variance = np.var(NovMetric[:n_samp_variance])
        if Variance < best_variance:
            if AE_Type == 0:
                path = os.path.join(savepath,f'aemodel_reduced_{model}.pth')
                torch.save(ae_model.state_dict(), path)
            else:
                path = os.path.join(savepath,f'aemodel_augmented_{model}.pth')
                torch.save(ae_model.state_dict(), path)
            best_variance = Variance
