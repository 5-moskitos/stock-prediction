import torch
import torch.nn as nn
from create_dataloader import create_dataloader
import os
import pickle
import sys

# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = 'cpu'


def load_model(directory, target_filename):
    file = f'{target_filename}.NS_data.h5'
    pickle_file = os.path.join(directory, file)
    print(f"pickle_file {pickle_file}")
    
    if os.path.exists(pickle_file):
        print("valid path")
        # checkpoint = 
        return torch.load(pickle_file)
    
    else:
        return None
    

def store_model(model,directory,target_filename):
    pickle_file = os.path.join(directory, f'{target_filename}.NS_data.h5')
    torch.save(model,pickle_file )

class StockPrediction(nn.Module):
    def __init__(self, input_size,hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, 
                            batch_first=True)
        
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


batch_size = 1
lookback = 60
learning_rate = 0.001
num_epochs = 100
input_size = 1
hidden_size = 128
num_layers = 4




def train_one_epoch(model,train_loader,epoch,loss_function,optimizer):
    model.train(True)
    print(f'Epoch: {epoch + 1}')
    running_loss = 0.0
    
    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        # print(x_batch.shape)
        # print(y_batch.shape)
        # return 
        # sys.exit()
        output = model(x_batch)
        loss = loss_function(output, y_batch)
        running_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_index % 100 == 99:
            avg_loss_across_batches = running_loss / 100
            print('Batch {0}, Loss: {1:.3f}'.format(batch_index+1,
                                                    avg_loss_across_batches))
            running_loss = 0.0
    print()

def validate_one_epoch(model,test_loader,loss_function):
    model.train(False)
    running_loss = 0.0
    
    for batch_index, batch in enumerate(test_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        # print(f"X :{x_batch.dtype} , {x_batch.shape} , {y_batch.shape}")
        # print(x_batch)
        # print(y_batch)
        # sys.exit()
        with torch.no_grad():
            output = model(x_batch)
            # print(f"predicted : {output}" , F"ture : {y_batch}")
            loss = loss_function(output, y_batch)
            running_loss += loss.item()
        # sys.exit()
    avg_loss_across_batches = running_loss / len(test_loader)
    
    print('Val Loss: {0:.3f}'.format(avg_loss_across_batches))
    print('***************************************************')
    print()



# dataset


def main():
    # Create a directory for pickle files
    pickle_directory = 'path_to_store_pickle_files'
    os.makedirs(pickle_directory, exist_ok=True)
    path = "/home/siddhant/3.SEM/Stock_Prediction/git/Ai-trade/database/stock_data"
    for file in os.listdir(path):
        if file.endswith('.csv'):
            csvfile_path = os.path.join(path,file)
            train_loader, test_loader = create_dataloader(csvfile_path, batch_size, lookback)
            # model = StockPrediction(input_size, hidden_size,num_layers)
            # model.to(device)
            pickle_directory = "../path_to_store_pickle_files/"
            model_name = file.split(".NS")[0]
            model = load_model(pickle_directory,model_name)
            loss_function = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            for epoch in range(num_epochs):
                train_one_epoch(model,train_loader,epoch,loss_function,optimizer)
                validate_one_epoch(model,test_loader,loss_function)
            # Save models dictionary as a pickle file
            # sys.exit()
            model_file = file.split('.csv')[0].strip() + '.h5'
            pickle_filename = os.path.join(pickle_directory, model_file)
            # pickle_filename = model_file
            model = model.to(torch.device('cpu'))
            torch.save(model,pickle_filename )
            # sys.exit()
            # with open(pickle_filename, 'wb') as pickle_file:
            #     pickle.dump(model, pickle_file)
        
        # break
                
    
if __name__ == "__main__":
    main()









