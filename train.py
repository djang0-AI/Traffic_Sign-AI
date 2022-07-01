import torch
import torch.nn as nn

from model import Net
from utils import test
from trainer import Trainer
from data_loader import get_data, get_transforms


def main():
    # setting hyperparameter
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    crit = nn.CrossEntropyLoss()
    epochs = 100

    # get data from get_data
    traindata, valdata, testdata = get_data()

    # setting model
    model = Net(43, 2, 1024, 64)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)
    
    # trainer
    trainer = Trainer(model, crit, optimizer, epochs, device)
    trainer.train(traindata, valdata)
    torch.save({
        'model': trainer.model.state_dict()
    }, 'best_model.pth')

    test(testdata, model, device)    

if __name__ == "__main__":
    main()
