import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader


def get_symmetric():
    symmetric = {'00': False, '01': False, '02': False, '03': False, '04': False, '05': False, '06': False, '07': False, '08': False, '09': False, '10': False, '11': True, '12': True, '13': True, '14': False, '15': True, '16': False,
             '17': True, '18': True, '19': False, '20': False, '21': False, '22': True, '23': False, '24': False, '25': False, '26': True, '27': False, '28': False, '29': False, '30': True, '31': False, '32': False, '33': False,
             '34': False, '35': True, '36': False, '37': False, '38': False, '39': False, '40': False, '41': False,'42': False}
    
    return symmetric


def get_mirrors():
    mirrors  = {'00': False, '01': False, '02': False, '03': False, '04': False, '05': False, '06': False, '07': False, '08': False, '09': False, '10': False, '11': False, '12': False, '13': False, '14': False, '15': False, '16': False,
            '17': False, '18': False, '19': '20', '20': '19', '21': False, '22': False, '23': False, '24': False, '25': False, '26': False, '27': False, '28': False, '29': False, '30': False, '31': False, '32': False, '33': '34',
            '34': '33', '35': False, '36': '37', '37': '36', '38': '39', '39': '38', '40': False, '41': False, '42': False,}

    return mirrors


def plot_loss(df, train_losses, val_losses, train_accuracy, val_accuracy):
    df = pd.DataFrame({"Train Loss": train_losses, "Validation Loss": val_losses, "Accuracy Train":train_accuracy,"Accuracy Validation":val_accuracy })
    df["Epoch"] = df.index +1
    df = pd.melt(df[["Accuracy Train", "Accuracy Validation","Epoch"]], id_vars=['Epoch'], var_name = " ")
    fig = plt.figure(figsize  = (10,8))
    sns.lineplot(data = df, x = "Epoch", y = "value", hue = " ")


def test(testdata, model, device):
    testloader = DataLoader(testdata, batch_size= 512, shuffle=False, drop_last= False)

    testdata.classes = testdata.classes.to(device)
    correct_test = 0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            inputs, labels =  data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _,pred_idxs = torch.topk(outputs, 1)
            correct_test += torch.eq(labels, pred_idxs.squeeze()).sum().item()
        print(correct_test/len(testdata))
