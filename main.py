
from torch.optim import lr_scheduler
import torch.optim as optim
from torch import nn
import os
import torch
import torchvision.models as models

from CNN import *


global device, batch_size, num_workers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

enable_train = True
enable_test = True



batch_size = 20
num_workers = 0

lr = 0.0001
num_epochs = 40
model_input_size = 224
model_flage = True


dataset_path = "../input_data/"
path = "../models/"

model_type = "ResNet34"
if model_type not in os.listdir(path):
    os.mkdir(path+model_type)



for i in range(10):
    model_name = model_type+"_Adam_" + str(lr).split(".")[-1]+"_"+str(i) +".pth"


    save_path, output_file = create_out_txt_file(path,model_type,model_name,lr)



    dataloader,l,class_names = load_data(model_input_size,output_file,batch_size,num_workers,dataset_path=dataset_path)


    model = models.resnet34(pretrained=True)

# Get number of layers and override last layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model.to(device)
# set up loss function and optimizer function
    criterion = nn.CrossEntropyLoss()
    optimizer_out = optim.Adam(model.parameters(), lr=lr)
    
    exp_lr_scheduler_out = lr_scheduler.StepLR(optimizer_out, step_size=10, gamma=0.1)

    if enable_train:
        out_model, best_train_acc, best_train_loss, best_acc, best_loss = \
            train_model(model, criterion, dataloader, l, optimizer_out, exp_lr_scheduler_out, device,
                        save_path,output_file, model_flage, num_epochs=num_epochs)

    if model_name in os.listdir(path + model_type) and enable_test:
        model.load_state_dict(torch.load(save_path))
        test_acc, test_loss = test_model(model, criterion, dataloader, l, device, output_file,len(class_names))
        model.to("cpu")
        torch.save(model.state_dict(), save_path)

