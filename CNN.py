
from torchvision import transforms, datasets
import time
import copy
import torch
from PIL import Image



def create_out_txt_file(path,model_type,model_name,lr):
    open(path + model_type + "/" + model_name.split(".")[0] + ".txt", "w").close()
    output_file = open(path + model_type + "/" + model_name.split(".")[0] + ".txt", 'a')

    output_file.write("lr = " + str(lr) + "\n" + "used model " + model_name.split("_")[0] + "\n")

    save_path = path + model_type + "/" + model_name
    return save_path, output_file

def load_data(new_size,output_file,batch_size,num_workers,dataset_path = "dataset/",n_train = 0.6,n_val = 0.2,n_test = 0.2):

    test_transform = transforms.Compose([transforms.Resize(new_size), transforms.CenterCrop(new_size),transforms.ToTensor()])


    Full_data = datasets.ImageFolder(dataset_path,transform=test_transform)

    train_size = int(n_train * len(Full_data))
    valid_size = int(n_val * len(Full_data))
    test_size = int(n_test * len(Full_data))

    print(len(Full_data))
    print(train_size,valid_size,test_size)
    diff = len(Full_data) - train_size - valid_size - test_size

    if diff == 1:
        train_size += 1
    elif diff == 2:
        train_size +=1
        valid_size +=1

    print(train_size,valid_size,test_size)
    output_file.write("train size is " + str(train_size) + " valid size is " + str(valid_size) + " test size is " + str(test_size) + "\n")
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(Full_data, [train_size , valid_size, test_size])

    if n_train > 0:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,num_workers=num_workers,shuffle = True)
    else:
        train_loader = 0
    if n_val > 0:
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size,num_workers=num_workers,shuffle = True)
    else:
        valid_loader = 0
    if n_test > 0:
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size , num_workers=num_workers,shuffle = True)
    else:
        test_loader = 0
    class_names = Full_data.classes

    print(class_names)
    output_file.write("0 is for " + str(class_names[0]) + " and 1 is for " + str(class_names[1])+"\n")



    l = {"train" : len(train_dataset) , "val" : len(valid_dataset), "test" : len(test_dataset)}

    dataloader = {"train" : train_loader , "val" : valid_loader, "test" : test_loader}

    return dataloader,l,class_names



def train_model(model, criterion, dataloader1, l,optimizer, scheduler, device,
                save_path ,output_file,model_flage=True,num_epochs=20):



    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        output_file.write('Epoch {}/{}'.format(epoch, num_epochs - 1))
        output_file.write("\n")
        print('-' * 50)
        output_file.write('-' * 50)
        output_file.write("\n")
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                # scheduler.step()
                model.train()  # Set model to training mode

            else:
                model.eval()


            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs1, labels1 in dataloader1[phase]:

                inputs1 = inputs1.to(device)

                target = labels1.to(device)

                # print(inputs1)
                # print(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    if model_flage:
                        outputs = model(inputs1)
                    else:
                        if phase == 'train':
                            outputs = model(inputs1).logits
                        else:
                            outputs = model(inputs1)

                    _, preds = torch.max(outputs, 1)

                    loss = criterion(outputs, target)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                # statistics
                running_loss += loss.item() * inputs1.size(0)
                running_corrects += torch.sum(preds == target.data)

            epoch_loss = running_loss / l[phase]
            epoch_acc = running_corrects.double() / l[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            output_file.write('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            output_file.write("\n")

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_loss = epoch_loss
                best_train_acc = train_acc
                best_train_loss = train_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), save_path)
                print("Saving model>>>>>>")
                output_file.write("Saving model>>>>>>")
                output_file.write("\n")
            elif phase == 'train':
                train_acc = epoch_acc
                train_loss = epoch_loss


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    output_file.write('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    output_file.write("\n")
    output_file.write('Best val Acc: {:4f}'.format(best_acc))
    output_file.write("\n")

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_train_acc, best_train_loss, best_acc, best_loss







def calculate_conf_matrix(res,pred,num_classes):
    if num_classes == 2:
        total_1 = int(sum(res == 1))
        total_0 = int(sum(res == 0))
        true_1 = int(sum(torch.logical_and(res == 1, pred == 1)))
        true_0 = int(sum(torch.logical_and(res == 0, pred == 0)))
        false_1 = total_1 - true_1
        false_0 = total_0 - true_0
        return true_1,false_1,true_0,false_0
    elif num_classes == 3:
        true_1 = int(sum(torch.logical_and(res==1,pred==1)))
        true_2 = int(sum(torch.logical_and(res == 2, pred == 2)))
        true_0 = int(sum(torch.logical_and(res == 0, pred == 0)))
        false_1_0 = int(sum(torch.logical_and(res==1,pred==0)))
        false_1_2 = int(sum(torch.logical_and(res == 1, pred == 2)))
        false_2_0 = int(sum(torch.logical_and(res == 2, pred == 0)))
        false_2_1 = int(sum(torch.logical_and(res == 2, pred == 1)))
        false_0_1 = int(sum(torch.logical_and(res == 0, pred == 1)))
        false_0_2 = int(sum(torch.logical_and(res == 0, pred == 2)))
        return true_2, false_2_0,false_2_1,true_1, false_1_0,false_1_2, true_0, false_0_1,false_0_2



def show_conf_matrix(output_file,true_1, false_1_0, true_0, false_0_1,num_classes,
                         false_0_2=0,false_1_2=0,true_2=0, false_2_0=0, false_2_1=0,n=2):
    if num_classes == 2:
        print("Confusion Matrix: ")
        print(" " * 10 + "    0" + " " * (len(str(true_0))) + "| 1")
        print(" " * 10 + "_" * 14)
        print(" " * 10 + " 0| " + str(true_0) + " " * (n - len(str(true_0))) + " | " + str(false_1_0))
        print(" " * 10 + "_" * 14)
        print(" " * 10 + " 1| " + str(false_0_1) + " " * (n - len(str(false_0_1))) + " | " + str(true_1))
        print(" " * 10 + "_" * 14)
        output_file.write("\n" + "Confusion Matrix: " + "\n")
        output_file.write(" " * 10 + "    0" + " " * (len(str(true_0))) + "| 1\n")
        output_file.write(" " * 10 + "_" * 14 + "\n")
        output_file.write(" " * 10 + " 0| " + str(true_0) + " " * (n - len(str(true_0))) + " | " + str(false_1_0) + "\n")
        output_file.write(" " * 10 + "_" * 14 + "\n")
        output_file.write(" " * 10 + " 1| " + str(false_0_1) + " " * (n - len(str(false_0_1))) + " | " + str(true_1) + "\n")
        output_file.write(" " * 10 + "_" * 14 + "\n")
    elif num_classes == 3:
        print("Confusion Matrix: ")
        print(" " * 10 + "    0" + " " * (len(str(true_0))) + "| 1"+ " " * (len(str(false_1_0))) + " | 2")
        print(" " * 10 + "_" * 26)
        print(" " * 10 + " 0| " + str(true_0) + " " * (n - len(str(true_0))) + " | " + str(false_1_0)
              + " " * (n - len(str(false_1_0))) + " | " + str(false_2_0))
        print(" " * 10 + "_" * 26)
        print(" " * 10 + " 1| " + str(false_0_1) + " " * (n - len(str(false_0_1))) + " | " + str(true_1)
              + " " * (n - len(str(true_1))) + " | " + str(false_2_1))
        print(" " * 10 + "_" * 26)
        print(" " * 10 + " 2| " + str(false_0_2) + " " * (n - len(str(false_0_2))) + " | " + str(false_1_2)
              + " " * (n - len(str(false_1_2))) + " | " + str(true_2))
        print(" " * 10 + "_" * 26)
        output_file.write("\n" + "Confusion Matrix: " + "\n")
        output_file.write(" " * 10 + "    0" + " " * (len(str(true_0))) + "| 1"+ " " * (len(str(false_1_0))) + " | 2\n")
        output_file.write(" " * 10 + "_" * 26 + "\n")
        output_file.write(" " * 10 + " 0| " + str(true_0) + " " * (n - len(str(true_0))) + " | " + str(false_1_0)
                          + " " * (n - len(str(false_1_0))) + " | " + str(false_2_0)+ "\n")
        output_file.write(" " * 10 + "_" * 26 + "\n")
        output_file.write(" " * 10 + " 1| " + str(false_0_1) + " " * (n - len(str(false_0_1))) + " | " + str(true_1)
                          + " " * (n - len(str(true_1))) + " | " + str(false_2_1)+ "\n")
        output_file.write(" " * 10 + "_" * 26 + "\n")
        output_file.write(" " * 10 + " 2| " + str(false_0_2) + " " * (n - len(str(false_0_2))) + " | " + str(false_1_2)
                          + " " * (n - len(str(false_1_2))) + " | " + str(true_2) + "\n")
        output_file.write(" " * 10 + "_" * 26 + "\n")



def test_model(model,criterion,dataloader1, l,device,output_file,num_classes):
    print("*" * 50)
    output_file.write("*" * 50)
    output_file.write("\n")
    print("Testing...")
    output_file.write("Testing...")
    output_file.write("\n")
    test_loss = 0.0
    correct = 0
    d = 0
    if num_classes == 2:
        true_1, false_1, true_0, false_0 = 0,0,0,0
    elif num_classes == 3:
        true_2, false_2_1, false_2_0,true_1, false_1_0, false_1_2, true_0 ,false_0_1,false_0_2\
            = 0,0,0,0,0,0,0,0,0
    model.eval()

    # iterate over test data
    for inputs, target in dataloader1["test"]:
        # move tensors to GPU if CUDA is available


        inputs = inputs.to(device)
        target = target.to(device)



        output = model(inputs)
        # calculate the batch loss
        loss = criterion(output, target)
        # update test loss
        test_loss += loss.item()*inputs.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)

        correct += (torch.sum(pred == target.data))
        d += pred.shape[0]

        # print("pred   = ",pred)
        # print("target = ", target)
        if num_classes == 2:
            temp_true_1, temp_false_1, temp_true_0, temp_false_0 = calculate_conf_matrix(target, pred,num_classes)
            true_0 += temp_true_0
            true_1 += temp_true_1
            false_0 += temp_false_0
            false_1 += temp_false_1
        elif num_classes == 3:
            temp_true_2, temp_false_2_0, temp_false_2_1, temp_true_1, temp_false_1_0, \
            temp_false_1_2, temp_true_0, temp_false_0_1, temp_false_0_2 = calculate_conf_matrix(target, pred,num_classes)
            true_0 += temp_true_0
            true_1 += temp_true_1
            true_2 += temp_true_2
            false_0_1 += temp_false_0_1
            false_0_2 += temp_false_0_2
            false_1_0 += temp_false_1_0
            false_1_2 += temp_false_1_2
            false_2_0 += temp_false_2_0
            false_2_1 += temp_false_2_1
    epoch_loss = test_loss / l["test"]
    correct = (int(correct)/d)*100
    # print(correct)
    print("Test Loss: " + str(epoch_loss))
    print('Test Accuracy: {:.6f}\n'.format(correct))
    output_file.write("Test Loss: " + str(epoch_loss))
    output_file.write("\n")
    output_file.write('Test Accuracy: {:.6f}\n'.format(correct))

    if num_classes == 2:
        show_conf_matrix(output_file,true_1, false_1, true_0, false_0,num_classes)
    elif num_classes == 3:
        show_conf_matrix(output_file,true_1, false_1_0, true_0, false_0_1,num_classes,
                         false_0_2,false_1_2,true_2, false_2_0, false_2_1)
    output_file.write("\n")
    print("\n")
    if num_classes == 2:
        true_1_ = float("{0:.2f}".format(100*true_1/(true_1+false_1)))
        false_1_ = float("{0:.2f}".format(100 * false_1 / (true_1 + false_1)))
        true_0_ = float("{0:.2f}".format(100 * true_0 / (true_0 + false_0)))
        false_0_ = float("{0:.2f}".format(100 * false_0 / (true_0 + false_0)))
        show_conf_matrix(output_file,true_1_, false_1_, true_0_, false_0_,num_classes)
    elif num_classes == 3:
        true_1_ = float("{0:.2f}".format(100 * true_1 / (true_1 + false_1_0+ false_1_2)))
        false_1_0_ = float("{0:.2f}".format(100 * false_1_0 / (true_1 + false_1_0+ false_1_2)))
        false_1_2_ = float("{0:.2f}".format(100 * false_1_2 / (true_1 + false_1_0 + false_1_2)))
        true_2_ = float("{0:.2f}".format(100 * true_2 / (true_2 + false_2_0+ false_2_1)))
        false_2_0_ = float("{0:.2f}".format(100 * false_2_0 / (true_2 + false_2_0+ false_2_1)))
        false_2_1_ = float("{0:.2f}".format(100 * false_2_1 / (true_2 + false_2_0 + false_2_1)))
        true_0_ = float("{0:.2f}".format(100 * true_0 / (true_0 + false_0_1+ false_0_2)))
        false_0_1_ = float("{0:.2f}".format(100 * false_0_1 / (true_0 + false_0_1+ false_0_2)))
        false_0_2_ = float("{0:.2f}".format(100 * false_0_2 / (true_0 + false_0_1 + false_0_2)))
        show_conf_matrix(output_file, true_1_, false_1_0_, true_0_, false_0_1_,num_classes,
                         false_0_2_,false_1_2_,true_2_, false_2_0_, false_2_1_,5)
    return correct,epoch_loss





def image_loader(image_name,transform):
    image = Image.open(image_name).convert('RGB')

    # print(image.size)
    image = transform(image)
    image = image.unsqueeze(0)
    return image




def single_test(img_path,model,device,new_size):
    model.eval()
    test_transform = transforms.Compose([transforms.Resize(new_size), transforms.CenterCrop(new_size),transforms.ToTensor()])
    inputs = image_loader(img_path,test_transform)
    inputs = inputs.to(device)

    outputs = model(inputs)

    _, pred = torch.max(outputs, 1)

    return pred

