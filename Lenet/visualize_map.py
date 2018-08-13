import matplotlib.pyplot as plt
import numpy as np
import Lenet
import torch
from torchvision import datasets, transforms

#%%
def histogram():
    model = Lenet.Lenet().cuda()
    model.load_state_dict(torch.load('./save/Lenet_model.dat'))
    model.eval()
    for name, param in model.state_dict().items():
        param = param.view(1,-1)
        print(name)
        plt.hist(param,60,facecolor='blue')
        plt.show()
    
#%%
def get_one_image(): 
    mean = [0.5071, 0.4867, 0.4408]
    stdv = [0.2675, 0.2565, 0.2761]
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])
    # Datasets
    train_set = datasets.CIFAR10('./dataset', train=True,  download=False)
    indices = torch.randperm(len(train_set))
    train_indices = indices[:1]
    image = train_set[train_indices][0]
    label = train_set[train_indices][1]
    plt.imshow(image)
    image = train_transforms(image)

    return image,label
#%%
def evaluate_one_image():  
    image,label= get_one_image() 
    image = image.unsqueeze(0)
    image = image.permute(0, 1, 2, 3).contiguous()
    model = Lenet.Lenet().cuda()
    model.load_state_dict(torch.load('./save/Lenet_model.dat'))
    model.eval()
    cifar=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','trunk']
    if torch.cuda.is_available():
        input_var = torch.autograd.Variable(image.cuda(async=True), volatile=True)
    conv1,conv2,conv3,output = model(input_var)
    _, pred = output.data.cpu().topk(1, dim=1)
    print('prediction:',cifar[pred]) 
    print('label:',cifar[label])       
#%%
def show():
    model = Lenet.Lenet().cuda()
    model.load_state_dict(torch.load('./save/Lenet_model.dat'))
    model.eval()
   
    image,label = get_one_image()
    image = image.unsqueeze(0)
    image = image.permute(0, 1, 2, 3).contiguous()
    #contiguous()  使得內存連續
    if torch.cuda.is_available():
        input_var = torch.autograd.Variable(image.cuda(async=True), volatile=True)
    conv1,conv2,conv3,output = model(input_var)
    conv1_feature = conv1.data.cpu().squeeze()
    plt.figure(figsize=(7, 7))
    for i in np.arange(0, 6):
        plt.subplot(2, 3, i + 1)
        plt.axis('off')
        plt.imshow(conv1_feature[i,:,:])
    plt.show()