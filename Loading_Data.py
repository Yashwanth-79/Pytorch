"""
PyTorch includes a package called torchvision which is used to load and prepare the dataset. 
It includes two basicfunctions namely Dataset and DataLoader which helps in transformation and loading of dataset.
"""
#Dataset is used to read and transform a datapoint from the given dataset.

trainset = torchvision.datasets.CIFAR10(root = './data', train = True,
   download = True, transform = transform)

#DataLoader is used to shuffle and batch data. 
It can be used to load the data in parallel with multiprocessing workers.

trainloader = torch.utils.data.DataLoader(trainset, batch_size = 4,
   shuffle = True, num_workers = 2)

