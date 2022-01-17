import tarfile
  
#open file
file = tarfile.open('dataset/cifar-10-python.tar.gz')
  
#extract file
file.extractall('dataset')
file.close()