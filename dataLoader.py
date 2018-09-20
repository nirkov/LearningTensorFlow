from mlxtend.data import loadlocal_mnist

def load_MNIST(data_address):
     train_images, train_labels = loadlocal_mnist(
               images_path=data_address+'/train-images.idx3-ubyte',
               labels_path=data_address+'/MNIST dataset/train-labels.idx1-ubyte')

     test_images, test_labels = loadlocal_mnist(
               images_path=data_address+'/t10k-images.idx3-ubyte',
               labels_path=data_address+'/t10k-labels.idx1-ubyte')

     return  train_images, train_labels,  test_images, test_labels
