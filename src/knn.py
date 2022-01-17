import pickle
import numpy as np
from PIL import Image
from matplotlib import pyplot
import math

#dataset file paths
BATCHES_META = "dataset\cifar-10-batches-py\\batches.meta"
DATA_BATCH_1 = "dataset\cifar-10-batches-py\data_batch_1"
DATA_BATCH_2 = "dataset\cifar-10-batches-py\data_batch_2"
DATA_BATCH_3 = "dataset\cifar-10-batches-py\data_batch_3"
DATA_BATCH_4 = "dataset\cifar-10-batches-py\data_batch_4"
DATA_BATCH_5 = "dataset\cifar-10-batches-py\data_batch_5"
TEST_BATCH = "dataset\cifar-10-batches-py\\test_batch"

#program constants
K1 = 5       #the KNN hyperparameter; represents how many 'nearest neighbours' are analyzed during the classification of an image


#subroutine to calculate K-Nearest Neighbours of a single image to be classified. Uses predefined hyperparameter 'K1'
def calculate_knn(image):
    pass


#subroutine to calculate the Euclidean distance of two RGB pixels.
#each pixel is a tuple of exactly 3 entries: the decimal (unit8) R, G, B values.
#   - pixel_p: the pixel being classified
#   - pixel_q: the neighbour pixel being compared against
def calculate_rgb_euclidean_dist(pixel_p, pixel_q):
    return math.sqrt( (pixel_q[0] - pixel_p[0])**2 + (pixel_q[1] - pixel_p[1])**2 + (pixel_q[2] - pixel_p[2])**2 )


def main():

    #load in CIFAR-10 labels list, and unpickle it into a list
    meta_dict = {}
    with open(BATCHES_META, 'rb') as data:
        meta_dict = pickle.load(data)

    #load in CIFAR-10 dataset, and unpickle it into a dict
    data_dict = {}
    with open(DATA_BATCH_1, 'rb') as data:
        data_dict = pickle.load(data, encoding='bytes')

    label_names = meta_dict['label_names']

    '''
    im = np.array(Image.open('uvic.jpg'))
    print(type(im))
    # <class 'numpy.ndarray'>

    print(im.dtype)
    # uint8

    print(im.shape)
    exit(0)
    '''

    #TESTING - get a single image into ndarray type (numpy)
    print(len(data_dict[b'data'][0][0:1024]))

    #grab first 3072 entries, representing RGB channels of one image
    im = np.array([data_dict[b'data'][0][0:]])

    #reshape the ndarray into 3 dimensions (for each colour channel) of 32x32
    #ndarray.shape.transpose ordering is: (Height/Row#, Width/Col#, Depth) --> (Y, X, Z)
    im2 = im.reshape([3, 32, 32]).transpose(1,2,0)
    
    print(type(im2))
    print(im2.dtype)
    print(im2.shape)
    
    print(im[0][1024:1056]) #first row (32 pixels) of the GREEN colour channel for the image
    print(im2[0][0][:])  #row 1 (top row), column 1 (first column), all dimensions (RGB)

    #index colour channel values like: [dimension][row (height)][column(s) (width)]
    #print(im2[1][0][0:10])
    

    #try plotting the image (note: it was transposed to (Y, X, Z))
    pyplot.imshow(im2[:, :, :])
    pyplot.show()

    exit(0)

    #iterate through each image (3072 colour entries)
    #for image_colour_codes in data_dict[b'data']:


    #print(meta_dict)
    print(data_dict.keys())
    print(data_dict[b'data'])
    print(len(data_dict[b'data'][1]))
    #print(data_dict[b'labels'])


if __name__ == "__main__":
    main()