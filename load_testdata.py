import os
from scipy import ndimage
import pickle


def load():
    data = []
    # Load Test sets
    data_path = "mnist_png/testing"
    test_type = os.listdir(data_path)

    # Training by directory type(0~9)
    index = 0
    for each_type in test_type:
        type_path = data_path + "/" + each_type
        test_set = os.listdir(type_path)
        for each_test_set in test_set:
            index = index + 1
            img_path = type_path + "/" + each_test_set
            img = ndimage.imread(img_path).astype(float)
            img = img / 255
            each_type = int(each_type)
            tmp = (img, each_type)
            data.append(tmp)
        print("Complete test training of ", each_type)

    filename = open("mnist_test.dat", "wb")
    pickle.dump(data, filename)
    filename.close()
    print("Done saving", index, "data sets into file: mnist_test.dat")

if __name__ == "__main__":
    load()


