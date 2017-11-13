#!/usr/bin/python3
# -*- coding: utf-8 -*-
import struct
import numpy as np
import matplotlib.pyplot as plt


class mnist_loader(object):
    """docstring for mnist_loader"""

    def __init__(self, images_path, labels_path):
        self.images_path = images_path
        self.labels_path = labels_path
        pass

    def load(self):
        """Load MNIST data from `path`"""
        with open(self.labels_path, 'rb') as lbpath:
            magic, n = struct.unpack('>II',
                                     lbpath.read(8))
            labels = np.fromfile(lbpath,
                                 dtype=np.uint8)

        with open(self.images_path, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack('>IIII',
                                                   imgpath.read(16))
            images = np.fromfile(imgpath,
                                 dtype=np.uint8).reshape(len(labels), 784)

        return images, labels

    def show(self, rows, cols, samples=[]):
        fig, ax = plt.subplots(nrows=rows, ncols=cols, sharex=True, sharey=True)
        ax = ax.flatten()
        for i in range(rows * cols):
            # img = imgs[i].reshape(28, 28)
            img = np.array(samples[i], dtype=np.uint8).reshape(28, 28)
            ax[i].imshow(img, cmap='Greys', interpolation='nearest')

        ax[0].set_xticks([])
        ax[0].set_yticks([])
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':

    loader = mnist_loader('../dataset/mnist/t10k-images-idx3-ubyte',
                          '../dataset/mnist/t10k-labels-idx1-ubyte')
    imgs, labels = loader.load()
    loader.show(2, 5, samples=imgs[:10])
