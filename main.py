
# from AutoEncoder import *

from AE_Models.mnist import *
from AE_Models.fashion_mnist import *
from AE_Models.cifar import *
from AE_Models.slr import *

def main():
    # Minst
    print('MNIST')
    net = VAE_MNIST()
    net.init_model()
    epoch = 25
    for i in range(epoch):
        net.fit_train(i)
        net.test(i)

    # VAE_fashion_mnist
    # print('Fashion MNIST')
    # net = VAE_fashion_mnist()
    # net.init_model()
    # epoch = 10
    # for i in range(epoch):
    #     net.fit_train(i)
    #     net.test(i)
    # # 
    # print('CIFAR')

    # net = VAE_cifar()
    # net.init_model()
    # epoch = 10
    # for i in range(epoch):
    #     net.fit_train(i)
    #     net.test(i)

    print('STL')
    net = VAE_stl()
    net.init_model()
    epoch = 5
    for i in range(epoch):
        net.fit_train(i)
        net.test(i)
 
if __name__ == "__main__":
    main()