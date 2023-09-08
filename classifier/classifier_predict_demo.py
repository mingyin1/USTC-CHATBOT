import logging
logging.basicConfig(level=logging.ERROR)
from network import NETWORK

if __name__ == '__main__':
    network = NETWORK("model/classifier229+9925.h5")
    while True:
        Input = input()
        print(network.predict(Input))