import logging
logging.basicConfig(level=logging.ERROR)
from network import NETWORK
from untils import read_data

if __name__ == '__main__': 
	train_data = read_data()
	load_path = None
	for i in range(10):
		network = NETWORK(load_path=load_path)
		load_path = network.train(train_data)