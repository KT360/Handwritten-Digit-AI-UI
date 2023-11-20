from BabyNetwork import BabyNetwork


#last layer should be 10
Network = BabyNetwork(15000,[784,30,10])

Network.Evaluate(30, 10)