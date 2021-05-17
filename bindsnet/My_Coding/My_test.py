from bindsnet.encoding import bernoulli_pre,bernoulli_RBF
for i in range(10):
    rate = bernoulli_pre(i*0.1,num_group=10)
    print(rate)
    print(bernoulli_RBF(datum=rate,neural_num=20,num_group=10,time=10))
