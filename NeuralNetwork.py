globals().clear()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import path
import math

# first we will create 10,000 random dots inside a 1 by 1 square. dots that fall inside the star of david
# will be classified as 1, dots outside as 0.
print('creating random dots for training and test')
dots=10000
data=np.random.rand(dots,2) #create 10,000 random dots with x and y coordinates
def inpolygon(point,xv,yv): #returns true if points in data are inside the polygon defined by xv yv
    p = path.Path([(xv[i],yv[i]) for i in range(len(xv))])
    return p.contains_point(point)

#create coordinates for 2 triangles to make up the star of david
xv1 = np.array([0.5,0,1,0.5])
yv1 = np.array([1,0.25,0.25,1])
xv2 = np.array([0.5,0,1,0.5])
yv2 = np.array([0,0.75,0.75,0])
ind1=np.zeros(dots)
ind2=np.zeros(dots)
data=np.c_[data,np.zeros(dots)]
for i in range(dots):
    if inpolygon(data[i][0:2:1],xv1,yv1): #[i] is row i, [0:2:1] from 0 to 2 not including 2
        ind1[i]=1
    if inpolygon(data[i][0:2:1],xv2,yv2):
        ind2[i]=1
    if ind1[i]==1 or ind2[i]==1:
        data[i][2]=1
plt.plot(data[data[:,2]==1,0], data[data[:,2]==1,1], 'bo') #dots that are inside star
plt.plot(data[data[:,2]==0,0], data[data[:,2]==0,1], 'ro') #dots that are outside star
plt.title('Data')
plt.show()

# separate test and train data
train_data=np.array(data[0:int(dots*0.9)]) #90%
test_data=np.array(data[int(dots*0.9):dots]) #10%

#parameters for NN
input_layer_size=2
hidden_layer_size=14 #all hidden layers are the same size

#set up weights using xavier's initialization method
w1=np.zeros((input_layer_size, hidden_layer_size)) #weights between input --> hidden layer
variance1= 2/(input_layer_size + hidden_layer_size)
stddev1= math.sqrt(variance1)
w2=np.zeros((hidden_layer_size,hidden_layer_size)) #weights between hidden layer 1 --> hidden layer 2
variance2= 2/(hidden_layer_size+hidden_layer_size)
stddev2= math.sqrt(variance2)
w3=np.zeros((hidden_layer_size,hidden_layer_size)) #weights between hidden layer 2 --> hidden layer 3
variance3= 2/(hidden_layer_size+hidden_layer_size)
stddev3= math.sqrt(variance3)
w4=np.zeros((hidden_layer_size,2)) #weights between hidden layer 3 --> output
variance4= 2/(hidden_layer_size + 2)
stddev4= math.sqrt(variance4)
#drawing random values from a normal distribution with an std according to the number of neurons, and inserting to weights
for i in range(hidden_layer_size):
    for j in range(input_layer_size):
        w1[j,i]= np.random.normal(0,stddev1)
        w4[i,j]= np.random.normal(0,stddev4)
for i in range(hidden_layer_size):
    for j in range(hidden_layer_size):
        w2[i,j] = np.random.normal(0, stddev2)
        w3[i,j] = np.random.normal(0, stddev3)

#start with a bias of 0
b1=np.zeros((1, hidden_layer_size))
b2=np.zeros((1, hidden_layer_size))
b3=np.zeros((1, hidden_layer_size))
b4=np.zeros((1, 2))

#for momemntum of deltaW(n-1)
alpha=0.03
dw1_n1=0
db1_n1=0
dw2_n1=0
db2_n1=0
dw3_n1=0
db3_n1=0
dw4_n1=0
db4_n1=0

#parameters for learning
learning_rate=0.15
iter_size=6000 #number of iterations
error_vec=math.nan #this shows the progress of training

print("initializing training with a learning rate of " +str(learning_rate)+ "...")
#train the net with cross entropy
for iteration in range(iter_size):
    #feed-forward
    shuffled_train=train_data[np.random.permutation(len(train_data)),:] #randomize data order in each iteration
    X = shuffled_train[:,0:2] #input
    Y = shuffled_train[:,2] #teacher
    HL1 = X@w1+b1 #hidden layer 1 before activation
    HL1 = np.tanh(HL1) #tanh activation function
    HL2 = HL1@w2+b2
    HL2 = np.tanh(HL2)
    HL3 = HL2@w3+b3
    HL3 = np.tanh(HL3)
    FL = HL3@w4+b4
    FL = np.exp(FL)/(np.sum(np.exp(FL),axis=1)[:,None]) #softmax

    #cost calculation
    data_error=0
    for i in range(len(X)):
        data_error = data_error + -1*math.log(FL[i, int(Y[i])])
    data_error=data_error/len(X)
    error_vec=np.append(error_vec,data_error)

    #back-propagation
    #final-hl3
    dFL = np.copy(FL)
    for i in range(len(X)):
        dFL[i,int(Y[i])] = dFL[i,int(Y[i])]-1
    dFL = dFL/len(X)
    dError_dw4 = HL3.conj().T @ dFL
    dError_db4 = np.ones((1,len(X))) @ dFL
    #back-propagation through hidden layer 3
    dHL3 = dFL @ w4.conj().T
    dHL3 = dHL3 * (1/(np.cosh(HL2 @ w3 +b3))**2)
    dError_dw3 = HL2.conj().T @ dHL3
    dError_db3 = np.ones((1,len(X))) @ dHL3
    # back-propagation through hidden layer 2
    dHL2 = dHL3 @ w3.conj().T
    dHL2 = dHL2 * (1 / (np.cosh(HL1 @ w2 + b2)) ** 2)
    dError_dw2 = HL1.conj().T @ dHL2
    dError_db2 = np.ones((1, len(X))) @ dHL2
    # back-propagation through hidden layer 1
    dHL1 = dHL2 @ w2.conj().T
    dHL1 = dHL1 * (1 / (np.cosh(X @ w1 + b1)) ** 2)
    dError_dw1 = X.conj().T @ dHL1
    dError_db1 = np.ones((1, len(X))) @ dHL1

    #update weights
    w4 = w4 - (dError_dw4 * learning_rate + (alpha * dw4_n1))
    b4 = b4 - (dError_db4 * learning_rate + (alpha * db4_n1))
    w3 = w3 - (dError_dw3 * learning_rate + (alpha * dw3_n1))
    b3 = b3 - (dError_db3 * learning_rate + (alpha * db3_n1))
    w2 = w2 - (dError_dw2 * learning_rate + (alpha * dw2_n1))
    b2 = b2 - (dError_db2 * learning_rate + (alpha * db2_n1))
    w1 = w1 - (dError_dw1 * learning_rate + (alpha * dw1_n1))
    b1 = b1 - (dError_db1 * learning_rate + (alpha * db1_n1))

    #update momentum for next step
    dw1_n1 = np.copy(dError_dw1)
    db1_n1 = np.copy(dError_db1)
    dw2_n2 = np.copy(dError_dw2)
    db2_n2 = np.copy(dError_db2)
    dw3_n3 = np.copy(dError_dw3)
    db3_n3 = np.copy(dError_db3)
    dw4_n4 = np.copy(dError_dw4)
    db4_n4 = np.copy(dError_db4)

    #decaying learning rate
    if error_vec[-1] <0.2 and error_vec[-1] >0.1:
        if learning_rate!=0.06:
            learning_rate = 0.06
            print("training: decreased learning rate to " + str(learning_rate) + "...")
    elif error_vec[-1] <0.1 and error_vec[-1] >0.07:
        if learning_rate != 0.03:
            learning_rate = 0.03
            print("training: decreased learning rate to " + str(learning_rate) + "...")
    elif error_vec[-1] < 0.07:
        if learning_rate != 0.01:
            learning_rate = 0.01
            print("training: decreased learning rate to " + str(learning_rate) + "...")

    if iteration == 0:
        print("training: starting with an error rate of " +str(round(error_vec[iteration+1]*100,2)) + "%...")
    if iteration == 999:
        print("training: after 1000 iterations, an error rate of " + str(round(error_vec[iteration + 1] * 100,2)) + "%...")
    if iteration == 1999:
        print("training: after 2000 iterations, an error rate of " + str(round(error_vec[iteration + 1] * 100,2)) + "%...")
    if iteration == 2999:
        print("training: after 3000 iterations, an error rate of " + str(round(error_vec[iteration + 1] * 100,2)) + "%...")
    if iteration == 3999:
        print("training: after 4000 iterations, an error rate of " + str(round(error_vec[iteration + 1] * 100,2)) + "%...")
    if iteration == 4999:
        print("training: after 5000 iterations, an error rate of " + str(round(error_vec[iteration + 1] * 100,2)) + "%...")

#evaluate training efficiency
idx=np.argmax(FL, axis=1)  #find indices where the max prob for each class occurs
num_correct_classifications = sum(idx == Y)
acc = num_correct_classifications/len(X)
print("Training finished with " + str(round(acc*100,2)) + "% correct classifications")
plt.plot(np.arange(1,iter_size+1),error_vec[1:])
plt.xlabel('iteration')
plt.ylabel('error[%]')
plt.title('Learning Process')
plt.show()

print("Initializing the neural network test...")
#test the NN
# feed-forward
X = test_data[:, 0:2]  # input
Y = test_data[:, 2]  # teacher
HL1 = X @ w1 + b1  # hidden layer 1 before activation
HL1 = np.tanh(HL1)
HL2 = HL1 @ w2 + b2
HL2 = np.tanh(HL2)
HL3 = HL2 @ w3 + b3
HL3 = np.tanh(HL3)
FL = HL3 @ w4 + b4
FL = np.exp(FL) / (np.sum(np.exp(FL), axis=1)[:, None])
idx=np.argmax(FL, axis=1)  #find indices where the max prob for each class occurs
num_correct_classifications = sum(idx == Y)
acc = num_correct_classifications/len(X)
print("Test had " + str(round(acc*100,2)) + "% correct classifications")

#plot the classifications
plt.plot(test_data[test_data[:,2]==1,0], test_data[test_data[:,2]==1,1], 'bo') #dots that are inside star
plt.plot(test_data[test_data[:,2]==0,0], test_data[test_data[:,2]==0,1], 'go') #dots that are outside star
plt.plot(test_data[test_data[:,2]!=idx,0], test_data[test_data[:,2]!=idx,1], 'r*') #dots that are outside star
plt.title('Data')
plt.show()
