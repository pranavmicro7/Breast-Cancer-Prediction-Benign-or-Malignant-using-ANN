#importing necessary libraries
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix

#importing data
data=pd.read_csv("data.csv")
data=data.iloc[:,1:32]
x=data.iloc[:,1:].values
y=data.iloc[:,0].values
y=y.reshape(569,1)

#cleaning data
ll=LabelEncoder()
y[:,0]=ll.fit_transform(y[:,0])
y=y.astype(float)
ll=StandardScaler()
xt=ll.fit_transform(x)

#diving data to train and test set
x_train,x_test,y_train,y_test=train_test_split(xt,y,test_size=.2,random_state=0)

#define learning rate and epoch
al=.03
epoch=2500

#funcion defining Neural Network
def nn(epoch,al,x_test):
    tf.reset_default_graph()
    
    #Placeholders
    x=tf.placeholder(dtype=tf.float64,shape=(455,30))
    y=tf.placeholder(dtype=tf.float64,shape=(455,1))
    
    #Variables
    tf.set_random_seed(0)
    w1=tf.Variable(tf.random_normal(shape=(30,20),dtype=tf.float64))
    w2=tf.Variable(tf.random_normal(shape=(20,15),dtype=tf.float64))
    w3=tf.Variable(tf.random_normal(shape=(15,6),dtype=tf.float64))
    w4=tf.Variable(tf.ones(shape=(6,1),dtype=tf.float64))
    
    #Activation function in the layers
    a1=tf.sigmoid(tf.matmul(x,w1))
    a2=tf.sigmoid(tf.matmul(a1,w2))
    a3=tf.sigmoid(tf.matmul(a2,w3))
    y_est=tf.sigmoid(tf.matmul(a3,w4))
    
    #Defining lost
    delta=tf.square(y_est-y)
    loss=tf.reduce_mean(delta)
    
    # Define a train operation to minimize the loss
    optimizer=tf.train.AdamOptimizer(learning_rate=al)
    train=optimizer.minimize(loss)

    #Initialize variables and run session
    init=tf.global_variables_initializer()
    sess=tf.Session()
    sess.run(init)
    
    #Add summary for histogram
    tf.summary.histogram("x",x)
    tf.summary.histogram("y",y)
    tf.summary.histogram("w1",w1)
    tf.summary.histogram("w2",w2)
    tf.summary.histogram("w3",w3)
    tf.summary.histogram("w4",w4)
    tf.summary.histogram("a1",a1)
    tf.summary.histogram("a2",a2)
    tf.summary.histogram("a3",a3)
    tf.summary.histogram("y_est",y_est)
    
    #Merge to a single operator
    merged_summary_op = tf.summary.merge_all()
    
    #Set the logs writer to the folder "/home/pranav/ten/logs"
    summary_writer = tf.summary.FileWriter('/home/pranav/ten/logs', graph_def=sess.graph_def)
    
    
    # Go through num_iter iterations
    for i in range (epoch):
        
        sess.run(train,feed_dict={x:x_train,y:y_train})
        #printing loss
        print(sess.run(loss,feed_dict={x:x_train,y:y_train}))
        ww1=sess.run(w1)
        ww2=sess.run(w2)
        ww3=sess.run(w3)
        ww4=sess.run(w4)
        #Write log for each iteration
        summary_str = sess.run(merged_summary_op, feed_dict={x: x_train, y: y_train})
        summary_writer.add_summary(summary_str,i)
    
    #Calculating prediction on given x_test data
    a11=tf.sigmoid(tf.matmul(x_test,ww1))
    a22=tf.sigmoid(tf.matmul(a11,ww2))
    a33=tf.sigmoid(tf.matmul(a22,ww3))
    ypred=tf.sigmoid(tf.matmul(a33,ww4))
    ypred=sess.run(ypred)    
    sess.close()    
    
    return ww1,ww2,ww3,ww4,ypred

#Calling function nn
w1,w2,w3,w4,ypred=nn(epoch,al,x_test)  

for i in range (114):
    if ypred[i][0]>.51:
        ypred[i][0]=1
    else:
        ypred[i][0]=0
#Checking Accuracy Over Test Set
accuracy_score(y_test,ypred)  
    
#confusion matrix
confusion_matrix(y_test,ypred)

""" 
To run TensorBoard, use the following command:-

tensorboard --logdir=/home/pranav/ten/logs

"""
