import tensorflow as tf

sess = tf.Session()

W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
linear_model = W*x + b

# assign loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares

# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

####### training data
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]
#######

#training loop 
init = tf.initialize_all_variables()
sess.run(init) # reset values to incorrect defaults
for i in range(1000):
    sess.run(train, {x:x_train, y:y_train})
    if i % 100==0 :
	print(sess.run([W, b, loss], {x:x_train,y: y_train}))
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x:x_train,y: y_train})

print("W: %s b: %s loss: %s" %(curr_W, curr_b, curr_loss))    
