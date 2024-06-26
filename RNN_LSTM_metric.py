import numpy as np 
import tensorflow as tf 
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense,LSTM
from sklearn.model_selection import train_test_split


#the numbers and inputs to be received from the input results
poses_orb = np.random.rand(1000, 7)         #placeholder values for number of inputs now
poses_openvslam = np.random.rand(1000, 7)

train_traj = np.stack((poses_orb, poses_openvslam), axis=2)
target_traj = (poses_orb + poses_openvslam) / 2     #this target data for training will be GT

sequence_length = 10
num_sequences = train_traj.shape[0] - sequence_length
X = np.array([train_traj[i:i+sequence_length] for i in range(num_sequences)])
y = np.array([target_traj[i+sequence_length] for i in range(num_sequences)])

X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.2,random_state=42)

model=Sequential([LSTM(128,return_sequences=False, input_shape=(sequence_length,7*2)), Dense(7)])
model.compile(optimizer='adam', loss='mse')
model.summary()
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

#random generator for now. Later this ll be the new results from both the SLAMs
test_traj_orb = np.random.rand(sequence_length, 7)
test_traj_openvslam = np.random.rand(sequence_length, 7)
test_traj = np.stack((test_traj_orb, test_traj_openvslam), axis=1).reshape(1, sequence_length, 7 * 2)

predicted_traj = model.predict(test_traj)
print(predicted_traj)
