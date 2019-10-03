import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix,accuracy_score
from keras.layers import Dense, Dropout, LSTM, Activation

class PredictEngineFailure():
	def __init__(self, train_file, test_file, labels_file):
		self.features = ['id','cycle','setting1','setting2','setting3','s1','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11','s12','s13','s14','s15','s16','s17','s18','s19','s20','s21']
		self.window = 50
		self.lstm = None
		self.acc_scores = 0
		self.feature_cols = self.features[2:]  
		self.train_data, self.test_data = self.preprocess_data(train_file, test_file, labels_file)
		self.X_train, self.X_test, self.y_train, self.y_test = self.generate_train_test(self.train_data, self.test_data)

# Function to load data from train, test and label files. Construct appropriate dataframes format
# for passing through the LSTM model.
	def preprocess_data(self, train_file, test_file, labels_file):
		period = 30
		train_set = pd.read_csv(train_file,sep = ' ',header = None).drop([26,27], axis = 1)
		train_set.columns = self.features
		test_set = pd.read_csv(test_file,sep = ' ',header = None).drop([26,27], axis = 1)
		test_set.columns = self.features
		y_labels = pd.read_csv(labels_file,sep = ' ',header = None).drop([1], axis = 1)
		y_labels.columns = ['cycles_left']
		y_labels['id'] = y_labels.index + 1
		rul = pd.DataFrame(dataset_test.groupby('id')['cycle'].max()).reset_index()
		rul.columns = ['id', 'max']
		y_labels['rtf'] = y_labels['cycles_left'] + rul['max']
		y_labels.drop('cycles_left', axis = 1, inplace = True)
		test_set = test_set.merge(y_labels, on = ['id'], how = ['left'])
		test_set['ttf'] = test_set['rtf'] - stest_set['cycle']
		test_set.drop('rtf', axis = 1, inplace = True)
		train_set['ttf'] = train_set.groupby(['id'])['cycle'].transform(max) - train_set['cycle']
		train_cpy = train_set.copy()
		test_cpy = test_set.copy()
		train_cpy['prev_usage'] = train_cpy['ttf'].apply(lambda x: 1 if x <= period else 0)
		test_cpy['prev_usage'] = test_cpy['ttf'].apply(lambda x: 1 if x <= period else 0)
		return train_cpy, test_cpy

# Feature scaling utility funciton.
	def _feature_scaling(self, train, test):
		scaler = MinMaxScaler()
		train[self.feature_cols] = scaler.fit_transform(train[self.feature_cols])
		test[self.feature_cols] = scaler.transform(test[self.feature_cols])
		return train, test

# Reshape data frames created for train and labels so that it can be passed
# into the LSTM model.
	def df_gen(self, ip_data, window, label=None):
		df_zeros = pd.DataFrame(np.zeros((window - 1, ip_data.shape[1])), columns=ip_data.columns)
		ip_data = df_zeros.append(ip_data,ignore_index=True)
		data_array = ip_data[self.feature_cols].values
		num_elements = data_array.shape[0]
		temp_arr = []
		for start, stop in zip(range(0, num_elements-window), range(window, num_elements)):
			if label is not None:
				temp_arr.append(ip_data[label][stop])
			else:
				temp_arr.append(data_array[start:stop, :])
		return np.array(temp_arr)

# Generate final train and test data splits.
	def generate_train_test(self, train_set, test_set):
		train_data, test_data = self.feature_scaling(train_set, test_set)
		X_train = np.concatenate(list(list(self.df_gen(train_set[train_set['id'] == id], window)) for id in train_set['id'].unique()))
		y_train = np.concatenate(list(list(self.df_gen(train_set[train_set['id'] == id], window, label='prev_usage')) for id in train_set['id'].unique()))
		X_test = np.concatenate(list(list(self.df_gen(test_set[test_set['id'] == id], window)) for id in test_set['id'].unique()))
		y_test = np.concatenate(list(list(self.df_gen(test_set[test_set['id'] == id], window, label='prev_usage')) for id in test_set['id'].unique()))
		return X_train, X_test, y_train, y_test

# Build LSTM model
	def buildLSTM(self):

		model = Sequential()
		model.add(LSTM(input_shape = (self.window, self.X_train.shape[2]), units = 100, return_sequences = True))
		model.add(Dropout(0.2))
		model.add(LSTM(units = 50, return_sequences = False))
		model.add(Dropout(0.2))
		model.add(Dense(units=1, activation='sigmoid'))
		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		return model

# Run training and generate accuracy score.	
	def run_train(self):
		self.lstm = self.buidLSTM()
		self.lstm.fit(self.X_train, self.y_train, epochs = 10, batch_size = 200, validation_split = 0.05, verbose = 1, callbacks = [EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 0, verbose = 0, mode = 'auto')])
		self.acc_scores = self.lstm.evaluate(self.X_train, self.y_train, verbose = 1, batch_size = 200)

# Predict machine failure for single machine based on machine ID. Value returned is probabulity of 
# failure within 30 cycles.	
	def predict_failure(self, machine_id):
		machine_data = self.test_data[self.test_data.id == machine_id]
    	machine_test = self.gen_sequence(machine_data, self.window, self.feature_cols)
    	pred = self.lstm.predict(machine_test)
    	failure_prob = list(pred[-1]*100)[0]
    	return failure_prob


if __name__=='__main__':
	machine_id = 2
	pef = PredictEngineFailure('train.txt', 'test.txt', 'labels.txt')
	pef.run_train()
	print("Engine Failure Probability = {}".format(pef.predict_failure(machine_id)))
