import numpy
import pandas
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras.metrics import fmeasure

# Fix random seed for reproducibility.
seed = 7
numpy.random.seed(seed)

# Load data set.
train_dataframe = pandas.read_csv('pm10_train.csv', header=None)
train_dataset = train_dataframe.values
train_X = train_dataset[:,0:7].astype(float)
train_Y = train_dataset[:,7]

test_dataframe = pandas.read_csv('pm10_test.csv', header=None)
test_dataset = test_dataframe.values
test_X = test_dataset[:,0:7].astype(float)
test_Y = test_dataset[:,7]

# Encode class values as integers.
encoder = LabelEncoder()
encoder.fit(train_Y)
encoded_train_y = encoder.transform(train_Y)

encoder.fit(test_Y)
encoded_test_y = encoder.transform(test_Y)

# Convert integers to dummy variables.(i.e. one-hot encode.)
dummy_train_y = np_utils.to_categorical(encoded_train_y)
dummy_test_y = np_utils.to_categorical(encoded_test_y)

class model_parameter:
	def __init__(self, il_activation, hl_activation, init, dropout_pram):
		self.il_activation = il_activation
		self.hl_activation = hl_activation
		self.init = init
		self.dropout_pram = dropout_pram

	def get_ilActivation(self):
		return self.il_activation

	def get_hlActivation(self):
		return self.hl_activation

	def get_init(self):
		return self.init

	def get_doPram(self):
		return self.dropout_pram

parameters = []
parameters.append(model_parameter(il_activation='relu', hl_activation='relu', init='he_uniform', dropout_pram=None))

parameter = None

scoreList = []

# Define baseline model.
def pm10_model():
	global parameter

	print 'Current parameters : %s, %s, %s, %s' % (parameter.get_ilActivation(), 
												parameter.get_hlActivation(), 
												parameter.get_init(), 
												parameter.get_doPram())
	hidden_layer = model.add(Dense(12, activation=parameter.get_hlActivation()))
	# Create Model.
	model = Sequential()
	# Input Layer
	model.add(Dense(21, input_dim=7, init=parameter.get_init(), activation=parameter.get_ilActivation()))
	# Hidden Layer(1)
	# model.add(Dense(12, activation=parameter.get_hlActivation()))
	
	hidden_layer
	# Output Layer
	model.add(Dense(4, activation='softmax'))

	# Compile Model.
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', fmeasure])
	return model

print 'Number of train data : %d' % len(train_X)
print 'Number of test data : %d' % len(test_X)

for sequence, parameter in enumerate(parameters):
	global param
	print '=========================%d=========================' % sequence
	param = parameter

	estimator = KerasClassifier(build_fn=pm10_model, nb_epoch=650, batch_size=7, verbose=0)
	print 'Training...'
	estimator.fit(train_X, dummy_train_y)

	print 'Accuracy Testing...'
	evaluate_score = cross_val_score(estimator, test_X, dummy_test_y)

	print '\nAccuracy : %.4f%%' % (evaluate_score[1].mean()*100)
	print 'F1-Measure : %.2f' % (evaluate_score[2].mean())
	print 'Loss : %.2f' % (evaluate_score[0].mean())

	scoreDict = {}
	scoreDict['Index'] = sequence
	scoreDict['Loss'] = '%.2f' % (evaluate_score[0].mean())
	scoreDict['F1-Measure'] = '%.2f' % (evaluate_score[2].mean())
	scoreDict['Accuracy'] = '%.4f%%' % (evaluate_score[1].mean()*100)

	scoreList.append(scoreDict)

for result in scoreList:
	print result