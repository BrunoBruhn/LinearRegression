import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics


def line(dataset_name, test_size):
	df = pd.read_csv(dataset_name) 

	#print(df['0'].values)
	X=df['0'].values.reshape(-1,1)
	y=df['0.1'].values.reshape(-1,1)

	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

	regressor = LinearRegression()  
	regressor.fit(X_train, y_train) #training the algorithm

	regression_line=[]

	for i in range(1000):
		regression_line.append(i*regressor.coef_[0][0]+regressor.intercept_[0])

	print('Regression line:     '+str(regressor.coef_[0][0])+" X + "+str(regressor.intercept_[0]))

	# a scatter plot comparing num_children and num_pets
	df.plot(kind='scatter',x='0',y='0.1',color='red')
	plt.plot(regression_line)
	plt.show()
	return regressor.coef_[0][0],regressor.intercept_[0]