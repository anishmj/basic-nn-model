# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Neural networks consist of simple input/output units called neurons (inspired by neurons of the human brain). These input/output units are interconnected and each connection has a weight associated with it. Neural networks are flexible and can be used for both classification and regression. In this article, we will see how neural networks can be applied to regression problems.

Regression helps in establishing a relationship between a dependent variable and one or more independent variables. Regression models work well only when the regression equation is a good fit for the data. Most regression models will not fit the data perfectly. Although neural networks are complex and computationally expensive, they are flexible and can dynamically pick the best type of regression, and if that is not enough, hidden layers can be added to improve prediction.

First import the libraries which we will going to use and Import the dataset and check the types of the columns and Now build your training and test set from the dataset Here we are making the neural network 2 hidden layer with activation layer as relu and with their nodes in them. Now we will fit our dataset and then predict the value.

## Neural Network Model

![rel](rel.png)

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM



~~~
### To Read CSV file from Google Drive :

from google.colab import auth
import gspread
from google.auth import default
import pandas as pd

## To train and test 
from sklearn.model_selection import train_test_split

## To scale 
from sklearn.preprocessing import MinMaxScaler

## To create a neural network model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

### Authenticate User:

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

### Open the Google Sheet and convert into DataFrame :

worksheet = gc.open('dlexp1').sheet1
rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns = rows[0])


df = df.astype({'Input':'float'})
df = df.astype({'Output':'float'})

df

X = df[['Input']].values
y = df[['Output']].values

X
y

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size = 0.4, random_state =35)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)
ai = Sequential([Dense(5 , activation = 'relu') ,Dense(10,activation = 'relu'), Dense(1)])
ai.compile(optimizer = 'rmsprop' , loss = 'mse')
ai.fit(X_train1 , y_train,epochs = 2200)

loss_df = pd.DataFrame(ai.history.history)
loss_df.plot()

X_test1 =Scaler.transform(X_test)
ai.evaluate(X_test1,y_test)
X_n1=[[4]]
X_n1_1=Scaler.transform(X_n1)
ai.predict(X_n1_1)

~~~

## Dataset Information

![sheet](sheet.png)

## OUTPUT

### Training Loss Vs Iteration Plot

![gra](gra.png)



### Test Data Root Mean Squared Error

![loss](loss.png)
![mse](mse.png)

### New Sample Data Prediction

![new](new.png)

## RESULT
Thus a neural network regression model for the given dataset is written and executed successfully.