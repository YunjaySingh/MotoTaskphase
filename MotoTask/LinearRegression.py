import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_df = pd.read_csv("/Users/mrityunjayasinghjathrana/Downloads/train.csv")
test_df = pd.read_csv("/Users/mrityunjayasinghjathrana/Downloads/test.csv")

x_train = train_df.drop(columns=['SalePrice']).values
y_train = train_df['SalePrice'].values

x_train = (x_train - np.mean(x_train, axis = 0))/np.std(x_train , axis = 0)

x_test = test_df.drop(columns=['SalePrice']).values
x_test = (x_test - np.mean(x_test , axis = 0))/np.std(x_test, axis = 0)
y_test = test_df['SalePrice'].values

class LinearRegression:
    def __init__(self , learning_rate=0.03, epochs = 9000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None 
        self.bias = None 
        
    def fit(self, x , y):
        n_samples , n_features = x.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        loss_history = []
        
        for epochs in range(self.epochs):
            y_predicted = np.dot(x , self.weights) + self.bias
            
            loss = (1/n_samples) * np.sum((y_predicted - y)**2)
            loss_history.append(loss)
            
            dw = (2/n_samples) * np.dot(x.T, (y_predicted - y))
            db = (2/n_samples) * np.sum(y_predicted - y)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
        return loss_history 
    
    def predict(self , x) :
        return np.dot(x, self.weights) + self.bias 
    
model = LinearRegression(learning_rate=0.03 , epochs = 9000)
loss_history = model.fit(x_train , y_train)

y_pred = model.predict(x_test)

plt.figure(figsize=(10, 6))
plt.plot(range(len(loss_history)), loss_history, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (Mean Squared Error)')
plt.title('Loss vs. Epochs')
plt.legend()
plt.show()

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

mse_custom = mean_squared_error(y_test, y_pred)

print(f"Mean Squared Error (Custom Model): {mse_custom}")
