import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
randomstate = 42

'''Hyperparameters'''
hp_nearest_neighbours = 20
hp_alpha = 0.1
hp_learning_rate = 0.001
hp_epochs = 1000
randomstate = 42

'''Pre-Processing'''
data = pd.read_csv('football_wages.csv')
cleaned_data = data.drop("nationality_name", axis=1)
X = cleaned_data.drop("log_wages", axis=1).to_numpy()
y = cleaned_data["log_wages"].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=randomstate)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

'''KNN'''
def knn(neighbours):
    knn = KNeighborsRegressor(n_neighbors=neighbours)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    mae_knn = mean_absolute_error(y_test, y_pred_knn)
    return  mae_knn


'''SGD'''
def sgd(alpha_, learning_rate_, epochs_):
    sgd = SGDRegressor(
        loss='epsilon_insensitive',
        alpha=alpha_,      # Regularization
        learning_rate='constant',
        eta0=learning_rate_,         # Learning rate  
        max_iter=epochs_,      # Number of epochs
        random_state=randomstate
    )
    sgd.fit(X_train_scaled, y_train)
    y_pred_sgd = sgd.predict(X_test_scaled)
    mae_sgd = mean_absolute_error(y_test, y_pred_sgd)
    return mae_sgd

'''Plotter'''
def plot(x, y, title="Scatter", xlabel="X", ylabel="Y"):
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()

'''Hyperparameter Analysis'''
def knn_neighbours():
    knn_mae_list = []
    knn_neighbours =[]
    for i in range(1,100):
        knn_neighbours.append(i)
        knn_mae_list.append(knn(i))
    best_mae = min(knn_mae_list)
    index = knn_mae_list.index(best_mae)
    best_num_neighbours = knn_neighbours[index]
    print(f'for {best_num_neighbours} neighbours, MAE of {best_mae}')
    plot(knn_neighbours, knn_mae_list, 'KNN number of Neighbour Analysis', 'Number of Neighbours', 'MAE')
    

def best_alpha(learning_rate_, epochs_):
    alpha_list = np.arange(0, 0.2, 0.01)
    alpha_mae_list = []
    for i in range(len(alpha_list)):
        alpha_mae_list.append(sgd(alpha_list[i],learning_rate_,epochs_))
    best_mae = min(alpha_mae_list)
    index = alpha_mae_list.index(best_mae)
    best_alpha = alpha_list[index]
    print(f'for {epochs_} epochs and learning rate: {learning_rate_}, best alpha is {best_alpha} with MAE of {best_mae}')
    #plot(alpha_list, alpha_mae_list, 'Best Alpha Analysis', 'Alpha', 'MAE')
    
def best_lr(alpha_, epochs_):
    lr_list = np.arange(0.000001, 0.1, 0.001)
    lr_mae_list = []
    for i in range(len(lr_list)):
        lr_mae_list.append(sgd(alpha_, lr_list[i], epochs_))
    best_mae = min(lr_mae_list)
    index = lr_mae_list.index(best_mae)
    best_lr = lr_list[index]
    print(f'for {epochs_} epochs and alpha: {alpha_}, best learning rate is {best_lr} with MAE of {best_mae}')
    #plot(lr_list, lr_mae_list, 'Best Learning Rate Analysis', 'Learning Rate', 'MAE')


knn_neighbours()