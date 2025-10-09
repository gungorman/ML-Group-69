import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
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
#X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=randomstate)

def pipeline_standard():
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def pipeline_minmax():
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


X_train_scaled, X_test_scaled = pipeline_standard()

'''KNN'''
def knn(neighbours):
    knn = KNeighborsRegressor(n_neighbors=neighbours)
    knn.fit(X_train_scaled, y_train)
    y_pred_knn = knn.predict(X_test_scaled)
    mae_knn = mean_absolute_error(y_test, y_pred_knn)
    return  mae_knn


'''SGD'''
def sgd(alpha_, learning_rate_, epochs_):
    sgd = SGDRegressor(
        loss='epsilon_insensitive',
        alpha=alpha_,
        learning_rate='constant',
        eta0=learning_rate_,
        warm_start=True,  
        max_iter=1,
        random_state=randomstate
    )
    mae_sgd_list = []
    for _ in range(epochs_):
        sgd.partial_fit(X_train_scaled, y_train)
        y_pred_sgd = sgd.predict(X_test_scaled)
        mae_sgd = mean_absolute_error(y_test, y_pred_sgd)
        mae_sgd_list.append(mae_sgd_list)
    return mae_sgd, mae_sgd_list


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
    #print(f'for {best_num_neighbours} neighbours, MAE of {best_mae}')
    #plot(knn_neighbours, knn_mae_list, 'KNN number of Neighbour Analysis', 'Number of Neighbours', 'MAE')
    return best_num_neighbours, best_mae
    

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

def manual_sgd_gridsearch():
    alpha_list = [0.0001, 0.001, 0.01, 0.1]
    lr_list = [0.001, 0.01, 0.1]
    epochs_list = [500, 1000, 1500, 2000]
    
    best_mae = float('inf')
    best_combo = None
    
    # This is manual multi-parameter grid search!
    for alpha in alpha_list:
        for lr in lr_list:
            for epochs in epochs_list:
                mae = sgd(alpha, lr, epochs)[0]
                if mae < best_mae:
                    best_mae = mae
                    best_combo = (alpha, lr, epochs)
    
    return best_combo, best_mae

def sgd_per_epoch():
    epoch_list = range(hp_epochs)
    mae_list = sgd(hp_alpha, hp_learning_rate, hp_epochs)[1]
    plot(epoch_list, mae_list, title="MAE over Epochs", xlabel="Epochs", ylabel="MAE")
best_combo, best_mae = manual_sgd_gridsearch()
best_num_neighbours, best_mae = knn_neighbours()

def baseline_model():
    dummy = DummyRegressor(strategy='median')
    dummy.fit(X_train_scaled, y_train)
    y_pred = dummy.predict(X_test_scaled)
    mae_dummy = mean_absolute_error(y_test, y_pred)
    return mae_dummy

'''Testing'''
print('-----KNN-----')
print(f'For {hp_nearest_neighbours} neighbours, the MAE was {knn(hp_nearest_neighbours)}')
print(f'The best number of neighbours was {best_num_neighbours} with a MAE of {best_mae}')
print('-----SGD-----')
print(f'For an alpha of {hp_alpha}, learning rate of {hp_learning_rate} and {hp_epochs} number of epochs, the MAE was {sgd(hp_alpha, hp_learning_rate, hp_epochs)[0]}')
#print(f'The best combination was alpha = {best_combo[0]}, learning rate = {best_combo[1]} and {best_combo[2]} number of epochs with an MAE of {best_mae}')
print('-----Dummy-----')
print(f"Baseline MAE:, {baseline_model()}")

sgd_per_epoch()

# Standardized Pipeline
'''
-----KNN-----
For 20 neighbours, the MAE was 0.28348586427235856
The best number of neighbours was 14 with a MAE of 0.27914830933977014
-----SGD-----
For an alpha of 0.1, learning rate of 0.001 and 1000 number of epochs, the MAE was 0.2869359981215274
The best combination was alpha = 0.0001, learning rate = 0.001 and 500 number of epochs with an MAE of 0.27914830933977014
'''
# Min-Max Normalized Pipeline
'''
-----KNN-----
For 20 neighbours, the MAE was 0.291327307369742
The best number of neighbours was 11 with a MAE of 0.2881526216782233
-----SGD-----
For an alpha of 0.1, learning rate of 0.001 and 1000 number of epochs, the MAE was 0.31809195933829926
The best combination was alpha = 0.001, learning rate = 0.001 and 500 number of epochs with an MAE of 0.2881526216782233
'''