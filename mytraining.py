import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle


def data_split(data, ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data) * ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[test_indices], data.iloc[train_indices]

if __name__=='__main__':
    df = pd.read_csv('data.csv')
    train, test = data_split(df, 0.2)
    
    X_train = train[['bodyPain', 'fever', 'age', 'runnyNose', 'diffBreath']].to_numpy()
    X_test = test[['bodyPain', 'fever', 'age', 'runnyNose', 'diffBreath']].to_numpy()
    
    Y_train = train[['infectionProb']].to_numpy().reshape(1200,)
    Y_test = test[['infectionProb']].to_numpy().reshape(4803,)
    
    clf = LogisticRegression() 
    clf.fit(X_train, Y_train)
    
    # open a file, where you ant to store the data
    file = open('model.pkl', 'wb')

    # dump information to that file
    pickle.dump(clf, file)

    # Code for inference
    inputFeatures = [90, -1, 22, -1, -1]
    infProb = clf.predict_proba([inputFeatures])[0][1]
    file.close()