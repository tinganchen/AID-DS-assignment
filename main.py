import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

def main():
    if not os.path.isdir('output'):
        os.makedirs('output')
    
    ## 1. Load data
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    input_test = test_data
    
    ## 2. Remove the columns - 'id', 'CustomerId', 'Surname'
    train_data = train_data.iloc[:, 3:]
    test_data = test_data.iloc[:, 3:]
    #train_data.columns ## 14
    #test_data.columns ## 13
    
    ## 3. Turn the categorical variables ('Geography' & 'Gender') into one-hot 
    # Train
    Geography_dummies = pd.get_dummies(train_data['Geography'])
    train_data = pd.concat([train_data, Geography_dummies], axis=1)
    
    Gender_dummies = pd.get_dummies(train_data['Gender'])
    train_data = pd.concat([train_data, Gender_dummies], axis=1)
    
    # Test
    Geography_dummies = pd.get_dummies(test_data['Geography'])
    test_data = pd.concat([test_data, Geography_dummies], axis=1)
    
    Gender_dummies = pd.get_dummies(test_data['Gender'])
    test_data = pd.concat([test_data, Gender_dummies], axis=1)

    sub_train_data = train_data.drop(['Geography', 'Gender'], axis=1)
    test_data = test_data.drop(['Geography', 'Gender'], axis=1)

    ## 4. Split a validation set from train data
    x_train, x_val, y_train, y_val = train_test_split(sub_train_data, 
                                                      sub_train_data['Exited'],
                                                      test_size = 0.01, random_state = 10)
    
    x_train = x_train.drop(['Exited'], axis=1)
    x_val = x_val.drop(['Exited'], axis=1)
    
    '''
    ## 5-1. Train and Predict (Logistic Regression)
    LogisticRegressionModel = linear_model.LogisticRegression()
    LogisticRegressionModel.fit(x_train, y_train)
    predicted = LogisticRegressionModel.predict(x_val)
    val_acc = LogisticRegressionModel.score(x_val, y_val) # 77.6% 
    val_f1 = f1_score(y_val, predicted) # 17.4%
    
    test_pred = LogisticRegressionModel.predict(test_data)
    df = pd.concat([input_test, pd.Series(test_pred, name = "Exited")], axis=1)
    df.to_csv('output/logistic_output.csv', index = False)
    
    # confusion matrix
    cm = confusion_matrix(y_val, predicted, labels = LogisticRegressionModel.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix = cm,
                                  display_labels = LogisticRegressionModel.classes_)
    disp.plot()
    plt.savefig('output/logistic_cm.pdf')
    plt.show()
    
    ## Plot - pairwise
    map_dict = {0: 'Not Exited', 1: 'Exited'}
    exited_cat = pd.Series(predicted).map(map_dict)
    x_val = x_val.reset_index(drop = True)
    sns.pairplot(pd.concat([x_val.iloc[:, :6], exited_cat], axis=1), hue = 0)
    plt.savefig('output/logistic_pairwise.pdf')
    plt.show()
    
    sns.pairplot(pd.concat([x_val.iloc[:, 6:], exited_cat], axis=1), hue = 0)
    plt.savefig('output/logistic_pairwise2.pdf')
    plt.show()
    '''
    
    ## 5-2. Train and Predict (XGboost)
    xgboostModel = XGBClassifier(n_estimators = 10, learning_rate = 0.3)
    xgboostModel.fit(x_train, y_train)
    predicted = xgboostModel.predict(x_val)
    val_acc = xgboostModel.score(x_val, y_val) # 86.1%
    val_f1 = f1_score(y_val, predicted) # 62.0%
    
    test_pred = xgboostModel.predict(test_data)
    df = pd.concat([input_test, pd.Series(test_pred, name = "Exited")], axis=1)
    df.to_csv('output/xgboost_output.csv', index = False)
    
    # confusion matrix
    cm = confusion_matrix(y_val, predicted, labels = xgboostModel.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix = cm,
                                  display_labels = xgboostModel.classes_)
    disp.plot()
    plt.savefig('output/xgboost_cm.pdf')
    plt.show()
    
    ## Plot - pairwise
    map_dict = {0: 'Not Exited', 1: 'Exited'}
    exited_cat = pd.Series(predicted).map(map_dict)
    x_val = x_val.reset_index(drop = True)
    sns.pairplot(pd.concat([x_val.iloc[:, :6], exited_cat], axis=1), hue = 0)
    plt.savefig('output/xgboost_pairwise.pdf')
    plt.show()
   
    sns.pairplot(pd.concat([x_val.iloc[:, 6:], exited_cat], axis=1), hue = 0)
    plt.savefig('output/xgboost_pairwise2.pdf')
    plt.show()
    
    '''
    ## PCA
    n_components = 4  # Choose the number of dimensions to reduce to
    pca = PCA(n_components=n_components)
    
    # Fit PCA on the training data and transform both training and validation sets
    x_train_reduced = pca.fit_transform(x_train)
    x_val_reduced = pca.transform(x_val)
    
    xgboostModel = XGBClassifier(n_estimators = 100, learning_rate = 0.3)
    xgboostModel.fit(x_train_reduced, y_train)
    predicted = xgboostModel.predict(x_val_reduced)
    val_acc = xgboostModel.score(x_val_reduced, y_val) # 86.1%
    val_f1 = f1_score(y_val, predicted) # 62.0%
    
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x_val_reduced[:, 0], x_val_reduced[:, 1], c=predicted, cmap='viridis', edgecolor='k', alpha=0.7)
    
    # Add legend
    classes = np.unique(y_val)
    legend = plt.legend(handles=scatter.legend_elements()[0], labels=[f"Class {cls}" for cls in classes], title="Classes")
    plt.gca().add_artist(legend)
    
    # Add labels and title
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("Validation Data Reduced to 2D with PCA")
    plt.grid(alpha=0.3)
    plt.savefig('output/xgboost_pca.pdf')
    plt.show()
    '''
    
if __name__ == '__main__':
    main()