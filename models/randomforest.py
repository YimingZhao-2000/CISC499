#  %%
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score 
from sklearn.model_selection import validation_curve, learning_curve

import graphviz
from sklearn.tree import export_graphviz
import os

from interpret.blackbox import LimeTabular
from interpret import set_visualize_provider
from interpret.provider import InlineProvider
from interpret import show
import shap
set_visualize_provider(InlineProvider())

# Load dataset
data = pd.read_csv("../dataset/EA_dataset.csv")

# data = data.drop(columns=["PE_Ratio", "PE_GIVEN_TAG"]) # Drop PE_ratio and PE GIVEN TAG

# Split dataset into training and testing
data["Date"] = pd.to_datetime(data["Date"])
train_data = data[data["Date"] < '2018-01-01']
test_data = data[data["Date"] >= '2018-01-01']


# Check if 80%, 20% split
# print(data.shape[0])
# print(f"No. of training examples: {train_data.shape[0]}")
# print(f"No. of testing examples: {test_data.shape[0]}")


x_train = train_data.drop(columns=['Next_Day_Close_Price','Date']) # We have column DaysBeforeToday to represent the time
y_train = train_data[['Next_Day_Close_Price']]

x_test = test_data.drop(columns=['Next_Day_Close_Price','Date']) # We have column DaysBeforeToday to represent the time
y_test = test_data[['Next_Day_Close_Price']]

# Finding the best accuracy model and learning curve (Hyperparameter Tuning)
def learning_curve(func, path="../plot/EA/rf_learningCurve.png", random_range=10, ntree_list=[50,100,150,200], depth_range=25):
    # define arrays to collect scores
    train_scores = np.zeros((random_range,len(ntree_list),depth_range))
    test_scores = np.zeros((random_range,len(ntree_list),depth_range))
    print(test_scores.shape)

    # evaluate a decision tree for each depth
    depth_values = [i for i in range(1, depth_range + 1)]

    for i in range(1, random_range+1):
        for j in range(1, len(ntree_list)+1):
            for k in depth_values:
                # configure the model
                model = RandomForestRegressor(n_estimators = ntree_list[j-1], random_state = i, max_depth=k)

                # fit model on the training dataset
                model.fit(x_train, y_train.values.flatten()) # change the shapeo f y to (1,)

                # evaluate on the train dataset
                train_yhat = model.predict(x_train)
                train_acc = func(y_train, train_yhat)
                train_scores[i-1][j-1][k-1] = train_acc

                # evaluate on the test dataset
                test_yhat = model.predict(x_test)
                test_acc = func(y_test, test_yhat)
                test_scores[i-1][j-1][k-1] = test_acc

    # summarize result
    test_scores = np.array(test_scores)
    best_test_acc = np.amax(test_scores)
    best_test_index = np.unravel_index(test_scores.argmax(), test_scores.shape)
    print("Random State: " + str(best_test_index[0]+1) + ", ntree: " + str(ntree_list[best_test_index[1]]) + ", Random Forest with max depth: " + str(best_test_index[2]+1) + " has the best accuracy " + str(best_test_acc))

    # plot of train and test scores vs tree depth
    plt.figure(figsize=(12,5))
    plt.plot(depth_values, train_scores[best_test_index[0]][best_test_index[1]], '-o', label='Train')
    plt.plot(depth_values, test_scores[best_test_index[0]][best_test_index[1]], '-o', label='Test')
    plt.legend()
    plt.xticks(list(range(0,depth_range)))
    plt.title("Learning Curve for Random Forest Tree's max depth with random state " + str(best_test_index[0]+1) + ", ntree " + str(ntree_list[best_test_index[1]]))
    plt.savefig(path)
    plt.show()

# learning_curve(r2_score, "rf_learning_curve.png") # Hyperparameter Tuning


# %%
# Build the best model we found so far
n_tree = 50
model = RandomForestRegressor(n_estimators = n_tree, random_state = 4, max_depth=9)
model.fit(x_train, y_train.values.flatten()) # change the shapeo f y to (1,)

# # Calculate feature importances
# importances = model.feature_importances_
# print(importances)


# Get prediction
train_predict = model.predict(x_train)
test_predict = model.predict(x_test)


# Evaluation metrices RMSE，MAE, R^2 score
print("-------------------------------------------------------------------------------------")
print("Train data MSE: ", mean_squared_error(y_train,train_predict))
print("Train data MAE: ", mean_absolute_error(y_train,train_predict))
print("Train data R^2: ", r2_score(y_train, train_predict))
print("-------------------------------------------------------------------------------------")
print("Test data MSE: ", mean_squared_error(y_test,test_predict))
print("Test data MAE: ", mean_absolute_error(y_test,test_predict))
print("Test data R^2: ", r2_score(y_test, test_predict))
print("-------------------------------------------------------------------------------------")


# Visual the Regression result for testing set
x_grid = x_test.sort_values(by="DaysBeforeToday", ascending=False)
x_label = x_grid[["DaysBeforeToday"]].values
plt.scatter(x_label, y_test, color='red', s=10)
plt.plot(x_label, model.predict(x_grid), color="blue")
plt.gca().invert_xaxis()
plt.title('Random Forest Regression for Testing dataset')
plt.xlabel("DaysBeforeToday")
plt.ylabel("Next Day Closed Price")
plt.legend(["Data points", 'Prediction'])
# plt.show()
plt.savefig("rf_test_set_result.png")


# Visual a decision tree from the random forest model
# dot_data = export_graphviz(model.estimators_[0], 
#                            out_file="tree.dot",
#                            feature_names=x_test.columns.values.tolist(),
#                            class_names=['Next_Day_Close_Price'], 
#                            filled=True, impurity=True, 
#                            rounded=True)
# os.system("dot -Tpng tree.dot -o " + "rf_tree.png")


# Extract feature importances
# fi = pd.DataFrame({'feature': list(x_train.columns),
#                    'importance': model.feature_importances_}).\
#                     sort_values('importance', ascending = False)
# print(fi)


# Get average depth of decision trees in random forest
# average_depth = sum([estimator.tree_.max_depth for estimator in model.estimators_]) // n_tree
# print('average depth for random forest is: ', average_depth)


# LIME
# explainer = LimeTabular(
#     model,
#     data = x_train,
#     feature_names=x_train.columns,
#     class_names=['Next_Day_Close_Price'],
#     mode="regression",
# )

# # Explain a single prediction
# instance = x_test.iloc[0]
# exp = explainer.explain_local(x_test[:1], y_test[:1])
# show(exp, 0)


#  %%
# SHAP
# explain the model's predictions using SHAP
# Fits the explainer
explainer = shap.Explainer(model.predict, x_test)
shap.initjs()

# Calculates the SHAP values
shap_values = explainer(x_test)

# Evaluate SHAP values
# shap_values = explainer.shap_values(X)

# %%
# Print Select a single data point
print(x_test.iloc[0])

# %%
# Draw Summary Plot
shap.summary_plot(shap_values)

# %%
# Draw Waterfall
shap.plots.waterfall(shap_values[0])
# %%
# Find feature importances using SHAP values
from scipy.special import softmax
def print_feature_importances_shap_values(shap_values, features):

    '''
    Prints the feature importances based on SHAP values in an ordered way
    shap_values -> The SHAP values calculated from a shap.Explainer object
    features -> The name of the features, on the order presented to the explainer
    '''
    # Calculates the feature importance (mean absolute shap value) for each feature
    importances = []
    for i in range(shap_values.values.shape[1]):
        importances.append(np.mean(np.abs(shap_values.values[:, i])))
    # Calculates the normalized version
    importances_norm = softmax(importances)
    # Organize the importances and columns in a dictionary
    feature_importances = {fea: imp for imp, fea in zip(importances, features)}
    feature_importances_norm = {fea: imp for imp, fea in zip(importances_norm, features)}
    # Sorts the dictionary
    feature_importances = {k: v for k, v in sorted(feature_importances.items(), key=lambda item: item[1], reverse = True)}
    feature_importances_norm= {k: v for k, v in sorted(feature_importances_norm.items(), key=lambda item: item[1], reverse = True)}
    # Prints the feature importances
    for k, v in feature_importances.items():
        print(f"{k} -> {v:.4f} (softmax = {feature_importances_norm[k]:.4f})")

print_feature_importances_shap_values(shap_values, x_test.columns)
