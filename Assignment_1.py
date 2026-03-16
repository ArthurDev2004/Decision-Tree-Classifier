import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, OneHotEncoder # will be used to encode the categorical data points so they work with the Scikit implementation
from sklearn.model_selection import train_test_split, cross_val_score

drug_data = pd.read_csv('drug200.csv') # returns a Pandas dataframe of the data that was read in from the csv file 



# split the features and the target variable 
drug_data_target_classes = drug_data['Drug'] # target classes 
drug_data_features = drug_data.iloc[:, 0:5] # the data samples of each of the features

# assign the variables with conventional names 
y = drug_data_target_classes
X = drug_data_features

# split into training and testing data  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,  stratify=y) # will split the data in train and test data, with a 80 20 split with balance in the class proportions present in both splits because of stratify 

# encode the categorical data to work with the implementation
labelEncoder = LabelEncoder() # basic label encoding will be used to encode the target class variables in training set, done in alphabetical order of the categorical data 
encoded_class_labels_training = labelEncoder.fit_transform(y_train)
encoded_class_labels_testing = labelEncoder.transform(y_test) # no need to do 'fit' which is essentially getting more information about the categorical data like what categorical data is present, how many of each, and what to encode; can already just do transformation since training data did the fitting

y_train = encoded_class_labels_training
y_test = encoded_class_labels_testing

# encode the features 
BP_encoder = OrdinalEncoder()
encoded_BP_feature_training = BP_encoder.fit_transform(X_train[["BP"]])
encoded_BP_feature_testing = BP_encoder.transform(X_test[["BP"]])

cholesterol_encoder = OrdinalEncoder()
encoded_cholestrol_feature_training = cholesterol_encoder.fit_transform(X_train[["Cholesterol"]])
encoded_cholestrol_feature_testing = cholesterol_encoder.transform(X_test[["Cholesterol"]])

# one hot encoding for sex since not ordinal data with some order to it
sex_encoder = OneHotEncoder(sparse_output=False).set_output(transform="pandas") # will return a pandas data frame which will be useful for putting all of the features in a single data frame
encoded_sex_feature_training = sex_encoder.fit_transform(X_train[["Sex"]])
encoded_sex_feature_testing = sex_encoder.transform(X_test[["Sex"]])


# create the new dataframes (structured data) with the encoded data (Training Set)
X_train["BP"] = encoded_BP_feature_training
X_train["Cholesterol"] = encoded_cholestrol_feature_training
X_train = pd.concat([X_train, encoded_sex_feature_training], axis=1) # concatenates the features training data samples dataframe with the dataframe for the one hot encoded sex feature (axis 1 is horizontal like in Numpy)
X_train.drop(columns=["Sex"], inplace=True) # will drop the column for Sex which has the categorical data which was turned one hot encoded in numerical data

# created the new dataframe for the encdoded testing data 
X_test["BP"] = encoded_BP_feature_testing
X_test["Cholesterol"] = encoded_cholestrol_feature_testing
X_test = pd.concat([X_test, encoded_sex_feature_testing], axis=1) # concatenates the features training data samples dataframe with the dataframe for the one hot encoded sex feature (axis 1 is horizontal like in Numpy)
X_test.drop(columns=["Sex"], inplace=True) # will drop the column for Sex which has the categorical data which was turned one hot encoded in numerical data

# train/fit the model with the proper encoded and processed data  (Entropy)
entropy_decision_tree = DecisionTreeClassifier(criterion="entropy") # will use entropy as the criteria for determining splits and nodes in the tree

entropy_decision_tree.fit(X_train, y_train) # will train/fit the model using the testing set of data given

entropy_y_pred = entropy_decision_tree.predict(X_test) # is used to do the inference/prediction of the testing data 

entropy_accuracy = accuracy_score(y_test, entropy_y_pred)
print("Entropy Accuracy:", entropy_accuracy)

entropy_confusion = confusion_matrix(y_test, entropy_y_pred)
print("Entropy Confusion Matrix:",entropy_confusion, sep="\n")

# get the cross validation average between all iterations 
entropy_cross_validation_scores = cross_val_score(entropy_decision_tree, X=X_train, y=y_train ,cv=10)
print("Entropy Cross Validation Average:", entropy_cross_validation_scores.mean())


# train/fit the model using Gini impurity criterion
gini_tree = DecisionTreeClassifier(criterion="gini")
gini_tree.fit(X_train, y_train)

gini_y_pred = gini_tree.predict(X_test)

gini_accuracy = accuracy_score(y_test, gini_y_pred)
print("Gini Criterion Accuracy:", gini_accuracy)

gini_confusion_matrix = confusion_matrix(y_test, gini_y_pred)
print("Gini Confusion Martix:", gini_confusion_matrix, sep="\n")

gini_cross_validation_scores = cross_val_score(gini_tree, X=X_train, y=y_train ,cv=10)
print("Gini Cross Validation Average:", gini_cross_validation_scores.mean())


# train/fit the model using Log Lass Criterion
log_loss_tree = DecisionTreeClassifier(criterion="log_loss")
log_loss_tree.fit(X_train, y_train)

log_loss_y_pred = log_loss_tree.predict(X_test)

log_loss_accuracy = accuracy_score(y_test, log_loss_y_pred)
print("Log Loss Accuracy Score:", log_loss_accuracy)

log_loss_confusion_matrix = confusion_matrix(y_test, log_loss_y_pred)
print("Log Loss Confusion Martix:", log_loss_confusion_matrix, sep="\n")

log_loss_cross_validation_scores = cross_val_score(log_loss_tree, X=X_train, y=y_train ,cv=10)
print("Log Loss Validation Average:", log_loss_cross_validation_scores.mean())
print(log_loss_cross_validation_scores)


# train a decision tree with entropy criterion with a max depth of 1 
small_depth_entropy_tree = DecisionTreeClassifier(criterion="entropy",max_depth=1)
small_depth_entropy_tree.fit(X_train, y_train)
small_depth_entropy_pred = small_depth_entropy_tree.predict(X_test)
small_depth_entropy_accuracy = accuracy_score(y_test, small_depth_entropy_pred)
print("Entropy Small Depth Accuracy",small_depth_entropy_accuracy)

# small depth tree with gini criterion
small_depth_gini_tree = DecisionTreeClassifier(criterion="gini",max_depth=2)
small_depth_gini_tree.fit(X_train, y_train)
small_depth_gini_pred = small_depth_gini_tree.predict(X_test)
small_depth_gini_accuracy = accuracy_score(y_test, small_depth_gini_pred)
print("Gini Small Depth Accuracy",small_depth_gini_accuracy)


small_depth_log_loss_tree = DecisionTreeClassifier(criterion="log_loss",max_depth=3)
small_depth_log_loss_tree.fit(X_train, y_train)
small_depth_log_loss_pred = small_depth_log_loss_tree.predict(X_test)
small_depth_log_loss_accuracy = accuracy_score(y_test, small_depth_log_loss_pred)
print("Log Loss Small Depth Accuracy",small_depth_log_loss_accuracy)
