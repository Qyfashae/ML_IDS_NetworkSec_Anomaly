#Coded by: QyFashae

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preproccessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.enseble import IsolationForest

# Examine proportion/types of traffic.
wpshark_ds = pd.read_csv("wpshark_dataset.csv", index_col=None)
x = wpshark_ds["label"].values
Counter(x).most_common()

# Convert all abnormal observ into a single "class"
def label_anomalous(text):
	# Binarize target labels into normal or anomalous traffic
	if text == "normal":
		return 0
	else:
		return 1

# Obtain ratio of normal/abnormal observ, contaminated parameter will be used in our isolation-forest.
print("Calc, dont exit terminal No.0!")
wpshark_ds["label"] = wpshark_ds["label"].apply(label_anomalous)
x = wpshark_ds["label"].values
counts = Counter(x).most_common()
contamination_parameter = counts[1][1] / (counts[0][1] + counts[1][1])

# Convert all cat's features into numerical form.
enc_dic = dict()
for c in wpshark_ds.columns:
	if wpshark_ds[c].dtype == "object":
		enc_dic[c] = LabelEncoder()
		kff_dff[c] = enc_dic[c].fit_transform(wpshark_ds[c])

# Split the dataset into normal[0]/abnormal[1] observ.
wpshark_ds_normal = wpshark_ds[wpshark_ds["label"] == 0]
wpshark_ds_abnormal = wpshark_ds[wpshark_ds["label"] == 1]
x_normal = wpshark_ds_normal.pop("label").values
y_normal = wpshark_ds_normal.values
x_anomaly = wpshark_ds_abnormal.pop("label").values
y_anomaly = wpshark_ds_abnormal.values

# Training test and split the dataset.
print("Calc, dont exit terminal No.1!")
x_normal_train, x_normal_test, y_normal_train, y_normal_test = train_test_split(x_normal, y_normal, test_size=0.7, random_state=13)
x_anomaly_train, x_anomaly_test, y_anomaly_train, y_anomaly_test = train_test_split(x_anomaly, y_anomaly, test_size=0.7, random_state=13)

# Inst, train and isolation forest classifier.
 x_train = np.concatenate(x_normal_train, x_anomaly_train)
 y_train = np.concatenate(y_normal_train, y_anomaly_train)
 x_test = np.concatenate(x_normal_test, x_anomaly_test)
 y_test = np.concatenate(y_normal_test, y_anomaly_test)

# Score the classifier on normal/abnormal traffic.
 if_t = IsolationForest(contamination=contamination_parameter)if_t.fit(x_train) 
 decisionScores_train_normal = if_t.decision_function(x_normal_train)
 decisionScore_train_anomaly = if_t.decision_function(x_anomaly_train)

# Plot the scores for the normal traffic set.
%matplotlib inline
plt.figure(figsize=(20, 10))
_ = plt.hist(decisionScores_train_normal, bins=50)

# Plot the scores on the anomaly traffic for visual observ.
plt.figure(figsize=(20, 10))
_ plt.hist(decisionScores_train_anomaly, bins=50)

# Select a cut-off so as to separate out the anomalies from the normal traffic.
cutoff = 0 

# Examine this cut-off on the test set.
print(Counter(y_test))
print(Counter(y_test[cutoff > if_t.decision_function(x_test)]))
print("Calc done, you can now cat_for the result!")
