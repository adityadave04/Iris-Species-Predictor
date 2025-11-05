import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report 
import matplotlib.pyplot as plt 
from sklearn.tree import plot_tree
import joblib

df = pd.read_csv("IRIS.CSV")
print(df)

x = df.drop("species",axis=1)
y= df["species"]

print("Input features:",x)

print("Target output: ",y)


le=LabelEncoder()
y = le.fit_transform(y)
print("label encoding:",y)

X_train,X_test,Y_train,Y_test= train_test_split(x,y,test_size=0.2,random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train,Y_train)
y_pred=model.predict(X_test)
# print("y_pred:",y_pred)

print("Accuracy Test Score:",accuracy_score(Y_test,y_pred))
print("Classification Report:", classification_report(Y_test,y_pred))




# Save the model
joblib.dump(model, "iris_model.pkl")

# Save the label encoder too (so you can decode prediction later)
joblib.dump(le, "label_encoder.pkl")


# plt.figure(figsize=(10,6))
# plot_tree(model,feature_names=x.columns,class_names=le.classes_,filled=True)
# plt.title("Decision tree on iris dataset")
# plt.show()


# feature_importance = model.feature_importances_
# features = X_train.columns 
# for name ,score in zip(features,feature_importance):
#     print(f"{name}:{score:.4f}")

#     plt.figure(figsize=(8, 5))
# plt.bar(features, feature_importance, color='skyblue')
# plt.title("Feature Importance")
# plt.ylabel("Importance Score")
# plt.xlabel("Features")
# plt.tight_layout()
# plt.show()