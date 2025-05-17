import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# to start the mlflow tracking server
mlflow.set_tracking_uri('http://localhost:5000')

iris=load_iris()
X=iris.data
y=iris.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

max_depth=9

# set experiment will create a experiment if not created 
mlflow.set_experiment('iris-dt')

#apply mlflow
# with mlflow.start_run(experiment_id='729668167199527193'):  
with mlflow.start_run():
    dt=DecisionTreeClassifier(max_depth=max_depth)
    dt.fit(X_train,y_train)
    y_pred=dt.predict(X_test)
    accuracy=accuracy_score(y_test,y_pred)
       # Create a confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    # Save the plot as an artifact
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")   # to log plots 
    mlflow.log_artifact(__file__)
    mlflow.sklearn.log_model(dt,'decision tree')
    mlflow.set_tag('author','prasann')
    mlflow.set_tag('model','decision tree')
    mlflow.log_metric('accuracy',accuracy)
    mlflow.log_param('max_depth',max_depth)
    print('accuracy',accuracy)
