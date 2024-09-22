# KNN Classification to predict the Rainfall Level of Mymensing
# KNN Classification to predict the Rainfall Level of Mymensing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix




D=pd.read_csv('E:/Village of Study/PMASDS/PM-ASDS 06 MULTIVARIATE ANALYSIS/Mymensingh.csv')
D.dropna(how='any',axis=0, inplace=True)
D.shape
D.head()
DD=D.drop(['ID','Station','Year','Month','T_RAN','A_RAIN'], axis=1)

DD.head()
X=DD.drop(['RAN'], axis=1)
Y=DD['RAN']



#Y=DD['RAN']
#X=DD.drop(['RAN'], axis=1)


np.random.seed(104729)
X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.25, random_state=100)

KNN=KNeighborsClassifier(n_neighbors=8,metric='euclidean')
KNN.fit(X_train, Y_train)

P=KNN.predict(X_test)
#P2=KNN.predict(X_train)

accuracy_score(Y_test, P)

#accuracy_score(Y_train, P2)

print(confusion_matrix(Y_test, P))

print(classification_report(Y_test, P))

# Optimal Value of k and Accuracy Rate for Optimal k
Ar=[]
Er=[]
As=[]
Es=[]
kk=[]
# Calculating error for K values between 1 and 40
for i in range(2, 40):
    knn = KNeighborsClassifier(n_neighbors=i,metric='euclidean')
    knn.fit(X_train, Y_train)
    pred_r = knn.predict(X_train)
    pred_i = knn.predict(X_test)
    Ar.append(accuracy_score(Y_train, pred_r))
    Er.append(1-accuracy_score(Y_train, pred_r))
    As.append(accuracy_score(Y_test, pred_i))
    Es.append(1-accuracy_score(Y_test, pred_i))
    kk.append(i)
   
    #pred_i = knn.predict(X_test)
    #error.append(np.mean(pred_i != Y_test))
    #accuracy.append(np.mean(pred_i == Y_test))

kk

# Plot for Accuracy rate and Error Rate for training and Test data
plt.figure(figsize=(9, 5))
plt.plot(range(2, 40), Ar, color='purple', linestyle='solid', marker='o',linewidth =2, markerfacecolor='purple', markersize=3)
plt.plot(range(2, 40), As, color='black', linestyle='solid', marker='o',linewidth =2, markerfacecolor='black', markersize=3)
plt.plot(range(2, 40), Er, color='red', linestyle='solid', marker='o',linewidth =2, markerfacecolor='red', markersize=3)
plt.plot(range(2, 40), Es, color='blue', linestyle='solid', marker='o',linewidth =2, markerfacecolor='blue', markersize=3)
plt.ylim([0,1])

plt.figure(figsize=(9, 5))
plt.plot(range(2, 40), Es, color='red', linestyle='solid', marker='o',linewidth =2, markerfacecolor='blue', markersize=10)
# only one line may be specified; full height
plt.axvline(x=8, color='b',ls='--',linewidth=2)
plt.title('Optimal value of k')
plt.xlabel('Value of k')
plt.ylabel('Error Rate')
#plt.savefig("/content/drive/MyDrive/Colab Notebooks/KNN/Optimal_Value_of_k_1.jpg", dpi=500)
plt.show()

plt.figure(figsize=(9, 5))
plt.plot(range(2, 40), As, color='red', linestyle='solid', linewidth =2, marker='o',
         markerfacecolor='blue', markersize=10)
# only one line may be specified; full height
plt.axvline(x=8, color='b',ls='--',linewidth=2,)
plt.title('Optimal value of k')
plt.xlabel('Value of k')
plt.ylabel('Accuracy Rate')
#plt.savefig("/content/drive/MyDrive/Colab Notebooks/KNN/Optimal_Value_of_k_2.jpg", dpi=500)
plt.show()




k=pd.DataFrame(Es).idxmin()+2
k
k[0]



#X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.30, random_state=100)
#KNN1=KNeighborsClassifier(n_neighbors=11)
np.random.seed(104729)
KNN1=KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='euclidean',
           metric_params=None, n_jobs=1, n_neighbors=k[0], p=2,
           weights='uniform')
KNN1.fit(X_train, Y_train)
P1=KNN1.predict(X_test)
#P1


accuracy_score(Y_test, P1)

print(confusion_matrix(Y_test, P1))

print(classification_report(Y_test, P1))

#====ROC Curve=====#
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import normalize
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix as cm

y_score=KNN1.predict_proba(X_test)
#Convert the target variable to one-hot encoding using Label Binarizer
LB=LabelBinarizer().fit(Y_train)
y_onehot_test=LB.transform(Y_test)

fpr_list=[]
tpr_list=[]
auc_list=[]

for class_id, class_name in enumerate(LB.classes_):
    fpr, tpr,_= roc_curve(y_onehot_test[:,class_id], y_score[: , class_id])
    auc_value=roc_auc_score(y_onehot_test[:,class_id], y_score[ :, class_id])
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    auc_list.append(auc_value)


plt.figure(figsize=(6,6))
for class_id, class_name in enumerate(LB.classes_):
    plt.plot(fpr_list[class_id], tpr_list[class_id], linewidth=2,label=f"{class_name} (AUC={auc_list[class_id]:0.3f})")

plt.plot([0,1], [0,1], color='purple',linestyle='-',linewidth=2)
plt.xlabel('False Positive Rate')
plt.xlabel('True Positive Rate')
plt.title('ROC curve for all classes for NB')
plt.legend()
plt.show()


from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score

K=[5,7]
p=-1
S=np.zeros((len(K),10))

for i in K:
    scoring_01 = ['accuracy','precision_macro','recall_macro','f1_macro']
    KN1=KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='euclidean',
           metric_params=None, n_jobs=1, n_neighbors=k[0], p=2, weights='uniform')
    #KN.fit(X, Y)
    scores_01 =cross_validate(KN1, X, Y,  cv=i, scoring=scoring_01,return_train_score=True)
    p=p+1
    QW=pd.DataFrame.mean(pd.DataFrame(scores_01), axis=0)
    S[p,]=np.array(QW)

scores_01.keys()

S1=pd.DataFrame(S)
S1.columns =['fit_time', 'score_time', 'test_accuracy', 'train_accuracy', 'test_precision_macro', 'train_precision_macro', 'test_recall_macro', 'train_recall_macro', 'test_f1_macro', 'train_f1_macro']
# Using DataFrame.insert() to add a column
S1.insert(0, "K-fold", K, True)
np.round(S1,4)

