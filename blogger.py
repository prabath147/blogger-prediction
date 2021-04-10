import tkinter as tk
import pandas as pd
root=tk.Tk()
root.config(bg='blue')
root.title('hi')
root.geometry('3100x3100')
heading='Blogger'
r=tk.Label(root,text=heading,font=("Arial", 22),bg='blue')
r.pack()
df=pd.read_excel('https://archive.ics.uci.edu/ml/machine-learning-databases/00255/kohkiloyeh.xlsx')


df=df.replace('yes',1)
df=df.replace('no',0)


v1=set(df['Degree'])
v1=list(v1)


s1=[4,5,6]
df=df.replace(v1,s1)


v2=set(df['caprice'])
v2=list(v2)
s2=[7,8,9]
df=df.replace(v2,s2)


v3=set(df['topic'])
v3=list(v3)
s3=[11,12,13,14,15]
df=df.replace(v3,s3)


t3=df[['Degree','caprice','topic','lmt','lpss']]
r3=df['pb']


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(t3,r3,test_size=0.3)

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
sc=StandardScaler()
x_train_sc=sc.fit_transform(x_train)
x_test_sc=sc.fit_transform(x_test)
r9=tk.Label(root,text="",font=("Arial", 12))
r9.pack()
r9.place(relx=0.58,rely=0.6)
def pct():
    r9.destroy()

    from sklearn.linear_model import Perceptron
    
    pc=Perceptron()
    pc.fit(x_train_sc,y_train)
    

   # print(accuracy_score(pc.predict(x_train_sc),y_train))
   # print(accuracy_score(pc.predict(x_test_sc),y_test))
    #print(confusion_matrix(y_train, pc.predict(x_train_sc)))
    #print(confusion_matrix(y_test, pc.predict(x_test_sc)))
    sp="accuracy of training :%.2f"%(accuracy_score(pc.predict(x_train_sc),y_train))
    sp1="accuracy of testing :%.2f"%(accuracy_score(pc.predict(x_test_sc),y_test))
    sp2="confusion matrix for training :",(confusion_matrix(y_train, pc.predict(x_train_sc)))
    sp3="confusion matrix for testing :",(confusion_matrix(y_test, pc.predict(x_test_sc)))
    r91=tk.Label(root,text=sp,font=("Arial", 12))
    r91.pack()
    r91.place(relx=0.58,rely=0.6)
    r92=tk.Label(root,text=sp1,font=("Arial", 12))
    r92.pack()
    r92.place(relx=0.58,rely=0.65)
    r93=tk.Label(root,text=sp2,font=("Arial", 12))
    r93.pack()
    r93.place(relx=0.58,rely=0.70)
    r94=tk.Label(root,text=sp3,font=("Arial", 12))
    r94.pack()
    r94.place(relx=0.58,rely=0.75)
    sp4="predicted train values",(pc.predict(x_train_sc))
    r95=tk.Label(root,text=sp4,font=("Arial", 12))
    r95.pack()
    r95.place(relx=0.1,rely=0.3)
    sp5="predicted test values",(pc.predict(x_test_sc))
    r95=tk.Label(root,text=sp5,font=("Arial", 12))
    r95.pack()
    r95.place(relx=0.1,rely=0.4)
    
def lgr():
    r9.destroy()
    from sklearn.linear_model import LogisticRegression
    lr=LogisticRegression()
    lr.fit(x_train_sc,y_train)
   # print(accuracy_score(lr.predict(x_train_sc),y_train))
    #print(accuracy_score(lr.predict(x_test_sc),y_test))
    #print(confusion_matrix(y_train, lr.predict(x_train_sc)))
    #print(confusion_matrix(y_test, lr.predict(x_test_sc)))
    sp="accuracy of training :%.2f"%(accuracy_score(lr.predict(x_train_sc),y_train))
    sp1="accuracy of testing :%.2f"%(accuracy_score(lr.predict(x_test_sc),y_test))
    sp2="confusion matrix for training :",(confusion_matrix(y_train, lr.predict(x_train_sc)))
    sp3="confusion matrix for testing :",(confusion_matrix(y_test, lr.predict(x_test_sc)))
    r91=tk.Label(root,text=sp,font=("Arial", 12))
    r91.pack()
    r91.place(relx=0.58,rely=0.6)
    r92=tk.Label(root,text=sp1,font=("Arial", 12))
    r92.pack()
    r92.place(relx=0.58,rely=0.65)
    r93=tk.Label(root,text=sp2,font=("Arial", 12))
    r93.pack()
    r93.place(relx=0.58,rely=0.70)
    r94=tk.Label(root,text=sp3,font=("Arial", 12))
    r94.pack()
    r94.place(relx=0.58,rely=0.75)
    sp4="predicted train values",(lr.predict(x_train_sc))
    r95=tk.Label(root,text=sp4,font=("Arial", 12))
    r95.pack()
    r95.place(relx=0.1,rely=0.3)
    sp5="predicted test values",(lr.predict(x_test_sc))
    r95=tk.Label(root,text=sp5,font=("Arial", 12))
    r95.pack()
    r95.place(relx=0.1,rely=0.4)

def svm():
    r9.destroy()
    from sklearn.svm import  SVC
    svc=SVC()
    svc.fit(x_train_sc,y_train)
   # print(accuracy_score(svc.predict(x_train_sc),y_train))
    #print(accuracy_score(svc.predict(x_test_sc),y_test))
    #print(confusion_matrix(y_train, svc.predict(x_train_sc)))
    #print(confusion_matrix(y_test, svc.predict(x_test_sc)))
    sp="accuracy of training :%.2f"%(accuracy_score(svc.predict(x_train_sc),y_train))
    sp1="accuracy of testing :%.2f"%(accuracy_score(svc.predict(x_test_sc),y_test))
    sp2="confusion matrix for training :",(confusion_matrix(y_train, svc.predict(x_train_sc)))
    sp3="confusion matrix for testing :",(confusion_matrix(y_test, svc.predict(x_test_sc)))
    r91=tk.Label(root,text=sp,font=("Arial", 12))
    r91.pack()
    r91.place(relx=0.58,rely=0.6)
    r92=tk.Label(root,text=sp1,font=("Arial", 12))
    r92.pack()
    r92.place(relx=0.58,rely=0.65)
    r93=tk.Label(root,text=sp2,font=("Arial", 12))
    r93.pack()
    r93.place(relx=0.58,rely=0.70)
    r94=tk.Label(root,text=sp3,font=("Arial", 12))
    r94.pack()
    r94.place(relx=0.58,rely=0.75)
    sp4="predicted train values",(svc.predict(x_train_sc))
    r95=tk.Label(root,text=sp4,font=("Arial", 12))
    r95.pack()
    r95.place(relx=0.1,rely=0.3)
    sp5="predicted test values",(svc.predict(x_test_sc))
    r95=tk.Label(root,text=sp5,font=("Arial", 12))
    r95.pack()
    r95.place(relx=0.1,rely=0.4)
    
def rc():
    r9.destroy()
    from sklearn.ensemble import RandomForestClassifier
    rfc=RandomForestClassifier(n_estimators=10)
    rfc.fit(x_train_sc,y_train)
    #print(accuracy_score(rfc.predict(x_train_sc),y_train))
    #print(accuracy_score(rfc.predict(x_test_sc),y_test))
    #print(confusion_matrix(y_train, rfc.predict(x_train_sc)))
    #print(confusion_matrix(y_test, rfc.predict(x_test_sc)))
    sp="accuracy of training :%.2f"%(accuracy_score(rfc.predict(x_train_sc),y_train))
    sp1="accuracy of testing :%.2f"%(accuracy_score(rfc.predict(x_test_sc),y_test))
    sp2="confusion matrix for training :",(confusion_matrix(y_train, rfc.predict(x_train_sc)))
    sp3="confusion matrix for testing :",(confusion_matrix(y_test, rfc.predict(x_test_sc)))
    r91=tk.Label(root,text=sp,font=("Arial", 12))
    r91.pack()
    r91.place(relx=0.58,rely=0.6)
    r92=tk.Label(root,text=sp1,font=("Arial", 12))
    r92.pack()
    r92.place(relx=0.58,rely=0.65)
    r93=tk.Label(root,text=sp2,font=("Arial", 12))
    r93.pack()
    r93.place(relx=0.58,rely=0.70)
    r94=tk.Label(root,text=sp3,font=("Arial", 12))
    r94.pack()
    r94.place(relx=0.58,rely=0.75)
    sp4="predicted train values",(rfc.predict(x_train_sc))
    r95=tk.Label(root,text=sp4,font=("Arial", 12))
    r95.pack()
    r95.place(relx=0.1,rely=0.3)
    sp5="predicted test values",(rfc.predict(x_test_sc))
    r95=tk.Label(root,text=sp5,font=("Arial", 12))
    r95.pack()
    r95.place(relx=0.1,rely=0.4)
def kc():
    r9.destroy()
    from sklearn.neighbors import KNeighborsClassifier
    clf=KNeighborsClassifier(n_neighbors=5)
    clf.fit(x_train_sc,y_train)
    y_pred=clf.predict(x_test_sc)
    
    
    from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
   # print(confusion_matrix(y_test,y_pred))
    #print(classification_report(y_test,y_pred))
    #print('Accuracy score %.2f'%accuracy_score(y_test,y_pred))
    sp="accuracy of training :%.2f"%(accuracy_score(clf.predict(x_train_sc),y_train))
    sp1="accuracy of testing :%.2f"%(accuracy_score(clf.predict(x_test_sc),y_test))
    sp2="confusion matrix for training :",(confusion_matrix(y_train, clf.predict(x_train_sc)))
    sp3="confusion matrix for testing :",(confusion_matrix(y_test, clf.predict(x_test_sc)))
    r91=tk.Label(root,text=sp,font=("Arial", 12))
    r91.pack()
    r91.place(relx=0.58,rely=0.6)
    r92=tk.Label(root,text=sp1,font=("Arial", 12))
    r92.pack()
    r92.place(relx=0.58,rely=0.65)
    r93=tk.Label(root,text=sp2,font=("Arial", 12))
    r93.pack()
    r93.place(relx=0.58,rely=0.70)
    r94=tk.Label(root,text=sp3,font=("Arial", 12))
    r94.pack()
    r94.place(relx=0.58,rely=0.75)
    sp4="predicted train values",(clf.predict(x_train_sc))
    r95=tk.Label(root,text=sp4,font=("Arial", 12))
    r95.pack()
    r95.place(relx=0.1,rely=0.3)
    sp5="predicted test values",(clf.predict(x_test_sc))
    r95=tk.Label(root,text=sp5,font=("Arial", 12))
    r95.pack()
    r95.place(relx=0.1,rely=0.4)





b1=tk.Button(root,text="perceptron",command=pct)

b1.pack()

b2=tk.Button(root,text="logistic regression",command=lgr)

b2.pack()

b3=tk.Button(root,text="svm",command=svm)

b3.pack()

b4=tk.Button(root,text="random forest classifier",command=rc)

b4.pack()

b5=tk.Button(root,text="K Neighbour classifier",command=kc)

b5.pack()
b1.place(relx=0.2,rely=0.2)
b2.place(relx=0.3,rely=0.2)
b3.place(relx=0.45,rely=0.2)
b4.place(relx=0.55,rely=0.2)
b5.place(relx=0.7,rely=0.2)

root.mainloop()