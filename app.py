from flask import Flask, render_template
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score, roc_auc_score, silhouette_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

app=Flask(__name__)


def analyze():
    df=pd.read_csv('marketing_campaign_DS14.csv', sep=None, engine='python')
    df['Dt_Customer']=pd.to_datetime(df['Dt_Customer'], dayfirst=True, errors='coerce')
    rows, cols=df.shape
    response=float(df.Response.mean()*100)
    avg_sp=float(df.Total_Spending.mean())
    avg_inc=float(df.Income.mean())
    corr=float(df.Income.corr(df.Total_Spending))

    lin_features=['Income','Customer_Age','Total_Children','Recency','NumDealsPurchases','NumWebPurchases','NumCatalogPurchases','NumStorePurchases','NumWebVisitsMonth','Campaign_Acceptance_Count','Customer_Tenure_Months','Digital_Engagement_Score','Loyalty_Score']
    X=df[lin_features]; y=df.Total_Spending
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
    lm=LinearRegression().fit(Xtr,ytr); pred=lm.predict(Xte)
    r2=float(r2_score(yte,pred)); adj=float(1-(1-r2)*(len(yte)-1)/(len(yte)-len(lin_features)-1))
    rmse=float(mean_squared_error(yte,pred)**.5); mae=float(mean_absolute_error(yte,pred))

    log_features=['Income','Customer_Age','Total_Children','Recency','NumDealsPurchases','NumWebPurchases','NumCatalogPurchases','NumStorePurchases','NumWebVisitsMonth','AcceptedCmp1','AcceptedCmp2','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5','Complain','Campaign_Acceptance_Count','Customer_Tenure_Months','Digital_Engagement_Score','Loyalty_Score','Satisfaction_Score']
    X=df[log_features]; y=df.Response
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    lg=Pipeline([('scale',StandardScaler()),('model',LogisticRegression(max_iter=2000,class_weight='balanced'))]).fit(Xtr,ytr)
    lp=lg.predict(Xte); lprob=lg.predict_proba(Xte)[:,1]
    log={'accuracy':float(accuracy_score(yte,lp)),'precision':float(precision_score(yte,lp,zero_division=0)),'recall':float(recall_score(yte,lp,zero_division=0)),'auc':float(roc_auc_score(yte,lprob))}

    cf=['Recency','Total_Spending','Total_Purchases','NumWebPurchases','NumCatalogPurchases','NumStorePurchases','Income','Digital_Engagement_Score','Loyalty_Score']
    Z=StandardScaler().fit_transform(df[cf]); scores={}
    for k in range(2,7):
        km=KMeans(n_clusters=k,random_state=42,n_init=20); lab=km.fit_predict(Z); scores[k]=float(silhouette_score(Z,lab))
    best=max(scores,key=scores.get); km=KMeans(n_clusters=best,random_state=42,n_init=20); df['Cluster']=km.fit_predict(Z)
    prof=df.groupby('Cluster')[cf].mean().round(2)
    clusters=[]
    for cid,row in prof.iterrows():
        clusters.append({'id':int(cid),'customers':int((df.Cluster==cid).sum()),'income':float(row.Income),'spending':float(row.Total_Spending),'purchases':float(row.Total_Purchases),'digital':float(row.Digital_Engagement_Score),'loyalty':float(row.Loyalty_Score)})

    tree=DecisionTreeClassifier(max_depth=4,min_samples_leaf=20,random_state=42,class_weight='balanced').fit(Xtr,ytr)
    tp=tree.predict(Xte); tprob=tree.predict_proba(Xte)[:,1]
    tm={'accuracy':float(accuracy_score(yte,tp)),'precision':float(precision_score(yte,tp,zero_division=0)),'recall':float(recall_score(yte,tp,zero_division=0)),'auc':float(roc_auc_score(yte,tprob))}
    ti=sorted(zip(log_features,tree.feature_importances_),key=lambda x:x[1],reverse=True)[:5]

    rf=RandomForestClassifier(n_estimators=300,max_depth=6,min_samples_leaf=10,random_state=42,class_weight='balanced').fit(Xtr,ytr)
    rp=rf.predict(Xte); rprob=rf.predict_proba(Xte)[:,1]
    rfm={'accuracy':float(accuracy_score(yte,rp)),'precision':float(precision_score(yte,rp,zero_division=0)),'recall':float(recall_score(yte,rp,zero_division=0)),'auc':float(roc_auc_score(yte,rprob))}
    ri=sorted(zip(log_features,rf.feature_importances_),key=lambda x:x[1],reverse=True)[:5]
    seg=(df.groupby('Customer_Segment').Response.mean()*100).sort_values(ascending=False).round(1).to_dict()
    return dict(rows=rows,cols=cols,response=response,avg_spending=avg_sp,avg_income=avg_inc,corr=corr,missing=int(df.isna().sum().sum()),duplicates=int(df.duplicated().sum()),r2=r2,adj=adj,rmse=rmse,mae=mae,log=log,cluster={'k':best,'sil':scores[best],'profiles':clusters},tree={'m':tm,'imp':ti},rf={'m':rfm,'imp':ri},seg=seg)

@app.route('/')
def home():
    try: return render_template('index.html',r=analyze(),error=None)
    except Exception as e: return render_template('index.html',r=None,error=str(e)),500

@app.route('/health')
def health(): return {'status':'ok'}

if __name__=='__main__':
    app.run(host='0.0.0.0',port=int(os.environ.get('PORT',8000)))
