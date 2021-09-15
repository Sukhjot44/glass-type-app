import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import plot_confusion_matrix,classification_report

# Load the dataset.
@st.cache()
def load_data():
	df=pd.read_csv("glass-types.csv",header=None)
	df.drop(columns=0,inplace=True)
	column_header=['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType']
# Create the required Python dictionary.
	df.rename(columns=dict(zip(df.columns,column_header),axis=1),inplace=True)
	return df

df=load_data()
 
X=df.iloc[:,:-1]
y=df['GlassType']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=42)

@st.cache()
def prediction(model,ri,na,mg,al,si,k,ca,ba,fe):
	glass_type=model.predict([[ri,na,mg,al,si,k,ca,ba,fe]])
	glass_type=glass_type[0]
	if glass_type==1:
		return 'building windows float processed'.upper()
	elif glass_type==2:
		return 'building windows non-float processed'.upper()
	elif glass_type==3:
		return 'vehicle windows float processed'.upper()
	elif glass_type==4:
		return 'vehicle windows non-float processed'.upper()
	elif glass_type==5:
		return ' containers'.upper()
	elif glass_type==6:
		return 'tableware'.upper()
	else:
		return 'headlamps'.upper()

st.title('Glass Type Predicter')
st.sidebar.title('Exploratory Data Analysis')
if st.sidebar.checkbox('show raw data'):
	st.subheader('full dataset')
	st.dataframe(df)

st.sidebar.subheader('scatterplot')
features_list=st.sidebar.multiselect('select the x axis values',('RI','Na','Mg','Al','Si','K','Ca','Ba','Fe'))
st.set_option('deprecation.showPyplotGlobalUse',False)
for feature in features_list:
	st.subheader(f'scatterplot bw {feature} and glass type')
	plt.figure(figsize=(12,6))
	sns.scatterplot(x=feature,y='GlassType',data=df)
	st.pyplot()


st.sidebar.subheader('scatterplot')
st.set_option('deprecation.showPyplotGlobalUse',False)
for feature in features_list:
	st.subheader(f'scatterplot bw {feature} and glass type')
	plt.figure(figsize=(12,6))
	sns.scatterplot(x=feature,y='GlassType',data=df)
	st.pyplot()
st.sidebar.subheader('visualization selector')
plot_types=st.sidebar.multiselect('select the plots or charts',('Histogram','Boxplot','Countplot','Piechart','Correlation heat map','Pairplot'))
if 'Histogram' in plot_types:
	st.subheader('Histogram')
	cols=st.sidebar.selectbox('select the columns to create a histogram ',('RI','Na','Mg','Al','Si','K','Ca','Ba','Fe'))
	plt.figure(figsize=(12,6))
	plt.title(f'Histogram {cols}')
	plt.hist(df[cols],bins='sturges',edgecolor='black')
	st.pyplot()

if 'Boxplot' in plot_types:
	st.subheader('Boxplot')
	cols=st.sidebar.selectbox('select the columns to create a Boxplot ',('RI','Na','Mg','Al','Si','K','Ca','Ba','Fe'))
	plt.figure(figsize=(12,6))
	plt.title(f' Boxplot for {cols}')
	sns.boxplot(df[cols])
	st.pyplot()
if 'Countplot' in plot_types:
	st.subheader('Countplot')
	plt.figure(figsize=(12,6))
	sns.countplot(x='GlassType',data=df)
	st.pyplot()

if 'Piechart' in plot_types:
	st.subheader('Piechart')
	pie_data=df['GlassType'].value_counts()
	plt.figure(figsize=(12,6))
	plt.title(f' Piechart for GlassType')
	plt.pie(pie_data,labels=pie_data.index,autopct='%1.2f%%',startangle=30,explode=np.linspace(0.06,0.16,6))
	st.pyplot()

if 'Correlation heat map' in plot_types:
	st.subheader('Correlation heat map')
	plt.figure(figsize=(12,6))
	plt.title(f'Correlation heat map')
	sns.heatmap(df.corr(),annot=True)
	st.pyplot()

if 'Pairplot' in plot_types:
	st.subheader('Pairplot')
	plt.figure(figsize=(12,6))
	plt.title(f' Pairplot')
	sns.pairplot(df)
	st.pyplot()

st.sidebar.subheader('select the values ')
ri=st.sidebar.slider('input RI',float(df['RI'].min()),float(df['RI'].max()))
na=st.sidebar.slider('input Na',float(df['Na'].min()),float(df['Na'].max()))
mg=st.sidebar.slider('input Mg',float(df['Mg'].min()),float(df['Mg'].max()))
al=st.sidebar.slider('input Al',float(df['Al'].min()),float(df['Al'].max()))
si=st.sidebar.slider('input Si',float(df['Si'].min()),float(df['Si'].max()))
K=st.sidebar.slider('input K',float(df['K'].min()),float(df['K'].max()))
ca=st.sidebar.slider('input Ca',float(df['Ca'].min()),float(df['Ca'].max()))
ba=st.sidebar.slider('input Ba',float(df['Ba'].min()),float(df['Ba'].max()))
fe=st.sidebar.slider('input Fe',float(df['Fe'].min()),float(df['Fe'].max()))

classifier=st.sidebar.selectbox('Classifier',('Support vector machine','LogisticRegression','RandomForestClassifier'))
if classifier=='Support vector machine':
	st.sidebar.subheader('model hyperparameters')
	c_value=st.sidebar.number_input('C (error rate)',1,100,step=1)
	kernel_input=st.sidebar.radio('kernel',('linear','rbf','poly'))
	gamma_input=st.sidebar.number_input('Gamma',1,100,step=1)
	if st.sidebar.button('Classify'):
		st.subheader('Support vector machine')
		svc_model=SVC(C=c_value,kernel=kernel_input,gamma=gamma_input)
		svc_model.fit(X_train,y_train)
		y_pred=svc_model.predict(X_test)
		accuracy=svc_model.score(X_test,y_test)
		glass_type=prediction(svc_model,ri,na,mg,al,si,K,ca,ba,fe)
		st.write('type of glass predicted is \n',glass_type)
		st.write('accuracy',accuracy.round(2))
		plot_confusion_matrix(svc_model,X_test,y_test)
		st.pyplot()

if classifier=='RandomForestClassifier':
	st.sidebar.subheader('model hyperparameters')
	n_estimators_input=st.sidebar.number_input('number of trees in forest',100,5000,step=10)
	max_depth_input=st.sidebar.number_input('max depth of the tree',1,100,step=1)
	if st.sidebar.button('Classify'):
		st.subheader('RandomForestClassifier')
		rf_clf=RandomForestClassifier(n_estimators=n_estimators_input,max_depth=max_depth_input,n_jobs=-1)
		rf_clf.fit(X_train,y_train)
		y_pred=rf_clf.predict(X_test)
		accuracy=rf_clf.score(X_test,y_test)
		glass_type=prediction(rf_clf,ri,na,mg,al,si,K,ca,ba,fe)
		st.write('type of glass predicted is \n',glass_type)
		st.write('accuracy',accuracy.round(2))
		plot_confusion_matrix(rf_clf,X_test,y_test)
		st.pyplot()
