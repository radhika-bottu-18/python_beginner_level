import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import streamlit as st 

st.header('Exploratory Data Analysis Of Titanic Dataset by Radhika.')
st.write('This is an EDA on the Titanic dataset ')
def load_data():
    data = pd.read_csv(r'C:\DataScienceAndAICourse\PYTHON\Exploratory-Data-Analysis\titanic dataset.csv')
    return data

data = load_data()

st.subheader('First 5 rows of the data')
st.dataframe(data.head())

st.subheader('First 10 rows of the data')
st.dataframe(data[:10])

st.subheader('Statistical summary of the data')
st.write(data.describe())

st.subheader('Missing Values.')
missing_data = data.isnull().sum()
st.write(missing_data)

if st.checkbox('Fill the missing Age values with median') :
  data['Age'].fillna(data['Age'].median(),inplace=True)

if st.checkbox('Fill missing cabin with mode') :
  data['Cabin'].fillna(data['Cabin'].mode()[0],inplace=True)

if st.checkbox('Drop duplicates from the data') : 
   data.drop_duplicates(inplace=True)

st.subheader('Cleaned Dataset') 
st.dataframe(data[0:10])

st.subheader('Age Distribution')
fig,ax = plt.subplots()
sns.histplot(data=data['Age'],ax=ax,kde=True)
ax.set_title('Age Distribution')
plt.legend(data['Age'])

st.pyplot(fig)

st.subheader('Gender Distribution')
fig,ax = plt.subplots()
sns.countplot(x='Sex',data=data,ax=ax)
ax.set_title('Gender Distrubution')
plt.legend(data['Sex'])
st.pyplot(fig)

st.subheader('Pclass versus survived')
fig,ax = plt.subplots()
sns.countplot(x='Pclass',hue='Survived', data=data,ax=ax)
ax.set_title('Pclass versus survived')
ax.legend()
st.pyplot(fig)

st.subheader('Feature Engineering: Familiy size')
data['FaimlySize'] =  data['SibSp'] + data['Parch']
fig,ax = plt.subplots()
ax.set_title('Familiy size distrubution.')
sns.histplot(data=data['FaimlySize'],ax=ax,kde=True)
plt.legend(data['FaimlySize'])
st.pyplot(fig)

insights = """
- Females have a higher survival rate than males.
- Passengers in 1st class had the highest survival rate.
- The majority of passengers are in Pclass 3.
- Younger passengers tended to survive more often.
"""
st.subheader('Key Insights')
st.write(insights)
