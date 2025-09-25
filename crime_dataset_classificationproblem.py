import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

"""# Problem Description"""

df=pd.read_csv('/content/crime_dataset_india.csv')
df.head()

df.shape,df.describe()

"""# Feature Engineering and Data Preparation"""

import datetime as dt
df['Date of Occurrence'] = pd.to_datetime(df['Date of Occurrence'])

df['Date Case Closed'] = df['Date Case Closed'].fillna(df['Date of Occurrence'])
df['Date Case Closed'] = pd.to_datetime(df['Date Case Closed'])

df['Days_to_close_cases']=df['Date Case Closed']-df['Date of Occurrence']

df['Days_to_close_cases']=df['Days_to_close_cases'].dt.days

df=df.drop(['Date Case Closed','Date of Occurrence'],axis=1)

df['Time of Occurrence'] = pd.to_datetime(df['Time of Occurrence'], format='%d-%m-%Y %H:%M')

df['Time'] = df['Time of Occurrence'].dt.time

df['Time of Occurrence_Hour'] = df['Time of Occurrence'].dt.hour

df['Time of Day'] = np.where((df['Time of Occurrence_Hour'] >= 6) & (df['Time of Occurrence_Hour'] < 12), 'Morning',
                    np.where((df['Time of Occurrence_Hour'] >= 12) & (df['Time of Occurrence_Hour'] < 16), 'Afternoon',
                    np.where((df['Time of Occurrence_Hour'] >= 16) & (df['Time of Occurrence_Hour'] < 18), 'Evening',
                    'Night')))

df=df.drop(['Time of Occurrence_Hour','Time'],axis=1)

df['Date Reported'] = pd.to_datetime(df['Date Reported'], format='%d-%m-%Y %H:%M')

df['Days_taken_to_report_cr']=df['Date Reported']-df['Time of Occurrence']

df['Days_taken_to_report_cr']=df['Days_taken_to_report_cr'].dt.days

df=df.drop(['Date Reported','Time of Occurrence'],axis=1)

df['Victim_age_group'] = np.where((df['Victim Age'] <= 12), 'Child',
                    np.where((df['Victim Age'] > 12) & (df['Victim Age'] < 18), 'Adolescent',
                    np.where((df['Victim Age'] >= 18) & (df['Victim Age'] < 25), 'Young Adult',
                    np.where((df['Victim Age'] >= 25) & (df['Victim Age'] < 40), 'Adult',
                    np.where((df['Victim Age'] >= 40) & (df['Victim Age'] <= 60), 'Middle Age', 'Old')))))

df=df.drop(['Crime Code','Victim Age'],axis=1)
df.head()

df.groupby(['Crime Domain','Crime Description'])[['Report Number']].count().reset_index().sort_values(by='Report Number',ascending=False)

"""# Extracting the Target Variable"""

serious_crimes = ['KIDNAPPING', 'HOMICIDE', 'DRUG OFFENSE']

df['Crime_Type'] = np.where(
    (df['Crime Domain'] == "Violent Crime") | (df['Crime Description'].isin(serious_crimes)),'serious violence', 'less violence')

safety_zones=df.groupby(['City'])['Report Number'].count().reset_index().sort_values(by=['Report Number'],ascending=False)

highZone_crimes= safety_zones[safety_zones['Report Number'] > 1500]['City'].tolist()
intermediateZones_crimes = safety_zones[(safety_zones['Report Number'] >= 700) & (safety_zones['Report Number'] <= 1500)]['City'].tolist()
lowZone_crimes = safety_zones[safety_zones['Report Number'] < 700]['City'].tolist()

conditions = [
    df['City'].isin(highZone_crimes),
    df['City'].isin(intermediateZones_crimes),
    df['City'].isin(lowZone_crimes)
]
choices = ['high crime zone', 'intermediate crime zone', 'low crime zone']

df['safety_zones'] = np.select(conditions, choices, default='unknown zone')

conditions = [
    ((df['Crime_Type'] == "serious violence") & (df['safety_zones'] == "high crime zone")) |
    ((df['Crime_Type'] == "less violence") & (df['safety_zones'] == "high crime zone")),
    ((df['Crime_Type'] == "less violence") & (df['safety_zones'] == "low crime zone")) |
    ((df['Crime_Type'] == "less violence") & (df['safety_zones'] == "intermediate crime zone"))|
    ((df['Crime_Type'] == "serious violence") & (df['safety_zones'] == "low crime zone"))|
    ((df['Crime_Type'] == "serious violence") & (df['safety_zones'] == "intermediate crime zone"))

]

values = ['unsafe', 'safe']

df['safety_tag'] = np.select(conditions, values, default='neutral')

df['safety_tag'].value_counts(normalize=True)*100

"""# Visualizing the Data

### Correlation Matrix with Target Feature
"""

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

df['City_Encoded'] = label_encoder.fit_transform(df['City'])

df1=df.drop(['safety_zones','Crime_Type','City_Encoded'],axis=1)

categorical_columns = ['City', 'Crime Description', 'Victim Gender', 'Weapon Used',
                       'Crime Domain', 'Case Closed', 'Time of Day', 'Victim_age_group', 'safety_tag']  # Add other categorical columns as needed

for col in categorical_columns:
    df1[col] = label_encoder.fit_transform(df1[col])

corr_matrix = df1.corr()

safety_tag_corr = corr_matrix['safety_tag'].sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of All Features with Safety Tag')
plt.show()

print(safety_tag_corr)

time_of_occ=df.groupby(['Time of Day','Victim_age_group'])[['Report Number']].count().reset_index().sort_values(by='Report Number', ascending=False)
time_of_occ.rename(columns={"Report Number": "Cases_reported"}, inplace=True)

time_of_day_cases = time_of_occ.groupby('Time of Day')['Cases_reported'].sum()
age_groups=time_of_occ.groupby('Victim_age_group')['Cases_reported'].sum()

plt.figure(figsize=(4, 4))
plt.pie(time_of_day_cases, labels=time_of_day_cases.index, autopct='%1.1f%%', colors=['#FFC898', '#A1FF8A', '#FFADA6', '#8ACBFF'])
plt.title("Occurence of Crimes by Time")
plt.show()

police_deployed=df.groupby(['City','safety_zones'])['Police Deployed'].sum().reset_index().sort_values(by='Police Deployed')

plt.figure(figsize=(10, 6))
sns.barplot(data=police_deployed, x='Police Deployed', y='City', hue='safety_zones', palette='coolwarm')

plt.title("Total Police Deployed by City and Safety Zones", fontsize=16, fontweight='bold')
plt.xlabel("Total Police Deployed", fontsize=14)
plt.ylabel("City", fontsize=14)
plt.legend(title="Safety Zones", loc='upper right')

plt.tight_layout()
plt.show()

case_closed=df.groupby(['City','Case Closed'])['Report Number'].count().reset_index().sort_values(by='Report Number',ascending=False)

plt.figure(figsize=(10, 6))

sns.barplot(data=case_closed, x='City', y='Report Number', hue='Case Closed')

plt.title("Status of cases by City", fontsize=16, fontweight='bold')
plt.xlabel("City", fontsize=14)
plt.ylabel("Count of cases", fontsize=14)
plt.legend(title="status of cases", loc='upper right')

plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()

df.groupby(['Victim_age_group'])['Report Number'].count().reset_index()

age_groups=df.groupby(['Victim_age_group'])['Report Number'].count().reset_index()

plt.figure(figsize=(4, 4))
plt.pie(age_groups['Report Number'], labels=age_groups['Victim_age_group'], autopct='%1.1f%%', colors=['#FFC898', '#A1FF8A', '#FFADA6', '#8ACBFF'])
plt.title("Total Number of Crimes Reported by Age Groups")
plt.show()

plt.figure(figsize=(5, 5))

sns.countplot(data=df, x='Crime Domain', hue='Victim Gender', palette='pastel')

plt.title('Crime Domain vs Victim Gender', fontsize=12, fontweight='bold')
plt.xlabel('Crime Domain', fontsize=10)
plt.ylabel('Count of Reports', fontsize=10)

plt.grid(True, linestyle='--', alpha=0.6)

plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()

plt.figure(figsize=(5, 5))

sns.pointplot(data=df, x='Victim_age_group', y='Days_to_close_cases', palette='viridis', markers='o', linestyles='-', dodge=True)

plt.title('Days to Close Cases by Victim Age Group (Pointplot)', fontsize=14, fontweight='bold')
plt.xlabel('Victim Age Group', fontsize=12)
plt.ylabel('Mean Days to Close Cases', fontsize=12)

plt.tight_layout()
plt.show()

"""# Data Preprocessing"""

df_num=df.select_dtypes(exclude='object')
df_num.drop(['Report Number'],axis=1,inplace=True)

df.isnull().sum()

df['Weapon Used'].value_counts()

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['City_Encoded'] = label_encoder.fit_transform(df['City'])

catcol = df[['Victim Gender', 'Case Closed', 'Time of Day', 'Victim_age_group', 'Crime_Type']]

df_categorical= pd.get_dummies(catcol, drop_first=False).astype(int)
df_categorical

df_final=pd.concat([df,df_categorical],axis=1)

df_final=df_final.drop(['City','Victim Gender','Weapon Used','Case Closed','Time of Day','Victim_age_group','Crime_Type_serious violence','Crime_Type_less violence'],axis=1)
df_final

df_final.drop(['Crime_Type','safety_zones'],axis=1,inplace=True)
df_final.isnull().sum()

X=df_final.drop('safety_tag',axis=1)
y=df_final['safety_tag']

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
X['Crime Description'] = label_encoder.fit_transform(X['Crime Description'])

"""## Feature Selection

Have used techniques like Recurrsive feature elimination and select K best
"""

import statsmodels.formula.api as smf
import statsmodels.tsa as tsa
from sklearn import metrics

df_final['safety_tag'] = df_final['safety_tag'].replace({'safe': 1, 'unsafe': 0})

df_final.info()

X=df_final.drop('safety_tag',axis=1)
y=df_final['safety_tag']

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
X['Crime Description'] = label_encoder.fit_transform(X['Crime Description'])
X['Crime Domain'] = label_encoder.fit_transform(X['Crime Domain'])
X

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
rfe = RFE(RandomForestClassifier(), n_features_to_select=10)
rfe = rfe.fit(X, y)
imp_vars_RFE = list(X.columns[rfe.support_])
imp_vars_RFE

from sklearn.feature_selection import SelectKBest, f_classif

SKB = SelectKBest(f_classif, k=10).fit(X, y )
imp_vars_SKB = list(X.columns[SKB.get_support()])
imp_vars_SKB

final_features=df_final[imp_vars_RFE]
final_features['safety_tag']=df_final['safety_tag']
final_features

"""## Train test Split

install walta to compare the best models that run in parallel to each other
"""

from wolta.data_tools import multi_split

from sklearn.model_selection import train_test_split
from wolta.data_tools import multi_split
X_train, X_test, y_trains, y_tests = multi_split(final_features, ['safety_tag'], 0.2, times=200)

X_train.shape, X_test.shape

"""# Model Development"""

from wolta.model_tools import compare_models
from sklearn.preprocessing import LabelEncoder
import pandas as pd

def preprocess_data(X):
    X_processed = pd.DataFrame(X, columns=final_features.columns[:-1])

    X_processed['Crime Description'] = X_processed['Crime Description'].astype(str)
    X_processed['Crime Domain'] = X_processed['Crime Domain'].astype(str)

    for col in X_processed.select_dtypes(include=['object']).columns:
        X_processed[col] = pd.to_numeric(X_processed[col], errors='coerce')
        X_processed[col] = X_processed[col].fillna(X_processed[col].mean())

    label_encoder = LabelEncoder()
    X_processed['Crime Description'] = label_encoder.fit_transform(X_processed['Crime Description'])
    X_processed['Crime Domain'] = label_encoder.fit_transform(X_processed['Crime Domain'])

    X_processed = X_processed.astype(float)

    return X_processed

results = compare_models(
    'clf',
    ['ada', 'cat', 'lbm', 'raf', 'dtr', 'ext', 'per', 'rdg'],
    ['acc', 'precision', 'f1'],
    preprocess_data(X_train), y_trains['safety_tag'], preprocess_data(X_test), y_tests['safety_tag'],
    get_result=True
)

from sklearn.ensemble import RandomForestClassifier

raf_model = RandomForestClassifier()

X_train_processed = preprocess_data(X_train)
X_test_processed = preprocess_data(X_test)

raf_model.fit(X_train_processed, y_trains['safety_tag'])

y_pred = raf_model.predict(X_test_processed)

from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_tests['safety_tag'], y_pred)
from sklearn import metrics
print(metrics.accuracy_score(y_tests['safety_tag'], y_pred))
print(metrics.classification_report(y_tests['safety_tag'], y_pred))

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=raf_model.classes_, yticklabels=raf_model.classes_)
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

X.loc[0,:]

X.columns

for i in X.columns:
  print(X[i].value_counts())
  print()

def predict_manual_input(model, features):
    input_data = {}
    for feature in features:
        if feature in ['Crime Description', 'Crime Domain', 'City_Encoded']:
            # Get unique values for categorical features
            unique_values = df[feature.replace('_Encoded', '')].unique()  # Get original city names if feature == 'City_Encoded'
            print(f"Options for '{feature}':")
            for i, value in enumerate(unique_values):
                print(f"{i + 1}. {value} (Encoded: {i})")  # Display encoded number with city name

            while True:
                try:
                    option = int(input(f"Enter option number for '{feature}': "))
                    if 1 <= option <= len(unique_values):
                        # Store the encoded numeric value for 'City_Encoded'
                        if feature == 'City_Encoded':
                            input_data[feature] = option - 1  # Store the encoded value (option - 1)
                        else:
                            input_data[feature] = unique_values[option - 1]
                        break
                    else:
                        print("Invalid option. Please enter a valid option number.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
        else:
            while True:
                try:
                  input_data[feature] = float(input(f"Enter value for '{feature}': "))
                  break
                except ValueError:
                  print("Invalid input. Please enter a number.")
        print()

    return "safe" if model.predict(preprocess_data(pd.DataFrame([input_data])))[0] == 1 else "unsafe"


features = final_features.columns[:-1]
prediction = predict_manual_input(raf_model, features)
print("Prediction:", prediction)