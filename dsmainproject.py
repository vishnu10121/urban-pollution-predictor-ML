#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[3]:


traffic = pd.read_csv("traffic.csv")
pollution = pd.read_csv("delhi pollution air.csv")
weather = pd.read_csv("weather.csv")


# In[6]:


traffic.head()


# In[7]:


pollution.head()


# In[8]:


weather.head()


# In[9]:


traffic.shape


# In[10]:


pollution.shape


# In[11]:


weather.shape


# In[12]:


traffic.info()


# In[13]:


pollution.info()


# In[14]:


weather.info()


# In[15]:


traffic.describe()


# In[16]:


pollution.describe()


# In[17]:


weather.describe()


# In[18]:


traffic.isnull().sum()


# In[19]:


pollution.isnull().sum()


# In[20]:


weather.isnull().sum()


# In[21]:


pollution = pollution[['Datetime','PM2.5','PM10','NO2','CO','SO2','AQI']]


# In[22]:


pollution = pollution.fillna(pollution.median(numeric_only=True))


# In[23]:


pollution.isnull().sum()


# In[28]:


weather.columns = ['datetime','temperature','humidity','wind_speed','pressure','visibility']


# In[29]:


weather.columns


# In[30]:


weather = weather[['datetime_utc','_tempm','_hum','_wspdm','_pressurem','_vism']]


# In[31]:


weather.columns = weather.columns.str.strip()


# In[32]:


weather = weather[['datetime_utc','_tempm','_hum','_wspdm','_pressurem','_vism']]


# In[33]:


weather.columns = ['datetime','temperature','humidity','wind_speed','pressure','visibility']


# In[34]:


weather.head()


# In[35]:


weather.isnull().sum()


# In[36]:


weather = weather.fillna(weather.median(numeric_only=True))


# In[37]:


pollution = pollution.fillna(pollution.median(numeric_only=True))


# In[39]:


pollution['Datetime'] = pd.to_datetime(pollution['Datetime'], errors='coerce')
weather['datetime'] = pd.to_datetime(weather['datetime'], errors='coerce')


# In[40]:


pollution = pollution.dropna(subset=['Datetime'])
weather = weather.dropna(subset=['datetime'])


# In[41]:


pollution['Datetime'].dtype
weather['datetime'].dtype


# In[42]:


pollution = pollution.sort_values('Datetime')
weather = weather.sort_values('datetime')


# In[43]:


data = pd.merge(pollution, weather, left_on='Datetime', right_on='datetime', how='inner')


# In[45]:


print("Traffic Shape:", traffic.shape)
print("Pollution Shape:", pollution.shape)
print("Weather Shape:", weather.shape)


# In[46]:


print("Traffic Missing Values")
print(traffic.isnull().sum())

print("\nPollution Missing Values")
print(pollution.isnull().sum())

print("\nWeather Missing Values")
print(weather.isnull().sum())


# In[47]:


pollution.describe()


# In[48]:


weather.describe()


# In[50]:


traffic.describe()


# In[51]:


data['hour'] = data['Datetime'].dt.hour
data['day'] = data['Datetime'].dt.day
data['month'] = data['Datetime'].dt.month
data['year'] = data['Datetime'].dt.year


# In[52]:


def pollution_level(pm):
    if pm <= 50:
        return "Good"
    elif pm <= 100:
        return "Moderate"
    elif pm <= 150:
        return "Unhealthy"
    else:
        return "Very Unhealthy"

data['pollution_category'] = data['PM2.5'].apply(pollution_level)


# In[53]:


X = data[['temperature','humidity','wind_speed','pressure','visibility','hour','month']]
y = data['PM2.5']


# In[54]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# In[55]:


from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)


# In[57]:


y_pred = model.predict(X_test)


# In[58]:


y_pred


# In[59]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))


# In[61]:


rf = RandomForestRegressor(n_estimators=100, random_state=42)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)


# In[62]:


print("MAE:", mean_absolute_error(y_test, rf_pred))
print("MSE:", mean_squared_error(y_test, rf_pred))
print("R2 Score:", r2_score(y_test, rf_pred))


# In[64]:


gb = GradientBoostingRegressor()

gb.fit(X_train, y_train)

gb_pred = gb.predict(X_test)

print("R2 Score:", r2_score(y_test, gb_pred))


# In[67]:


import matplotlib.pyplot as plt

importance = rf.feature_importances_

features = pd.Series(importance, index=X.columns)

features.sort_values().plot(kind="barh")
plt.title("Feature Importance")
plt.show()


# In[68]:


from sklearn.ensemble import GradientBoostingRegressor

gb = GradientBoostingRegressor()

gb.fit(X_train, y_train)

gb_pred = gb.predict(X_test)

print("MAE:", mean_absolute_error(y_test, gb_pred))
print("R2 Score:", r2_score(y_test, gb_pred))


# In[69]:


from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(random_state=42)

dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

print("MAE:", mean_absolute_error(y_test, dt_pred))
print("R2 Score:", r2_score(y_test, dt_pred))


# In[70]:


from sklearn.svm import SVR

svr = SVR()

svr.fit(X_train, y_train)

svr_pred = svr.predict(X_test)

print("MAE:", mean_absolute_error(y_test, svr_pred))
print("R2 Score:", r2_score(y_test, svr_pred))


# In[71]:


from sklearn.ensemble import ExtraTreesRegressor

et = ExtraTreesRegressor(n_estimators=200, random_state=42)

et.fit(X_train, y_train)

et_pred = et.predict(X_test)

print("MAE:", mean_absolute_error(y_test, et_pred))
print("R2 Score:", r2_score(y_test, et_pred))


# In[72]:


from sklearn.ensemble import AdaBoostRegressor

ada = AdaBoostRegressor(n_estimators=100, random_state=42)

ada.fit(X_train, y_train)

ada_pred = ada.predict(X_test)

print("MAE:", mean_absolute_error(y_test, ada_pred))
print("R2 Score:", r2_score(y_test, ada_pred))


# In[73]:


importance = rf.feature_importances_

features = pd.Series(importance, index=X.columns)

features.sort_values().plot(kind="barh")

plt.title("Feature Importance")
plt.show()


# In[74]:


from sklearn.model_selection import cross_val_score

scores = cross_val_score(rf, X, y, cv=5, scoring='r2')

print("Cross Validation R2 Scores:", scores)
print("Average R2:", scores.mean())


# In[75]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, None]
}

grid = GridSearchCV(RandomForestRegressor(), param_grid, cv=3)

grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)


# In[76]:


rf_final = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    random_state=42
)

rf_final.fit(X_train, y_train)

final_pred = rf_final.predict(X_test)


# In[77]:


print("MAE:", mean_absolute_error(y_test, final_pred))
print("MSE:", mean_squared_error(y_test, final_pred))
print("R2 Score:", r2_score(y_test, final_pred))


# In[78]:


importance = rf_final.feature_importances_

pd.Series(importance, index=X.columns).sort_values().plot(kind='barh')

plt.title("Feature Importance")
plt.show()


# In[79]:


models = {
    "Linear Regression": 0.006,
    "Gradient Boosting": 0.049,
    "Random Forest": 0.123,
    "Decision Tree": 0.124,
    "Extra Trees": 0.124,
    "SVR": -0.10,
    "AdaBoost": -0.15
}

pd.Series(models).plot(kind="bar")

plt.title("Model Comparison (R2 Score)")
plt.show()


# In[82]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

traffic['traffic_density_level'] = le.fit_transform(traffic['traffic_density_level'])
traffic['road_type'] = le.fit_transform(traffic['road_type'])
traffic['time_of_day'] = le.fit_transform(traffic['time_of_day'])


# In[86]:


X = data[
[
'temperature',
'humidity',
'wind_speed',
'pressure',
'visibility',
'hour',
'month'
]
]

y = data['PM2.5']


# In[87]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)


# In[88]:


from sklearn.ensemble import RandomForestRegressor

rf_final = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    random_state=42
)

rf_final.fit(X_train, y_train)

final_pred = rf_final.predict(X_test)


# In[89]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("MAE:", mean_absolute_error(y_test, final_pred))
print("MSE:", mean_squared_error(y_test, final_pred))
print("R2 Score:", r2_score(y_test, final_pred))


# In[90]:


import pandas as pd
import matplotlib.pyplot as plt

importance = rf_final.feature_importances_

pd.Series(importance, index=X.columns).sort_values().plot(kind='barh')

plt.title("Feature Importance")
plt.show()


# In[93]:


model_scores = {
    "Linear Regression": 0.006,
    "Random Forest": 0.123,
    "Gradient Boosting": 0.049,
    "Decision Tree": 0.124,
    "Extra Trees": 0.124,
    "SVR": -0.10,
    "AdaBoost": -0.15
}

pd.Series(model_scores).plot(kind='bar')

plt.title("Model Comparison (R2 Score)")
plt.ylabel("R2 Score")
plt.show()


# In[94]:


import seaborn as sns

errors = y_test - final_pred

sns.histplot(errors, kde=True)

plt.title("Prediction Error Distribution")
plt.show()


# In[95]:


plt.scatter(y_test, final_pred)

plt.xlabel("Actual PM2.5")
plt.ylabel("Predicted PM2.5")

plt.title("Actual vs Predicted Pollution")

plt.show()


# In[96]:


feature_importance = pd.Series(
rf_final.feature_importances_,
index=X.columns
).sort_values(ascending=False)

print(feature_importance)


# In[100]:


model_results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest",
        "Gradient Boosting",
        "Decision Tree",
        "Extra Trees",
        "SVR",
        "AdaBoost"
    ],
    "R2 Score": [
        r2_score(y_test, y_pred),
        r2_score(y_test, rf_pred),
        r2_score(y_test, gb_pred),
        r2_score(y_test, dt_pred),
        r2_score(y_test, et_pred),
        r2_score(y_test, svr_pred),
        r2_score(y_test, ada_pred)
    ]
})

print(model_results)


# In[101]:


model_results.plot(
    x="Model",
    y="R2 Score",
    kind="bar",
    legend=False,
    figsize=(10,5)
)

plt.title("Model Performance Comparison")
plt.show()


# In[98]:


import seaborn as sns

plt.figure(figsize=(10,6))
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="coolwarm")

plt.title("Feature Correlation Heatmap")
plt.show()


# In[102]:


import seaborn as sns

errors = y_test - final_pred

sns.histplot(errors, kde=True)

plt.title("Prediction Error Distribution")

plt.show()


# In[ ]:


1. Wind speed has the strongest effect on PM2.5 dispersion.
2. Humidity and temperature also influence pollution levels.
3. Tree-based models (Random Forest, Decision Tree, Extra Trees) performed best.
4. Weather variables explain part of the pollution variation, but other urban factors are required for more accurate prediction.

