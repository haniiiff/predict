import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('data.csv')

# Data Cleansing
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Sidebar
st.sidebar.header('User Input Features')

# Collect user input features
def user_input_features():
    date = st.sidebar.date_input('Date of Data')
    year = date.year  # Get the year from the input date
    users = st.sidebar.slider('Users', 0, 100000000, 5000000)
    facebook_users_percent = st.sidebar.slider('Facebook Users %', 0, 100, 50)
    population = st.sidebar.slider('Population', 0, 2000000000, 10000000)

    # Convert user input into a dataframe for model prediction
    data = {'Users': users,
            'Facebook_Users%': facebook_users_percent,
            'Year': year,  # Use the year from the input date
            'Population': population}
    features = pd.DataFrame(data, index=[0])
    return features

df_pred = user_input_features()

# Separate the features and target variables
X = df[['Users', 'Facebook_Users%', 'Year']]  # Add the 'Year' column to X
y = df['Population']

# Split the dataset into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the linear regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict the population
y_pred = regressor.predict(df_pred)

# Show the predicted population to the user
st.subheader('Predicted Population')
st.write(f'{int(y_pred[0]):,}')

# Evaluate the model
y_pred_test = regressor.predict(X_test)
r2 = r2_score(y_test, y_pred_test)
mse = mean_squared_error(y_test, y_pred_test)

# Show the model performance
st.subheader('Model Performance')
st.write(f'R-squared: {r2:.4f}')
st.write(f'Mean Squared Error: {mse:.4f}')

# Plot the model
fig, ax = plt.subplots()
ax.scatter(X_test['Users'], y_test, color='black')
ax.plot(X_test['Users'], y_pred_test, color='blue', linewidth=3)
ax.set_xlabel('Users')
ax.set_ylabel('Population')
ax.set_title('Regression Model')
st.pyplot(fig)
