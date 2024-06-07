import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


@st.cache_data
def load_data():
    '''
    Загрузка данных
    '''
    data = pd.read_csv('data/transformed_data.csv')
    return data

@st.cache_data
def preprocess_data(data_in):
    '''
    Предобработка данных: масштабирование числовых признаков и разделение на обучающую и тестовую выборки
    '''
    data_out = data_in.copy()
    # Масштабирование числовых признаков
    scaler = MinMaxScaler()
    numeric_cols = data_out.select_dtypes(include=np.number).columns.tolist()
    data_out[numeric_cols] = scaler.fit_transform(data_out[numeric_cols])
    
    # Разделение на обучающую и тестовую выборки
    X = data_out.drop(columns=['Accommodation cost'])  # Замените 'target_column' на название вашего столбца с целевой переменной
    y = data_out['Accommodation cost']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

# Модели
models_list = ['Linear Regression', 'KNN', 'SVR', 'Decision Tree', 'Random Forest', 'Gradient Boosting']
reg_models = {'Linear Regression': LinearRegression(), 
              'KNN': KNeighborsRegressor(), 
              'SVR': SVR(), 
              'Decision Tree': DecisionTreeRegressor(), 
              'Random Forest': RandomForestRegressor(), 
              'Gradient Boosting': GradientBoostingRegressor()}

@st.cache_data
def evaluate_model(model_name, X_train, X_test, y_train, y_test, **kwargs):
    '''
    Оценка качества модели и отображение результатов
    '''
    model = reg_models[model_name]
    model.set_params(**kwargs)  # Установка гиперпараметров модели
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Метрики качества
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Отображение результатов
    st.write(f'Модель: {model_name}')
    st.write(f'Среднеквадратичная ошибка (MSE): {mse:.2f}')
    st.write(f'Средняя абсолютная ошибка (MAE): {mae:.2f}')
    st.write(f'Коэффициент детерминации (R^2): {r2:.2f}')
# Загрузка данных
data = load_data()

# Предобработка данных
X_train, X_test, y_train, y_test = preprocess_data(data)

# Отображение боковой панели с выбором модели и гиперпараметра
st.sidebar.header('Модель машинного обучения')
model_name = st.sidebar.selectbox('Выберите модель', models_list)

hyperparameters = {}  # Словарь для хранения гиперпараметров и их значений

if model_name == 'KNN':
    n_neighbors = st.sidebar.slider('n_neighbors', min_value=1, max_value=20, value=5)
    hyperparameters['n_neighbors'] = n_neighbors
elif model_name == 'SVR':
    C = st.sidebar.slider('C', min_value=0.1, max_value=10.0, value=1.0)
    hyperparameters['C'] = C

# Отображение результатов оценки модели
st.header('Оценка качества модели')
evaluate_model(model_name, X_train, X_test, y_train, y_test, **hyperparameters)
