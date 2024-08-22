import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import streamlit as st
import pandas as pd
import plotly.express as px

# Configuração da página para usar toda a largura
st.set_page_config(layout="wide")

# CSS personalizado para adicionar o logo no canto superior direito
st.markdown("""
    <style>
        .reportview-container .main .block-container {
            max-width: 1200px;
            padding-top: 2rem;
            padding-right: 2rem;
            padding-left: 2rem;
            padding-bottom: 2rem;
        }
        .stPlotlyChart {
            width: 100%;
        }
        .dataframe {
            width: 100%;
        }
        .st-emotion-cache-1y4p8pa {
            max-width: 100%;
        }
        .st-bk {
            background-color: #f0f2f6;
        }
        /* CSS para ajustar a largura dos componentes */
        .custom-container {
            max-width: 800px;  /* Ajuste a largura conforme necessário */
            margin: 0 auto;
        }
        .custom-slider {
            width: 100%;  /* Faz com que o slider use a largura máxima disponível no contêiner */
        }
        .custom-button {
            display: block;
            margin-top: 1rem;
        }
        /* CSS para o logo no canto superior direito */
        .logo-container {
            position: absolute;
            top: 0;
            right: 0;
            padding: 10px;
        }
        .logo-container img {
            height: 100px;  /* Ajuste a altura do logo */
        }
    </style>
    <div class="logo-container">
        <img src="http://agro.guilhermegiorgi.xyz/system/logo.png" alt="Logo">
    </div>
""", unsafe_allow_html=True)

# Função para gerar dados mais realistas com produtividade média de 1800 kg/ha
def generate_realistic_data(n_samples):
    temperature = np.random.uniform(24, 43, n_samples)  # 24°C a 43°C
    humidity = np.random.uniform(10, 90, n_samples)     # 10% a 90%
    fertilizer = np.random.uniform(20, 280, n_samples)  # 20 kg/ha a 280 kg/ha
    
    # Ajuste dos coeficientes para que a média da produtividade seja 1800 kg/ha
    y = 0.5 * temperature + 1.2 * humidity + 0.8 * fertilizer + 1000 + np.random.normal(0, 100, n_samples)
    y = np.maximum(y, 0)  # Garantir que a produtividade não seja negativa
    
    X = np.column_stack((temperature, humidity, fertilizer))  # Combine as variáveis em uma matriz
    return X, y  # Retorne X e y

# Gerar dados
n_samples = 1000
X, y = generate_realistic_data(n_samples)

# Dividir os dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Criar e treinar o modelo
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(3,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)

# Avaliar o modelo
test_loss = model.evaluate(X_test_scaled, y_test)

# Função para fazer previsões
def predict_yield(temperature, humidity, fertilizer):
    new_data = np.array([[temperature, humidity, fertilizer]])
    new_data_scaled = scaler.transform(new_data)
    return model.predict(new_data_scaled)[0][0]

# Streamlit Dashboard
st.title('Dashboard de Previsão de Produtividade Agrícola - Algodão')

# Layout de uma única coluna
with st.container():
    st.markdown('<div class="custom-container">', unsafe_allow_html=True)
    st.header('Parâmetros de Entrada')
    
    # Ajuste a largura dos sliders e botões com CSS
    temperature = st.slider('Temperatura (°C)', 24.0, 43.0, 28.0, key='temperature', help="Ajuste a temperatura")
    humidity = st.slider('Umidade (%)', 10.0, 90.0, 60.0, key='humidity', help="Ajuste a umidade")
    fertilizer = st.slider('Fertilizante (kg/ha)', 20.0, 280.0, 100.0, key='fertilizer', help="Ajuste a quantidade de fertilizante")
    
    if st.button('Fazer Previsão', key='predict', help="Clique para prever a produtividade"):
        prediction = predict_yield(temperature, humidity, fertilizer)
        st.markdown(f'<p style="font-size:24px; color:#4CAF50;">Previsão de Produtividade: {prediction:.2f} kg/ha</p>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="custom-container">', unsafe_allow_html=True)
    st.header('Visualização dos Dados de Treinamento')
    
    # Dados de treinamento
    df = pd.DataFrame(X, columns=['Temperatura', 'Umidade', 'Fertilizante'])
    df['Produtividade'] = y
    
    # Tabela de dados
    st.dataframe(df, use_container_width=True)
    
    # Gráfico 3D
    fig1 = px.scatter_3d(df, x='Temperatura', y='Umidade', z='Fertilizante', color='Produtividade')
    fig1.update_layout(height=600)
    st.plotly_chart(fig1, use_container_width=True)

    # Curva de aprendizado
    st.header('Desempenho do Modelo')
    fig2 = px.line(history.history, y=['loss', 'val_loss'], labels={'index': 'Época', 'value': 'Perda'})
    fig2.update_layout(title='Curva de Aprendizado', xaxis_title='Época', yaxis_title='Perda')
    st.plotly_chart(fig2, use_container_width=True)

    # Métricas do Modelo
    st.header('Métricas do Modelo')
    st.write(f'Erro Médio Quadrático no Conjunto de Teste: {test_loss:.2f}')

    # Importância das Variáveis
    st.header('Importância das Variáveis')
    weights_first_layer = model.layers[0].get_weights()[0]
    feature_importance = np.sum(np.abs(weights_first_layer), axis=1)
    importance_df = pd.DataFrame({
        'Variável': ['Temperatura', 'Umidade', 'Fertilizante'], 
        'Importância': feature_importance
    })
    fig3 = px.bar(importance_df, x='Variável', y='Importância')
    st.plotly_chart(fig3, use_container_width=True)
