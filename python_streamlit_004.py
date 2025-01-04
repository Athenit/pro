import streamlit as st
import pandas as pd

# Criando um dataframe fictício
dados = {
    'produto': ['maça', 'banana', 'laranja', 'manga'],
    'preço': [3.5, 2, 4.2, 5],
    'Quantidade': [10, 15, 8, 20]
}

df = pd.DataFrame(dados)

# Exibindo a tabela
st.title('Tabela de produtos')
st.dataframe(df) # Exibição interativa
st.table(df) # Exibição estática

