import streamlit as st
import requests
import json

# Título da aplicação
st.title("Consulta de Produtos")

# Campo de entrada para o ID do produto
product_id = st.text_input("Digite o ID do produto (productId):", "")

# Botão para buscar as informações
if st.button("Buscar Produto"):
    if product_id:
        # URL da API
        url = f"https://www.rommanel.com.br/api/catalog_system/pub/products/search?fq=productId:{product_id}"
#
        # Faz a requisição
        response = requests.get(url)

        # Verifica se o produto foi encontrado
        if response.status_code == 200 and response.json():
            data = response.json()[0]  # Extrai o primeiro item da lista

            # Exibe as informações formatadas
            st.subheader("Informações do Produto")
            st.write(f"**Nome do Produto:** {data.get('productName', 'N/A')}")
            st.write(f"**Descrição:** {data.get('description', 'N/A')}")
            st.write(f"**Preço:** R$ {data['items'][0]['sellers'][0]['commertialOffer']['Price']:.2f}")
            st.write(f"**Link do Produto:** [Clique aqui]({data.get('link', '#')})")

            st.subheader("Imagens")
            for image in data['items'][0]['images']:
                st.image(image['imageUrl'], caption=image.get('imageText', 'Imagem do Produto'))

            st.subheader("Outras Especificações")
            st.write(f"**Material:** {', '.join(data.get('Material', []))}")
            st.write(f"**Gênero:** {', '.join(data.get('Gênero', []))}")
            st.write(f"**Faixa Etária:** {', '.join(data.get('Faixa Etária', []))}")
            st.write(f"**Tema:** {', '.join(data.get('Tema', []))}")
            st.write(f"**Coleção:** {', '.join(data.get('Coleção', []))}")
        else:
            st.error("Produto não encontrado. Verifique o ID e tente novamente.")
    else:
        st.warning("Por favor, insira o ID do produto.")

