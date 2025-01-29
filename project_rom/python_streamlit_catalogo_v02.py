import streamlit as st
import requests
import re
import json

# Título da aplicação
st.title("Consulta de Produtos")

# Campo de entrada para o ID do produto
product_id_base = st.text_input("Digite o ID base do produto (Exemplo: 820252):", "")

# Botão para buscar as informações
if st.button("Buscar Produto"):
    if product_id_base:
        try:
            # Gera a URL inicial com o ID base fornecido pelo usuário
            url_inicial = f"https://www.rommanel.com.br/joias/{product_id_base}"

            # Faz a requisição para a URL inicial
            response = requests.get(url_inicial)

            if response.status_code == 200:
                # Busca o número do produto na chave "shelfProductIds"
                match = re.search(r'"shelfProductIds":\["(\d+)"\]', response.text)
                if match:
                    product_id = match.group(1)  # Captura o valor completo (Exemplo: "82025200")

                    # Monta a URL da API
                    url_api = f"https://www.rommanel.com.br/api/catalog_system/pub/products/search?fq=productId:{product_id}"

                    # Faz a requisição para a API
                    api_response = requests.get(url_api)
                    if api_response.status_code == 200 and api_response.json():
                        data = api_response.json()[0]  # Extrai o primeiro item da lista

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
                        st.error("Não foi possível obter as informações do produto na API.")
                else:
                    st.error("Não foi possível encontrar o número do produto na URL inicial.")
            else:
                st.error("Erro ao acessar a URL inicial. Verifique o ID base do produto.")
        except Exception as e:
            st.error(f"Ocorreu um erro: {e}")
    else:
        st.warning("Por favor, insira o ID base do produto.")
