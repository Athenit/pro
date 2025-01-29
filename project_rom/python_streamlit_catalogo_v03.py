import streamlit as st
import requests
import re

# Título da aplicação
st.title("Consulta de Produtos - Rommanel")

# Campo de entrada numérica para o ID do produto
product_id_base = st.number_input(
    "Digite o ID base do produto (Exemplo: 820252):", 
    min_value=0, 
    step=1, 
    format="%d", 
    value=0
)

# Exibe um aviso enquanto o usuário não digitar um valor válido
if product_id_base == 0:
    st.info("Digite o ID do produto para continuar.")

# Lista de URLs para buscar o produto
urls = [
    f"https://www.rommanel.com.br/joias/{int(product_id_base)}",
    f"https://www.rommanel.com.br/joias/aneis/{int(product_id_base)}",
    f"https://www.rommanel.com.br/relogios/{int(product_id_base)}",
]

# Botão para buscar as informações
if st.button("Buscar Produto") and product_id_base != 0:
    product_found = False

    for url in urls:
        try:
            # Faz a requisição para a URL atual
            response = requests.get(url)

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
                        product_found = True

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
                        break
        except Exception as e:
            st.error(f"Erro ao acessar a URL {url}: {e}")

    if not product_found:
        st.error("Produto não encontrado nas URLs fornecidas.")
