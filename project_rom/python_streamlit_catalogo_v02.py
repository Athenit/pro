import streamlit as st
import requests
import re
import time

# Verifica se a variável não existe ainda e a inicializa
if "product_id_base_swap" not in st.session_state:
    st.session_state.product_id_base_swap = 0


# Função para limpar o campo de entrada
def limpar_input():
    st.session_state.product_id_base_swap = 0


# Título da aplicação
st.title("Consulta de Produtos")


# Campo de entrada numérica para o ID do produto
product_id_base = st.number_input(
    "Digite o ID base do produto (Exemplo: 820252):", 
    min_value=0, 
    step=1, 
    format="%d", 
    key="product_id_base_swap"
)


# Criação das colunas para os botões
col1, col2 = st.columns([2, 1])

with col1:
    # Botão swap para buscar o produto
    if st.button("Buscar Produto", key="btn_buscar"):
        pass
        # st.session_state.product_id_base_swap = product_id_base

with col2:
    # Botão para limpar o campo
    if st.button("Limpar", on_click=limpar_input, key="btn_limpar"):
        pass

# Lista de URLs para buscar o produto
urls = [
    f"https://www.rommanel.com.br/joias/{int(product_id_base)}",
    f"https://www.rommanel.com.br/joias/aneis/{int(product_id_base)}",
    f"https://www.rommanel.com.br/relogios/{int(product_id_base)}",
]

# Botão para buscar as informações
if st.session_state.product_id_base_swap and product_id_base != 0:
    product_found = False

    for url in urls:
        try:
            # Faz a requisição para a URL atual
            response = requests.get(url)

            if response.status_code == 200:
                # Para a URL específica de "aneis", captura o ID dentro da tag <div>
                if "aneis" in url:
                    match = re.search(r'<div class="content-shelf teste" id="(\d+)">', response.text)
                else:
                    # Busca o número do produto na chave "shelfProductIds" para outras URLs
                    match = re.search(r'"shelfProductIds":\["(\d+)"\]', response.text)
                    if not match:
                        match = re.search(r'"productId":\["(\d+)"\]', response.text)

                if match:
                    product_id = match.group(1)  # Captura o ID completo (ex.: "27000100")

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
            else:
                st.warning(f"Não foi possível acessar a URL: {url}")

        except Exception as e:
            st.error(f"Erro ao acessar a URL {url}: {e}")

        # Timer de 1 segundo entre as consultas
        time.sleep(1)

    if not product_found:
        st.error("Produto não encontrado nas URLs fornecidas.")
