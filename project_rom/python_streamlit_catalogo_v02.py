import streamlit as st
import requests
import re
import time
import urllib.parse
from io import BytesIO

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
    key="product_id_base_swap",
)

# Criação das colunas para os botões
col1, col2 = st.columns([2, 1])

with col1:
    # Botão para buscar o produto
    if st.button("Buscar Produto", key="btn_buscar"):
        pass

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
    product_data = {}

    for url in urls:
        try:
            # Faz a requisição para a URL atual
            response = requests.get(url)

            if response.status_code == 200:
                product_id = None

                # Para a URL específica de "aneis", captura o ID dentro da tag <div>
                if "aneis" in url:
                    match = re.search(
                        r'<div class="content-shelf teste" id="(\d+)">', response.text
                    )
                    if match:
                        product_id = match.group(1)
                else:
                    # Busca o número do produto na chave "shelfProductIds" ou "productId"
                    match_shelf = re.search(r'"shelfProductIds":\["(\d+)"\]', response.text)
                    match_div = re.search(r'<div class="content-shelf teste" id="(\d+)"', response.text)
                    
                    if match_shelf:
                        product_id = match_shelf.group(1)
                    elif match_div:
                        product_id = match_div.group(1)

                if product_id:
                    # Monta a URL da API
                    url_api = f"https://www.rommanel.com.br/api/catalog_system/pub/products/search?fq=productId:{product_id}"

                    # Faz a requisição para a API
                    api_response = requests.get(url_api)
                    if api_response.status_code == 200 and api_response.json():
                        data = api_response.json()[0]  # Extrai o primeiro item da lista
                        product_found = True
                        product_data = data
                        
                        # Exibe as informações formatadas
                        st.subheader("Informações do Produto")
                        st.write(f"**Nome do Produto:** {data.get('productName', 'N/A')}")
                        st.write(f"**Descrição:** {data.get('description', 'N/A')}")
                        st.write(
                            f"**Preço:** R$ {data['items'][0]['sellers'][0]['commertialOffer']['Price']:.2f}"
                        )
                        st.write(
                            f"**Link do Produto:** [Clique aqui]({data.get('link', '#')})"
                        )

                        st.subheader("Imagens")
                        image_urls = []

                        # Exibe cada imagem e adiciona botão de download
                        for idx, image in enumerate(data['items'][0]['images'], start=1):
                            image_url = image['imageUrl']
                            st.image(
                                image_url,
                                caption=image.get('imageText', f"Imagem {idx} do Produto"),
                            )
                            # Armazena URL para posterior compartilhamento
                            image_urls.append(image_url)

                            # Tenta baixar a imagem para gerar o botão de download
                            try:
                                img_response = requests.get(image_url)
                                if img_response.status_code == 200:
                                    # Converte em bytes
                                    img_bytes = img_response.content
                                    # Botão para download
                                    st.download_button(
                                        label=f"Baixar Imagem {idx}",
                                        data=img_bytes,
                                        file_name=f"produto_{product_id}_img_{idx}.jpg",
                                        mime="image/jpg"
                                    )
                            except Exception as e:
                                st.warning(f"Não foi possível gerar download para esta imagem. Erro: {e}")

                        st.subheader("Outras Especificações")
                        st.write(f"**Material:** {', '.join(data.get('Material', []))}")
                        st.write(f"**Gênero:** {', '.join(data.get('Gênero', []))}")
                        st.write(f"**Faixa Etária:** {', '.join(data.get('Faixa Etária', []))}")
                        st.write(f"**Tema:** {', '.join(data.get('Tema', []))}")
                        st.write(f"**Coleção:** {', '.join(data.get('Coleção', []))}")

                        # Link para compartilhar via WhatsApp (apenas texto + links de imagem)
                        if image_urls:
                            whatsapp_message = f"Produto: {data.get('productName', 'N/A')}\n"
                            whatsapp_message += f"Preço: R$ {data['items'][0]['sellers'][0]['commertialOffer']['Price']:.2f}\n"
                            whatsapp_message += f"Link: {data.get('link', '#')}\n\n"
                            whatsapp_message += "Imagens:\n"
                            for img_url in image_urls:
                                whatsapp_message += f"{img_url}\n"
                            
                            whatsapp_url = f"https://wa.me/?text={urllib.parse.quote(whatsapp_message)}"
                            st.markdown(f"[Compartilhar no WhatsApp]({whatsapp_url})", unsafe_allow_html=True)

                        break
            else:
                st.warning(f"Não foi possível acessar a URL: {url}")

        except Exception as e:
            st.error(f"Erro ao acessar a URL {url}: {e}")

        # Timer de 1 segundo entre as consultas
        time.sleep(1)

    if not product_found:
        st.error("Produto não encontrado nas URLs fornecidas.")
