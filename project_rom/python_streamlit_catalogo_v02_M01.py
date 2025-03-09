import streamlit as st
import requests
import re
import time
import logging
from requests.exceptions import RequestException
import streamlit.components.v1 as components

# Configuração do logging para auxiliar na depuração
logging.basicConfig(level=logging.INFO)

# Inicializa a variável de sessão se não existir
if "product_id_base_swap" not in st.session_state:
    st.session_state.product_id_base_swap = 0

def limpar_input():
    """
    Função para limpar o campo de entrada, resetando o estado da sessão.
    """
    st.session_state.product_id_base_swap = 0

def buscar_produto(product_id_base, session):
    """
    Busca o produto nas URLs fornecidas e retorna os dados do produto caso encontrado.
    
    Args:
        product_id_base (int): ID base do produto informado.
        session (requests.Session): Sessão para realizar as requisições HTTP.
        
    Returns:
        dict: Dados do produto se encontrado; caso contrário, None.
    """
    # Lista de URLs a serem consultadas
    urls = [
        f"https://www.rommanel.com.br/joias/{int(product_id_base)}",
        f"https://www.rommanel.com.br/joias/aneis/{int(product_id_base)}",
        f"https://www.rommanel.com.br/relogios/{int(product_id_base)}",
    ]
    
    for url in urls:
        try:
            # Exibe um spinner enquanto realiza a requisição
            with st.spinner(f"Buscando produto na URL: {url}"):
                response = session.get(url, timeout=5)
            
            # Se a URL foi acessada com sucesso
            if response.status_code == 200:
                # Se for a URL de "aneis", captura o ID do produto na tag <div>
                if "aneis" in url:
                    match = re.search(r'<div class="content-shelf teste" id="(\d+)">', response.text)
                else:
                    # Para as demais URLs, captura o ID do produto na chave "shelfProductIds"
                    match = re.search(r'"shelfProductIds":\["(\d+)"\]', response.text)
                
                if match:
                    product_id = match.group(1)
                    # Monta a URL da API para obter os dados do produto
                    url_api = f"https://www.rommanel.com.br/api/catalog_system/pub/products/search?fq=productId:{product_id}"
                    api_response = session.get(url_api, timeout=5)
                    if api_response.status_code == 200:
                        json_data = api_response.json()
                        if json_data:
                            return json_data[0]  # Retorna o primeiro item encontrado
            else:
                st.warning(f"Não foi possível acessar a URL: {url} (Status: {response.status_code})")
        
        except RequestException as e:
            st.error(f"Erro ao acessar a URL {url}: {e}")
            logging.error(f"Erro na URL {url}: {e}")
        
        # Aguarda 1 segundo antes de tentar a próxima URL
        time.sleep(1)
    
    return None

def exibir_produto(data):
    """
    Exibe as informações do produto na interface do usuário.
    
    Args:
        data (dict): Dados do produto.
    """
    st.subheader("Informações do Produto")
    st.write(f"**Nome do Produto:** {data.get('productName', 'N/A')}")
    st.write(f"**Descrição:** {data.get('description', 'N/A')}")
    
    # Tratamento para exibir o preço de forma segura
    try:
        price = data['items'][0]['sellers'][0]['commertialOffer']['Price']
        st.write(f"**Preço:** R$ {price:.2f}")
    except (IndexError, KeyError, TypeError):
        st.write("**Preço:** N/A")
    
    st.write(f"**Link do Produto:** [Clique aqui]({data.get('link', '#')})")
    
    st.subheader("Imagens")
    try:
        for image in data['items'][0]['images']:
            st.image(image['imageUrl'], caption=image.get('imageText', 'Imagem do Produto'))
    except (IndexError, KeyError, TypeError):
        st.write("Imagens não disponíveis.")
    
    st.subheader("Outras Especificações")
    # Função auxiliar para exibir campos que são listas
    def exibir_lista(campo):
        valor = data.get(campo, [])
        return ', '.join(valor) if isinstance(valor, list) else str(valor)
    
    st.write(f"**Material:** {exibir_lista('Material')}")
    st.write(f"**Gênero:** {exibir_lista('Gênero')}")
    st.write(f"**Faixa Etária:** {exibir_lista('Faixa Etária')}")
    st.write(f"**Tema:** {exibir_lista('Tema')}")
    st.write(f"**Coleção:** {exibir_lista('Coleção')}")

def obter_codigo():
    """
    Tenta ler o código atual para disponibilizar o download.
    """
    try:
        with open(__file__, "r", encoding="utf-8") as f:
            codigo = f.read()
        return codigo
    except Exception as e:
        logging.error(f"Erro ao ler o arquivo para download: {e}")
        return None

def exibir_jogo_da_velha():
    """
    Exibe um jogo da velha implementado em p5.js usando um componente HTML.
    """
    # Código HTML com o jogo da velha em p5.js
    tic_tac_toe_html = """
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
      <meta charset="UTF-8">
      <title>Jogo da Velha</title>
      <script src="https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.4.0/p5.js"></script>
      <style>
        body { display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }
      </style>
    </head>
    <body>
      <script>
        let board;
        let w;
        let h;
        let currentPlayer = 'X';
        let available = [];

        function setup() {
          createCanvas(400, 400);
          w = width / 3;
          h = height / 3;
          board = [
            ['', '', ''],
            ['', '', ''],
            ['', '', '']
          ];
          for (let j = 0; j < 3; j++) {
            for (let i = 0; i < 3; i++) {
              available.push({i, j});
            }
          }
        }

        function equals3(a, b, c) {
          return a == b && b == c && a != '';
        }

        function checkWinner() {
          let winner = null;

          // Checa linhas e colunas
          for (let i = 0; i < 3; i++) {
            if (equals3(board[i][0], board[i][1], board[i][2])) {
              winner = board[i][0];
            }
            if (equals3(board[0][i], board[1][i], board[2][i])) {
              winner = board[0][i];
            }
          }
          
          // Checa diagonais
          if (equals3(board[0][0], board[1][1], board[2][2])) {
            winner = board[0][0];
          }
          if (equals3(board[2][0], board[1][1], board[0][2])) {
            winner = board[2][0];
          }

          if (winner != null) {
            noLoop();
            setTimeout(() => { alert("O vencedor é: " + winner); }, 10);
          } else if (available.length == 0) {
            noLoop();
            setTimeout(() => { alert("Empate!"); }, 10);
          }
        }

        function mousePressed() {
          let i = floor(mouseX / w);
          let j = floor(mouseY / h);
          // Verifica se a posição está vazia
          if (board[j][i] == '') {
            board[j][i] = currentPlayer;
            // Remove a posição da lista de disponíveis
            for (let k = available.length - 1; k >= 0; k--) {
              if (available[k].i === i && available[k].j === j) {
                available.splice(k, 1);
              }
            }
            currentPlayer = (currentPlayer == 'X') ? 'O' : 'X';
          }
        }

        function draw() {
          background(255);
          strokeWeight(4);

          // Desenha a grade
          line(w, 0, w, height);
          line(w * 2, 0, w * 2, height);
          line(0, h, width, h);
          line(0, h * 2, width, h * 2);

          // Desenha os X e O
          textSize(32);
          for (let j = 0; j < 3; j++) {
            for (let i = 0; i < 3; i++) {
              let x = w * i + w / 2;
              let y = h * j + h / 2;
              let spot = board[j][i];
              textAlign(CENTER, CENTER);
              text(spot, x, y);
            }
          }
          checkWinner();
        }
      </script>
    </body>
    </html>
    """
    components.html(tic_tac_toe_html, height=500)

def main():
    """
    Função principal que organiza a interface, a lógica de busca e a opção de distrair com um jogo da velha.
    """
    st.title("Consulta de Produtos")

    # Cria um separador para a funcionalidade de consulta de produtos
    st.header("Consulta de Produtos")
    
    # Campo de entrada para o ID base do produto
    product_id_base = st.number_input(
        "Digite o ID base do produto (Exemplo: 820252):", 
        min_value=0, 
        step=1, 
        format="%d", 
        key="product_id_base_swap"
    )
    
    # Criação de colunas para os botões
    col1, col2 = st.columns([2, 1])
    
    with col1:
        buscar = st.button("Buscar Produto", key="btn_buscar")
    with col2:
        st.button("Limpar", on_click=limpar_input, key="btn_limpar")
    
    # Se o botão de buscar for clicado e o ID for válido, inicia a busca
    if buscar and product_id_base != 0:
        # Cria uma sessão para melhorar o desempenho das requisições
        with requests.Session() as session:
            produto = buscar_produto(product_id_base, session)
            if produto:
                exibir_produto(produto)
            else:
                st.error("Produto não encontrado nas URLs fornecidas.")
    
    # Disponibiliza o código para download, se possível
    codigo = obter_codigo()
    if codigo:
        st.download_button(
            label="Download do Código",
            data=codigo,
            file_name="python_streamlit_catalogo_v02_melhorado.py",
            mime="text/x-python"
        )
    else:
        st.info("Opção de download do código não disponível neste ambiente.")
    
    st.markdown("---")
    st.header("Jogo da Velha")
    # Checkbox para exibir o jogo da velha
    if st.checkbox("Jogar Jogo da Velha"):
        exibir_jogo_da_velha()

if __name__ == "__main__":
    main()
