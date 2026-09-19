import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# ==========================================
# 1. CONFIGURAÇÃO DA PÁGINA E CREDENCIAIS
# ==========================================
st.set_page_config(
    page_title="Trabalho Garantido", 
    page_icon="🎓", 
    layout="centered",
    initial_sidebar_state="auto"
)

# Truque para forçar o navegador a entender que a página é em Português e desativar o tradutor automático
st.markdown(
    """
    <script>
        var doc = window.parent.document;
        doc.documentElement.lang = 'pt-BR';
    </script>
    """,
    unsafe_allow_html=True
)


st.title("🎓 Assistente Virtual Trabalho Garantido")
st.write(
    "Faça perguntas sobre o edital do programa e tire suas dúvidas "
    "com base nos documentos oficiais."
)

# Configurar a chave de API
# Verifica primeiro nos Secrets da nuvem e depois
# nas variáveis de ambiente locais
google_api_key = None

try:
    if "GOOGLE_API_KEY" in st.secrets:
        google_api_key = st.secrets["GOOGLE_API_KEY"]
except Exception:
    pass

if not google_api_key:
    google_api_key = os.environ.get("GOOGLE_API_KEY")

if google_api_key:
    os.environ["GOOGLE_API_KEY"] = google_api_key
else:
    st.error(
        "A chave GOOGLE_API_KEY não foi encontrada. "
        "Certifique-se de que a definiu no terminal "
        "ou nos Secrets do Streamlit."
    )
    st.stop()


# ==========================================
# 2. INICIALIZAÇÃO E CACHE DO RAG
# ==========================================
@st.cache_resource
def inicializar_rag():

    pdf_path = "EDITAL-Trabalho-Garantido-26-2.pdf"

    if not os.path.exists(pdf_path):
        return None, None, "Arquivo PDF do edital não encontrado na pasta do projeto!"

    # A. Carregamento e Fatiamento do PDF
    loader = PyPDFLoader(pdf_path)
    documentos = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    blocos_texto = text_splitter.split_documents(documentos)

    # B. Criação da Base Vetorial
    # Embeddings + ChromaDB
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2"
    )

    banco_vetorial = Chroma.from_documents(
        blocos_texto,
        embeddings
    )

    retriever = banco_vetorial.as_retriever(
        search_kwargs={"k": 3}
    )

    # C. Instancia o LLM do Gemini
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        temperature=0
    )

    return retriever, llm, None


# Executa o carregamento
with st.spinner("Processando o edital e preparando o assistente..."):
    retriever, llm, erro = inicializar_rag()

if erro:
    st.error(erro)
    st.stop()


# ==========================================
# 3. GERENCIAMENTO DO HISTÓRICO DE MENSAGENS
# ==========================================
if "mensagens" not in st.session_state:
    st.session_state.mensagens = []

# Exibe o histórico na tela
for mensagem in st.session_state.mensagens:
    with st.chat_message(mensagem["role"]):
        st.markdown(mensagem["content"])


# ==========================================
# 4. INTERFACE INTERATIVA DO CHAT COM TRATAMENTO DE ERRO
# ==========================================
if pergunta := st.chat_input("Digite sua dúvida sobre o edital..."):
    
    # Salva e exibe a pergunta do usuário
    st.session_state.mensagens.append({"role": "user", "content": pergunta})
    with st.chat_message("user"):
        st.markdown(pergunta)

    # Processa a resposta buscando os trechos e consultando o Gemini de forma segura
    with st.chat_message("assistant"):
        with st.spinner("Pesquisando no edital..."):
            try:
                # 1. Busca os documentos relevantes no ChromaDB
                documentos_relacionados = retriever.invoke(pergunta)
                contexto = "\n\n".join([doc.page_content for doc in documentos_relacionados])
                
                # 2. Monta o prompt com o contexto e restrições anti-alucinação
                prompt_completo = (
                    "Você é um assistente virtual especializado no Edital do Programa Trabalho Garantido da FMU.\n"
                    "Use estritamente os trechos do edital fornecidos abaixo para responder à pergunta do usuário.\n"
                    "Se não souber a resposta ou se ela não estiver no texto, diga honestamente que não encontrou.\n\n"
                    f"Contexto do edital:\n{contexto}\n\n"
                    f"Pergunta do usuário: {pergunta}"
                )
                
                # 3. Invoca o modelo diretamente
                resposta_llm = llm.invoke(prompt_completo)
                
                # Extrai o texto limpo da resposta
                if hasattr(resposta_llm, 'content'):
                    texto_resposta = resposta_llm.content
                else:
                    texto_resposta = str(resposta_llm)
                
                st.markdown(texto_resposta)
                
                # Salva a resposta bem-sucedida no histórico
                st.session_state.mensagens.append({"role": "assistant", "content": texto_resposta})

            except Exception as e:
                # Tratamento amigável caso ocorra o erro 429 de cota excedida ou falha de rede
                erro_str = str(e)
                if "429" in erro_str or "RESOURCE_EXHAUSTED" in erro_str:
                    texto_resposta = (
                        "⚠️ **Ops! O limite de requisições gratuitas foi atingido temporariamente.**\n\n"
                        "Como este assistente está rodando na camada gratuita da API do Gemini, "
                        "ultrapassamos o número de consultas permitidas em um curto período. "
                        "Por favor, aguarde alguns segundos ou tente novamente mais tarde!"
                    )
                else:
                    texto_resposta = f"⚠️ Ocorreu um erro inesperado ao processar a sua pergunta: `{erro_str}`"
                
                st.error(texto_resposta)
                # Opcional: salva a mensagem de aviso no histórico para manter o fluxo coerente
                st.session_state.mensagens.append({"role": "assistant", "content": texto_resposta})