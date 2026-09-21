import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Conectores para Groq (LLM) e HuggingFace (Embeddings Gratuitos)
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# ==========================================
# CARREGA AS VARIÁVEIS DO ARQUIVO .env
# ==========================================
load_dotenv()

# ==========================================
# 1. CONFIGURAÇÃO DA PÁGINA
# ==========================================
st.set_page_config(
    page_title="Trabalho Garantido",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="auto"
)

# ==========================================
# PERSONALIZAÇÃO VISUAL - FMU TECH
# ==========================================
st.markdown("""
<style>
/* Fundo geral */
.stApp {
    background-color: #F7F4FC;
}
/* Título principal */
h1 {
    color: #7C22F5 !important;
    font-weight: 800 !important;
}
/* Texto abaixo do título */
.stApp p {
    color: #29232F;
}
/* Caixa onde o usuário digita */
div[data-testid="stChatInput"] {
    border: 2px solid #8A2BFF;
    border-radius: 14px;
    box-shadow: 0 0 10px rgba(138, 43, 255, 0.18);
}
/* Mensagens do chat */
div[data-testid="stChatMessage"] {
    background-color: #FFFFFF;
    border-left: 5px solid #8A2BFF;
    border-radius: 12px;
    padding: 12px 16px;
    margin-bottom: 12px;
    box-shadow: 0 2px 8px rgba(60, 20, 100, 0.08);
}
/* Spinner */
div[data-testid="stSpinner"] {
    color: #8A2BFF;
}
/* Cursor e detalhes do campo */
textarea {
    caret-color: #8A2BFF !important;
}
/* Links */
a {
    color: #7C22F5 !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🎓 Assistente Virtual Trabalho Garantido")

st.write(
    "Faça perguntas sobre o edital do programa e tire suas dúvidas "
    "com base nos documentos oficiais."
)

# ==========================================
# 2. CONFIGURAÇÃO DA CHAVE DA API DA GROQ
# ==========================================
groq_api_key = None

# No Streamlit Cloud ou Local, procura primeiro nos Secrets (.streamlit/secrets.toml)
try:
    if "GROQ_API_KEY" in st.secrets:
        groq_api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

# Se não encontrar nos secrets, procura a chave carregada do .env
if not groq_api_key:
    groq_api_key = os.environ.get("GROQ_API_KEY")

# Valida e injeta a credencial na memória do ambiente
if groq_api_key:
    groq_api_key = groq_api_key.strip().strip('"').strip("'")
    os.environ["GROQ_API_KEY"] = groq_api_key
else:
    st.error(
        "A chave GROQ_API_KEY não foi encontrada. "
        "Certifique-se de que ela foi configurada no arquivo .env ou nos Secrets do Streamlit."
    )
    st.stop()

# ==========================================
# 3. INICIALIZAÇÃO E CACHE DO RAG
# ==========================================
@st.cache_resource
def inicializar_rag():
    pdf_path = "EDITAL-Trabalho-Garantido-26-2.pdf"

    if not os.path.exists(pdf_path):
        return (
            None,
            None,
            "Arquivo PDF do edital não encontrado na pasta do projeto!"
        )

    loader = PyPDFLoader(pdf_path)
    documentos = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    blocos_texto = text_splitter.split_documents(documentos)

    # Embeddings locais e gratuitos do Hugging Face
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    banco_vetorial = Chroma.from_documents(
        blocos_texto,
        embeddings
    )

    retriever = banco_vetorial.as_retriever(
        search_kwargs={"k": 3}
    )

    # Modelo atualizado e em produção na API da Groq
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0
    )

    return retriever, llm, None

# ==========================================
# 4. CARREGAMENTO DO RAG
# ==========================================
with st.spinner("Processando o edital e preparando o assistente..."):
    retriever, llm, erro = inicializar_rag()

if erro:
    st.error(erro)
    st.stop()

# ==========================================
# 5. HISTÓRICO DAS MENSAGENS
# ==========================================
if "mensagens" not in st.session_state:
    st.session_state.mensagens = []

for message in st.session_state.mensagens:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==========================================
# 6. INTERFACE DO CHAT
# ==========================================
if pergunta := st.chat_input("Digite sua dúvida sobre o edital..."):

    st.session_state.mensagens.append(
        {
            "role": "user",
            "content": pergunta
        }
    )

    with st.chat_message("user"):
        st.markdown(pergunta)

    with st.chat_message("assistant"):
        with st.spinner("Pesquisando no edital..."):
            try:
                # 1. RECUPERA OS TRECHOS DO EDITAL
                documentos_relacionados = retriever.invoke(pergunta)

                contexto = "\n\n".join(
                    [doc.page_content for doc in documentos_relacionados]
                )

                # 2. MONTA O PROMPT
                prompt_completo = (
                    "Você é um assistente virtual especializado no Edital do Programa Trabalho Garantido da FMU.\n\n"
                    "Use estritamente os trechos do edital fornecidos abaixo para responder à pergunta do usuário.\n"
                    "Se não souber a resposta ou se ela não estiver no texto, diga honestamente que não encontrou.\n\n"
                    f"Contexto do edital:\n{contexto}\n\n"
                    f"Pergunta do usuário: {pergunta}"
                )

                # 3. ENVIA PARA A GROQ
                resposta_llm = llm.invoke(prompt_completo)

                # 4. EXTRAI SOMENTE O TEXTO
                if isinstance(resposta_llm.content, list):
                    texto_resposta = "".join(
                        bloco.get("text", "")
                        for bloco in resposta_llm.content
                        if isinstance(bloco, dict) and bloco.get("type") == "text"
                    )
                else:
                    texto_resposta = resposta_llm.content

                # 5. EXIBE A RESPOSTA
                st.markdown(texto_resposta)

                st.session_state.mensagens.append(
                    {
                        "role": "assistant",
                        "content": texto_resposta
                    }
                )

            except Exception as e:
                erro_str = str(e)
                st.error("⚠️ A API da Groq retornou um erro.")
                st.code(erro_str, language="text")
