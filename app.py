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
# PERSONALIZAÇÃO VISUAL - PALETA TECH DARK COM CAMPO CLARO
# ==========================================
st.markdown("""
<style>
/* Fundo geral da aplicação - Roxo Noturno do Banner */
.stApp {
    background-color: #0B0813;
}

/* Título principal - Neon Violeta Vibrante */
h1 {
    color: #A855F7 !important;
    font-weight: 800 !important;
}

/* Texto abaixo do título - Branco acinzentado de alto contraste */
.stApp p {
    color: #E2E8F0;
}

/* Caixa onde o usuário digita (Fundo CLARO com borda neon) */
div[data-testid="stChatInput"] {
    border: 2px solid #9933FF;
    border-radius: 14px;
    background-color: #FFFFFF !important;
    box-shadow: 0 0 15px rgba(153, 51, 255, 0.3);
}

/* 🌟 CONFIGURAÇÃO DO CAMPO CLARO COM LETRAS PRETAS */
div[data-testid="stChatInput"] textarea {
    background-color: #FFFFFF !important;
    color: #1A1A1A !important;
    -webkit-text-fill-color: #1A1A1A !important;
    caret-color: #9933FF !important;
}

/* Garante o contraste preto mesmo com foco ou seleção do navegador */
div[data-testid="stChatInput"] textarea:focus,
div[data-testid="stChatInput"] textarea:active {
    background-color: #FFFFFF !important;
    color: #1A1A1A !important;
    -webkit-text-fill-color: #1A1A1A !important;
}

/* Torna o texto de sugestão (placeholder) legível em cinza escuro sutil */
div[data-testid="stChatInput"] textarea::placeholder {
    color: #718096 !important;
    -webkit-text-fill-color: #718096 !important;
}

/* Botão de enviar dentro da caixa de texto */
div[data-testid="stChatInput"] button {
    background-color: #9933FF !important;
    color: #FFFFFF !important;
}

/* Mensagens do chat (Fundo Dark Tech com borda lateral brilhante) */
div[data-testid="stChatMessage"] {
    background-color: #13111C;
    border-left: 5px solid #9933FF;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
}

/* Força todo o texto interno das respostas a ficar branco */
div[data-testid="stChatMessage"] p, 
div[data-testid="stChatMessage"] li, 
div[data-testid="stChatMessage"] td, 
div[data-testid="stChatMessage"] th,
div[data-testid="stChatMessage"] span {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

/* Estilização das tabelas */
div[data-testid="stChatMessage"] table {
    border-collapse: collapse;
    width: 100%;
    margin-top: 10px;
}
div[data-testid="stChatMessage"] th, div[data-testid="stChatMessage"] td {
    border: 1px solid #3A354A !important;
    padding: 8px !important;
    text-align: left;
}
div[data-testid="stChatMessage"] th {
    background-color: #1E1A2E !important;
}

/* Aumenta o tamanho físico das imagens dos avatares no chat */
div[data-testid="stChatMessage"] img {
    width: 52px !important;
    height: 52px !important;
    max-width: 52px !important;
    max-height: 52px !important;
}

/* Ícone de carregamento / Spinner */
div[data-testid="stSpinner"] {
    color: #A855F7;
}

/* Links em geral */
a {
    color: #A855F7 !important;
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

try:
    if "GROQ_API_KEY" in st.secrets:
        groq_api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

if not groq_api_key:
    groq_api_key = os.environ.get("GROQ_API_KEY")

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

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    banco_vetorial = Chroma.from_documents(
        blocos_texto,
        embeddings
    )

    retriever = banco_vetorial.as_retriever(
        search_kwargs={"k": 5}
    )

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
# 5. HISTÓRICO DAS MENSAGENS (NOMES REAIS)
# ==========================================
if "mensagens" not in st.session_state:
    st.session_state.mensagens = []

for message in st.session_state.mensagens:
    avatar_chat = "byte_fundo_branco.png" if message["role"] == "assistant" else "aluno_chatbot.png"
    
    with st.chat_message(message["role"], avatar=avatar_chat):
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

    with st.chat_message("user", avatar="aluno_chatbot.png"):
        st.markdown(pergunta)

    with st.chat_message("assistant", avatar="byte_fundo_branco.png"):
        with st.spinner("Pesquisando no edital..."):
            try:
                documentos_relacionados = retriever.invoke(pergunta)

                contexto = "\n\n".join(
                    [doc.page_content for doc in documentos_relacionados]
                )

                # TRATAMENTO DE TEXTO AVANÇADO: Limpa as marcas HTML vindas do PDF bruto
                contexto = contexto.replace("<br>", "\n")
                contexto = contexto.replace("<ul>", "").replace("</ul>", "")
                contexto = contexto.replace("<li>", "- ").replace("</li>", "\n")

                prompt_completo = (
                    "Você é um assistente virtual especializado no Edital do Programa Trabalho Garantido da FMU.\n\n"
                    "CRONOGRAMA DO PROGRAMA (DADOS FIXOS E CRÍTICOS):\n"
                    "- Divulgação do Edital e abertura das inscrições: 05/08/2026\n"
                    "- Encerramento das inscrições: 20/11/2026\n"
                    "- Divulgação dos aprovados: 30/11/2026\n"
                    "- Início do PROGRAMA: 05/08/2026\n\n"
                    "Use os trechos do edital fornecidos abaixo e o cronograma acima para responder à pergunta do usuário.\n"
                    "Se a informação não puder ser extraída nem dos trechos e nem do cronograma acima, diga honestamente que não encontrou.\n\n"
                    f"Contexto recuperado do edital:\n{contexto}\n\n"
                    f"Pergunta do usuário: {pergunta}"
                )

                resposta_llm = llm.invoke(prompt_completo)

                if isinstance(resposta_llm.content, list):
                    texto_resposta = "".join(
                        bloco.get("text", "")
                        for bloco in resposta_llm.content
                        if isinstance(bloco, dict) and bloco.get("type") == "text"
                    )
                else:
                    texto_resposta = resposta_llm.content

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
