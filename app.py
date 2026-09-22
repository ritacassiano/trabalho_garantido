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
# PERSONALIZAÇÃO VISUAL - PALETA SUMMIT TECH DARK CORRIGIDA
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

/* Caixa onde o usuário digita (Fundo escuro com borda neon) */
div[data-testid="stChatInput"] {
    border: 2px solid #9933FF;
    border-radius: 14px;
    background-color: #13111C !important;
    box-shadow: 0 0 15px rgba(153, 51, 255, 0.3);
}

/* Força o texto digitado na caixa de entrada a ficar branco e visível */
div[data-testid="stChatInput"] textarea {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    caret-color: #A855F7 !important;
}

/* Torna o texto de sugestão (placeholder) legível em cinza claro */
div[data-testid="stChatInput"] textarea::placeholder {
    color: #A0AEC0 !important;
    -webkit-text-fill-color: #A0AEC0 !important;
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

/* Força o texto dentro dos balões de mensagens anteriores a ficar branco */
div[data-testid="stChatMessage"] p, div[data-testid="stChatMessage"] li {
    color: #FFFFFF !important;
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
# 5. HISTÓRICO DAS MENSAGENS (COM OS NOVOS AVATARES 3D CORRIGIDOS)
# ==========================================
if "mensagens" not in st.session_state:
    st.session_state.mensagens = []

for message in st.session_state.mensagens:
    # 🌟 Atualizado: Mapeia para o nome real aluno_chatbot.png
    avatar_chat = "byte_rosto.png" if message["role"] == "assistant" else "aluno_chatbot.png"
    
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

    # 🌟 Atualizado: Usa o nome correto para o balão do usuário
    with st.chat_message("user", avatar="aluno_chatbot.png"):
        st.markdown(pergunta)

    with st.chat_message("assistant", avatar="byte_rosto.png"):
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
