import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain
import pandas as pd
# Inicializando o modelo LLM com a chave da API do OpenAI
llm = ChatOpenAI(api_key="sk-proj-E8jiUamWK3kHp7WVygD6T3BlbkFJtfFsUxlnjPj0MbdwvyFn")

# Carregando e dividindo os documentos
loader = CSVLoader(file_path="Cardapio.csv")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)

# Criando Embeddings e Índice Vetorial
embeddings = OpenAIEmbeddings(openai_api_key="sk-proj-E8jiUamWK3kHp7WVygD6T3BlbkFJtfFsUxlnjPj0MbdwvyFn")
vector = FAISS.from_documents(documents, embeddings)

# Configurando o prompt para a cadeia de documentos
prompt_template = ChatPromptTemplate.from_template("""
Você é um atendente de restaurante que vende marmitas responda sempre as questões se baseando nas regras do negócio, se a pergunta não se aplicar as regras responda a vontade mas mantendo alguma similaridade com o negócio
Regras do negocio:
Trabalhamos sob encomenda e nossas entregas ocorrem nas Terças e Sextas com 48h de antecedência para os pedidos. 

Nosso preço é único de R$19,00 para as marmitas pequenas e R$22,00 para as marmitas grandes. O pedido pode ser feito através do número do prato e quantidade desejada.

Nossos cardápios mudam conforme a estação do ano

Aceitamos como pagamento dinheiro, pix, picpay, crédito, débito ou Vouchers (Sodexo (alimentação e refeição), VR, ticket (flex e refeição) e Alelo (refeição)- acréscimo de 15% em todos os Vouchers)

Promoção: Comprando 15 unidades ou mais conseguimos a entrega grátis em Porto Alegre (promoção válida com pagamento via pix, dinheiro ou picpay)

Para fazer um pedido basta escolher pelos números dos pratos e enviar o número e quantidade desejada.

voce anotara os pedidos e dira o valor final, forma de pagamento e entrega 
<context>
{context}
</context>

Question: {input}
""")

# Criando a cadeia de documentos
document_chain = create_stuff_documents_chain(llm, prompt_template)

# Configurando o sistema de recuperação
retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Função para fazer perguntas ao modelo
def ask_question(question):
    response = retrieval_chain.invoke({"input": question})
    return response["answer"]

# Configurando a interface do Streamlit
st.write("Iniciando o aplicativo Streamlit...")
st.title('Exemplo de Adição de Imagem')
file_path = "Cardapio.csv"
df = pd.read_csv(file_path,delimiter=";")
st.write(df)
st.write("---")
st.title("Chatbot de Restaurante")
st.write("Faça suas perguntas sobre o cardápio e o funcionamento do restaurante.")

question = st.text_input("Faça sua pergunta:")

if st.button("Enviar"):
    if question:
        answer = ask_question(question)
        st.write("Resposta:", answer)
    else:
        st.write("Por favor, insira uma pergunta.")

# Rodando o Streamlit
if __name__ == "__main__":
    st.write("Iniciando o aplicativo Streamlit...")
    st.title('Exemplo de Adição de Imagem')
    file_path = "Cardapio.csv"
    df = pd.read_csv(file_path)


 