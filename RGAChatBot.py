import streamlit as st
import tiktoken
from loguru import logger

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS

# from streamlit_chat import message
from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory

def main():
    st.set_page_config(page_title="Organizational Behaviorism Chatbot", page_icon=":books:")

    st.title("조직행동론 챗봇 :blue[대진대학교] :books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = False

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", "content": "안녕하세요! 대진대학교 조직행동론 강의에 관한 질문 챗봇입니다! 질문을 입력해주세요!"}]

    with st.sidebar:
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        process = st.button("Process")

    if process:
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        files_text = get_predefined_text()
        text_chunks = get_text_chunks(files_text)
        vectorstore = get_vectorstore(text_chunks)

        st.session_state.conversation = get_conversation_chain(vectorstore, openai_api_key)
        st.session_state.processComplete = True

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    # Chat logic
    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        if st.session_state.processComplete:
            with st.chat_message("assistant"):
                chain = st.session_state.conversation

                with st.spinner("답변 생성중..."):
                    with get_openai_callback() as cb:
                        result = chain({"question": query})
                        st.session_state.chat_history = result['chat_history']
                    response = result['answer']
                    source_documents = result.get('source_documents', [])

                    st.markdown(response)
                    with st.expander("참고 문서 확인"):
                        for doc in source_documents:
                            st.markdown(f"{doc.metadata['source']}", help=doc.page_content)

                # Add assistant message to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def get_predefined_text():
    file_paths = [
        "OB_Week1_경영환경의 변화와 조직행동론.pdf",
        "OB_Week2_개인차이에 대한 이해1_성격, 지각, 귀인.pdf",
        "OB_Week3_개인차이에 대한 이해(2)_태도, 가치관, 윤리, 능력.pdf",
        "OB_Week4_학습 및 강화.pdf",
        "OB_Week5_동기부여.pdf",
        "OB_Week6_팀관리.pdf",
        "OB_Week7_스트레스와 갈등관리.pdf",
        "OB_Week9_리더십이론1_전통적 리더십이론.pdf",
        "OB_Week10_리더십이론2_현대적 리더십이론.pdf",
        "OB_Week11_커뮤니케이션.pdf",
        "OB_Week12_의사결정.pdf",
        "OB_Week13_조직문화.pdf",
        "OB_Week14_변화관리와 조직개발.pdf"
    ]
    
    documents = []
    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load_and_split())
    
    return documents

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb

def get_conversation_chain(vectorstore, openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-3.5-turbo', temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type='mmr', verbose=True),
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True
    )
    return conversation_chain

if __name__ == '__main__':
    main()
