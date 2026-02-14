from dotenv import load_dotenv
load_dotenv()

import warnings
warnings.filterwarnings("ignore")

# pip install langchain langchain-anthropic langchain-community
# pip install chromadb       # 向量数据库（本地免费）
# pip install langchain-huggingface  # Embedding模型
# pip install -r requirements.txt
# pip install sentence-transformers


from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
# pip install faiss-cpu
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# ========== 1. 加载文档 ==========
print("📄 加载文档...")
loader = DirectoryLoader("docs", glob="**/*.txt", loader_cls=TextLoader)
documents = loader.load()
print(f"   加载了 {len(documents)} 个文档")

# ========== 2. 文档切片 ==========
print("✂️ 切分文档...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,      # 每个片段最大200字符
    chunk_overlap=50     # 片段之间重叠50字符，保持上下文
)
chunks = splitter.split_documents(documents)
print(f"   切分为 {len(chunks)} 个片段")

# ========== 3. 向量化并存入向量数据库 ==========
print("🔢 向量化并存入数据库...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
# vectorstore = Chroma.from_documents(
#     documents=chunks,
#     embedding=embeddings,
    # persist_directory="./chroma_db"  # 持久化到本地
# )

vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # 每次检索3个最相关片段
print("   ✅ 向量数据库创建完成")

# ========== 4. 构建 RAG Chain ==========
llm = ChatAnthropic(model_name="claude-sonnet-4-20250514")

prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个知识库助手。请严格根据以下参考文档回答用户的问题。
如果文档中没有相关信息，请如实告诉用户你不知道，不要编造。

参考文档：
{context}"""),
    ("human", "{question}")
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# ========== 5. 对话循环 ==========
print("\n🤖 RAG 知识库助手已启动（输入 quit 退出）\n")

while True:
    question = input("你: ")
    if question.lower() in ["quit", "exit"]:
        break

    response = rag_chain.invoke(question)
    print(f"AI: {response.content}\n")