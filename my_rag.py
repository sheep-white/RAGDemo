from langchain_community.document_loaders import TextLoader       # 文本加载器：把 knowledge.txt 读成 Document 对象（正文 + 来源元数据）
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 递归分块器：优先按 段落→换行→空格 切，尽量不切断句子
import os                                                        # 用来从环境变量读取密钥
from dotenv import load_dotenv                                 # 用来加载 .env 文件
from langchain_openai import OpenAIEmbeddings                  # 通义的向量模型：把文字变成 1024 维向量
from langchain_chroma import Chroma                             # Chroma 向量数据库：存向量 + 原文，支持语义检索
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

load_dotenv()  # 把 .env 里的密钥/接口地址读进环境变量；好处：密钥不写死在代码里

loader = TextLoader("knowledge.txt", encoding="utf-8")  # encoding 必须指定：Windows 下不写中文文件容易报编码错误
docs = loader.load()  # 读成 Document 列表；Document = page_content（正文）+ metadata（来源文件等）

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,    # 当前是实验值，真实长文档建议恢复为 300（每块最大字符数，超过才切）
    chunk_overlap=50,  # 当前是实验值，建议恢复为 50（相邻块重叠字符数，防止句子在边界被切断后语义不完整）
)

chunks = splitter.split_documents(docs)  # 执行切分，得到 Document 小块列表


embeddings = OpenAIEmbeddings(
    model="text-embedding-v3",                        # 通义的向量模型，输出 1024 维
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
    check_embedding_ctx_length=False,  # 关闭本地分词：阿里云接口只认原始字符串，开着会把 token 编号数组发过去导致 400 报错
)

if os.path.exists("./chroma_db"):
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
    )
else:
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db",
    )

retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k":3,
        "score_threshold":0.4,
    },
)

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
)

prompt = ChatPromptTemplate.from_template("""你是知识库问答助手。请严格根据下面的参考资料回答问题。如果参考资料为空或与问题无关，请直接回答"知识库没有相关信息"，绝对不要编造。
历史对话（仅用于理解当前问题中的指代，不要从中找答案）：
{history}

参考资料：
{context}

问题：{question}
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"history":itemgetter("history"),
     "context":itemgetter("question") | retriever | format_docs,
     "question":itemgetter("question"),
     }
    | prompt
    | llm
    | StrOutputParser()
)

condense_prompt = ChatPromptTemplate.from_template("""根据下面的历史对话，把当前问题改写成一个不依赖上下文、可以独立理解的完整问题。只输出改后的问题本身，不要回答它。

历史对话：
{history}

当前问题：{question}
改写后的问题：
""")

condense_chain = condense_prompt | llm | StrOutputParser()



if __name__ == "__main__":
    print("知识库问答已启动，输入问题开始对话（输入q退出）")

    history = []

    def format_history(history):
        recent = history[-3:]
        return "\n".join(f"问：{q}\n答：{a}" for q,a in recent)

    while True:
        question = input("我：")
        if question.strip().lower() in ("q","quit","exit"):
            print("再见！")
            break
        if not question.strip():
            continue
        standalone = condense_chain.invoke({"question":question,"history":format_history(history)})
        answer = chain.invoke({"question":standalone,"history":format_history(history)})
        history.append((question,answer))
        print(f"助手:{answer}")