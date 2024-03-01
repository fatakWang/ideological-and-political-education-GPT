# 1.配置ChatGLM

## 1.1 下载项目

```bash
git clone https://github.com/THUDM/ChatGLM-6B.git
```

## 1.2 下载依赖

进入项目：

```bash
cd ChatGLM-6B
```

下载依赖：

```bash
pip install -r requirements.txt
```

## 1.3 上传模型权重到服务器

本项目选用的是int4量化后的权重：https://huggingface.co/THUDM/chatglm-6b-int4

## 1.4 修改api.py

修改模型为int4量化后的版本，再将模型权重修改成本地权重所在的位置：

```python
# 修改前
#tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
#   model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()

#修改后
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True,resume_download=True).quantize(4).half().cuda()
```

## 1.5 运行api.py在后台启动chaglm

命令行运行

```
python api.py
```

# 2. 构建知识库

## 2.1 打开ipynb

## 2.2 安装依赖

```python
!pip install langchain
!pip install pypdf
!pip install sentence_transformers
!pip install markupsafe==2.0.1
!pip install faiss-cpu
!pip install pymupdf==1.19.0 
```

## 2.3 下载文本向量化模型并解压

本大作业选用的是llama的all-mpnet-base-v2，因为all-mpnet-base-v2是目前性能表现最好的文本向量化模型

```python
!wget -O llama_langchain.zip https://bj.bcebos.com/v1/ai-studio-online/5e882eb5d58648658950dd6ac9afb5ced4ca7a30e1f8414ba711967980b05a12?responseContentDisposition=attachment%3B%20filename%3Dllama_langchain%20.zip
!unzip  ./llama_langchain.zip
```



## 2.4 导入相关包 上传pdf

```python
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from tqdm import tqdm 
from langchain.document_loaders import TextLoader,PyPDFLoader
from llama_langchain.models.chinese_text_splitter import ChineseTextSplitter
from typing import Any, List, Dict, Mapping, Optional
from langchain import PromptTemplate
```



## 2.5 加载pdf

```python
pdfRoot = "./pdfdata"
pdf_files_name = os.listdir(pdfRoot)

# 加载pdf
loaders = []
for pdf_file_name in pdf_files_name:
    loaders.append(PyPDFLoader(pdfRoot+"/"+pdf_file_name))

# 将pdf拆分成每一页，得到pagecontent和metedata
documents = []
for loader in loaders:
    documents += loader.load_and_split()

# 数据清洗，消除字符串里面的’\n‘
for i in range(len(documents)):
    documents[i].page_content = documents[i].page_content.replace('\n','')

# 将每一份文档拆分成若干个chunk
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 800,
    chunk_overlap  = 100,
    separators=["，","。","？","！","、"]
)

splitted_documents = text_splitter.split_documents(documents)
```



## 2.6 构建知识库（文本向量数据库）

```python
embeddings = HuggingFaceEmbeddings(model_name="./llama_langchain/all-mpnet-base-v2",
                                   model_kwargs={'device':"cuda"})
db = FAISS.from_documents(splitted_documents, embeddings)
db.save_local('./vector_store')
```



## 2.7 利用知识库做语义检索

```python
## 本地知识库进行向量相似度匹配
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
VECTOR_SEARCH_TOP_K = 2
query = '什么是英雄史观与群众史观？'
def get_docs_with_score(docs_with_score):
    docs = []
    for doc, score in docs_with_score:
        doc.metadata["score"] = score
        docs.append(doc)
    return docs

vector_store = FAISS.load_local('./vector_store', embeddings)
vector_store.chunk_size = CHUNK_SIZE
related_docs_with_score = vector_store.similarity_search_with_score(query, k = 10)
related_docs = get_docs_with_score(related_docs_with_score)
```

## 2.8 实现学霸、老师功能

```python
def good_learner(query):
    ans=vector_store.similarity_search(query)
    return "emmm，我想了一会，我感觉这个知识点可能位于教材{}的第{}页,\
我背给你听嗷，好好记住吧！下次别问我了。这部分内容是：\n【{}】".format(ans[0].metadata['source'].split('/')[-1].split('.')[0],ans[0].metadata['page']+1,ans[0].page_content)

def teacher(query, k=10):
    embedding_vector = embeddings.embed_query(query)
    anss = vector_store.similarity_search_by_vector(embedding_vector, k=k)

    result_string = "同学啊，我刚才翻了翻书，你提到的这个问题与以下这些内容有关\n\n"
    
    for index, ans in enumerate(anss):
        result_string += "{}、{}的第{}页 ：【{}】\n\n".format(index + 1,
                                                       ans.metadata['source'].split('/')[-1].split('.')[0],
                                                       ans.metadata['page'] + 1, ans.page_content)

    result_string += "好好学习，未来是你们的！"

    return result_string
```



# 3. 实现知识问答

## 3.1  接入1.5启动的chatglm

```python
from langchain.llms import ChatGLM
from langchain.chains import RetrievalQA

llm = ChatGLM(
    endpoint_url='http://127.0.0.1:8000',
    max_token=80000,
    top_p=0.9
)
```

## 3.2 构建prompt、利用知识库检索到的答案嵌入prompt的上下文形成提问

```python
QA_CHAIN_PROMPT = PromptTemplate.from_template("""根据下面的上下文（context）内容回答问题。
如果你不知道答案，就回答不知道，不要试图编造答案。
答案最多3句话，保持答案简介。
总是在答案结束时说”谢谢你的提问！“
{context}
问题：{question}
""")
retriever = db.as_retriever()
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    verbose=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

ans = qa.run("什么是社会主义？")
print(ans)
```

## 3.3 搭建前端界面

学霸：

```python
import gradio as gr

def good_learner_chatbot(message, history):
    return good_learner(message)

gr.ChatInterface(
    good_learner_chatbot,
    chatbot=gr.Chatbot(height=300),
    textbox=gr.Textbox(placeholder="我是学霸，快来问我问题吧！", container=False, scale=7),
    title="思政大学霸",
    description="快来问学霸问题吧！",
    theme="soft",
    examples=["什么是社会主义？", "马克思、恩格斯", "封建社会"],
    cache_examples=True,
    retry_btn=None,
    undo_btn="Delete Previous",
    clear_btn="Clear",
).launch(share=True)
```

老师：

```python
def teacher_chatbot(message, history):
    return teacher(message)

gr.ChatInterface(
    teacher_chatbot,
    chatbot=gr.Chatbot(height=300),
    textbox=gr.Textbox(placeholder="我是老师，快来问我问题吧！", container=False, scale=7),
    title="思政老师",
    description="快来问老师问题吧！",
    theme="soft",
    examples=["什么是社会主义？", "马克思、恩格斯", "封建社会"],
    cache_examples=True,
    retry_btn=None,
    undo_btn="Delete Previous",
    clear_btn="Clear",
).launch(share=True)
```

教授：

```python
def professor_chatbot(message, history):
    return professor(message)

gr.ChatInterface(
    professor_chatbot,
    chatbot=gr.Chatbot(height=300),
    textbox=gr.Textbox(placeholder="我是教授，快来问我问题吧！", container=False, scale=7),
    title="思政教授",
    description="快来问教授问题吧！",
    theme="soft",
    examples=["什么是社会主义？", "马克思、恩格斯", "封建社会"],
    cache_examples=True,
    retry_btn=None,
    undo_btn="Delete Previous",
    clear_btn="Clear",
).launch(share=True)
```

