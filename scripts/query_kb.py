import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# --- Load .env locally ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Config ---
INDEX_PATH = "index/openai_index"
EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-3.5-turbo"

def get_qa_chain():
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL, api_key=OPENAI_API_KEY)
    vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_type="similarity", k=5)

    prompt_template = """
You are a highly accurate legal assistant for a citizen-facing police chatbot.

Using the context provided below, answer the user’s legal or procedural query in a structured format with clear headings.

Context:
{context}

Question:
{question}

Provide the response in the following format:

---

**Issue Type:**  
(Briefly describe the nature of the issue)

**Applicable Law Sections:**  
(Mention relevant sections from BNS, CrPC, or other laws)

**Steps to Follow:**  
(Explain the procedure step-by-step for the citizen)

**Additional Help / Contacts:**  
(Suggest relevant police contacts or resources, if available)

---
"""
    prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0.2, api_key=OPENAI_API_KEY)

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

    return chain
