import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# ------------ CONFIG ------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

INDEX_PATH = "../index/openai_index"
EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-3.5-turbo"
# --------------------------------


def load_vectorstore():
    try:
        embeddings = OpenAIEmbeddings(
            model=EMBED_MODEL,
            api_key=OPENAI_API_KEY
        )
        return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"[❌] Failed to load FAISS index: {e}")
        return None


def build_qa_chain(vectorstore):
    try:
        retriever = vectorstore.as_retriever(search_type="similarity", k=5)

        template = """
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
        prompt = PromptTemplate(input_variables=["context", "question"], template=template)

        llm = ChatOpenAI(
            temperature=0.2,
            model=LLM_MODEL,
            api_key=OPENAI_API_KEY
        )

        return RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt}
        )

    except Exception as e:
        print(f"[❌] Failed to build QA chain: {e}")
        return None


def main():
    print("[📂] Loading vector store...")
    vectorstore = load_vectorstore()

    if not vectorstore:
        print("[❌] Vector store loading failed.")
        return

    print("[🤖] Initializing RAG chain...")
    qa_chain = build_qa_chain(vectorstore)

    if qa_chain is None:
        print("[❌] QA chain initialization failed.")
        return

    print("\nAsk your legal or complaint-related question (Ctrl+C to exit):\n")
    while True:
        try:
            query = input("🧑 You: ")
            response = qa_chain.run(query)
            print(f"\n🤖 Bot:\n{response}\n")
        except KeyboardInterrupt:
            print("\n👋 Exiting. Stay safe!")
            break
        except Exception as e:
            print(f"[⚠️] Error during response generation: {e}\n")


if __name__ == "__main__":
    main()
