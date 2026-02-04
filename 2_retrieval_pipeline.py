from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

persistent_directory = "db/chromadb"

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"},
)

query = "Is NVIDIA A GPU COMPANY?"

retriever = db.as_retriever(search_kwargs={"k": 4})

# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={"k": 5, "score_threshold": 0.3},
# )


# initiate the retriever to the query provided
relevant_docs = retriever.invoke(query)

# if (relevant_docs):
#     for i, doc in enumerate(relevant_docs, 1):
#         print(f"Document {i}: {doc.page_content}")
# else:
#     print("no docs generated!")


combined_input = f"""Based on the following documents, please answer this question: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I dont have enough information to answer that question based on the provided documents
"""

model = ChatOpenAI(model="gpt-5-nano", temperature=0, max_tokens=1000)

messages = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content=combined_input)
]

result = model.invoke(messages)

print(result.content)
