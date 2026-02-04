from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

persistent_directory = "db/chromadb"
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
)

model = ChatOpenAI(model="gpt-4o")

chat_history = []

def ask_question(user_question):
    print(f"You asked: {user_question}")

    system_message = "Given the chat history, rewrite the new question to be a standalone and searchable. Just return the rewritten question"

    if chat_history:
        messages = [
            SystemMessage(content=system_message)
            + chat_history
            + HumanMessage(content=f"New Question : {user_question}")
        ]

        result = model.invoke(messages)
        search_question = result.content.strip()
        print(f"Searching for {search_question}")
    else:
        search_question = user_question

    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(search_question)

    print(f"Found {len(docs)} relevant documents: ")
    for i, doc in enumerate(docs, 1):
        lines = doc.page_content.split("\n")[:2]
        preview = "\n".join(lines)
        print(f"Doc {i}: {preview}")

    combined_input = f"""Based on the following documents, please answer this question: {search_question}

    Documents:
    {"\n".join([f"- {doc.page_content}" for doc in docs])}

    Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I dont have enough information to answer that question based on the provided documents
    """

    answer_message = "You are a helpful assistant that answer questions based on provided documents and conversation history."

    messages = [
        SystemMessage(content=answer_message)
        + chat_history
        + HumanMessage(content=combined_input)
    ]

    result = model.invoke(messages)
    answer = result.content

    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    print(f"Answer: {answer}")

    return answer


def start_chat():
    return


if __name__ == "__main__":
    start_chat()
