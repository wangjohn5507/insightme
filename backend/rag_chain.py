from langchain.chat_models import init_chat_model
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
from typing_extensions import List, TypedDict, Annotated
from langchain import hub
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

import time

# 紀錄每個 thread_id 最後一次活動時間
session_last_active = {}
SESSION_TIMEOUT_SECONDS = 10 * 60  # 30 分鐘

from dotenv import load_dotenv
load_dotenv()

from consts import INDEX_NAME
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

prompt = hub.pull("langchain-ai/retrieval-qa-chat")
condense_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
llm = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0.0)

# thread_id -> MemorySaver
memory_map: dict[str, MemorySaver] = {}
# thread_id -> last accessed time
last_access_map: dict[str, float] = {}
# TTL for session in seconds (e.g., 30 min)
SESSION_TTL = 1800

def is_expired(thread_id: str) -> bool:
    last_access = last_access_map.get(thread_id, 0)
    return (time.time() - last_access) > SESSION_TTL

def get_memory(thread_id: str) -> MemorySaver:
    if thread_id not in memory_map:
        memory_map[thread_id] = MemorySaver()
    last_access_map[thread_id] = time.time()
    return memory_map[thread_id]

def get_memory(thread_id: str) -> MemorySaver:
    # 如果还没创建，new 一个
    if thread_id not in memory_map:
        memory_map[thread_id] = MemorySaver()
    return memory_map[thread_id]

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    messages: Annotated[List[BaseMessage], add_messages]

def evaluate(state: State):
    evaluate_prompt = """
    You are an AI assistant represents me and responsible for answering questions about me and my background.
    Before answering, you must first determine whether the question is relevant to my background or personal information.

    Question: {question}

    Criteria for determining relevance:
    1. If the question asks about my education, skills, interests, work experience, personal attributes, or any other background information, it should be considered relevant.
    2. If the question is general or does not clearly request personal information, it should be considered not_relevant.
    3. If it is uncertain whether the question is related, make a reasonable guess based on the information available.
    4. Do not be too strict in your evaluation. If the question is somewhat related, consider it relevant.

    Respond with:
    "relevant" if the question meets one of the criteria.
    "not_relevant" if the question is unrelated or inappropriate.
    """

    relevance_chain = (
        ChatPromptTemplate.from_template(evaluate_prompt)
        | llm
        | StrOutputParser()
    )

    result = relevance_chain.invoke({"question": state["question"]})
    print(f"Relevance evaluation result: {result}")
    if "not" in result.lower():
        return "reject"
    else:
        return "proceed"
    
# def reject_question(state: State):
#     return {
#         "answer": "I apologize, but I cannot answer this question as it is not relevant to my expertise or background.",
#         "context": []
#     }


def retrieve(state: State):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = PineconeVectorStore(index=index, embedding=embeddings, namespace="my_namespace")
    chat_history = state["messages"][:-1]  # 排除最後一個消息
    print(f"Chat history: {chat_history}")
    condense_chain = condense_prompt | llm
    standalone_question = condense_chain.invoke({"input": state["question"], "chat_history": chat_history}).content
    print(f"Condensed question: {standalone_question}")
    retrived_docs = vector_store.similarity_search(standalone_question, k=4)
    return {"context": retrived_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"input": state["question"], "context": docs_content, "chat_history": state["messages"][:-1]})
    response = llm.invoke(messages)
    return {"answer": response}

def run_graph(question, thread_id):
    # 檢查過期
    if is_expired(thread_id):
        memory_map.pop(thread_id, None)
        print(f"Session for {thread_id} expired. Resetting memory.")

    memory = get_memory(thread_id)

    # 先檢查問題相關性
    relevance_result = evaluate({
        "question": question, 
        "messages": [HumanMessage(content=question)]
    })
    
    # 如果問題不相關，直接返回拒絕消息
    if relevance_result == "reject":
        return "I apologize, but I cannot answer this question as it is not relevant to my expertise or background.", []

    graph_builder = StateGraph(State)
    # graph_builder.add_node("evaluate", evaluate)
    graph_builder.add_node("retrieve", retrieve)
    graph_builder.add_node("generate", generate)
    
    # graph_builder.add_node("reject", reject_question)
    # graph_builder.set_conditional_entry_point(
    #     evaluate, {"proceed": "retrieve", "reject": "reject"}
    # )

    graph_builder.add_edge(START, "retrieve")
    graph_builder.add_edge("retrieve", "generate")
    graph = graph_builder.compile(checkpointer=memory)

    # 使用唯一的線程 ID
    config = {"configurable": {"thread_id": thread_id}}

    result = graph.invoke({"question": question, "messages": [HumanMessage(content=question)]}, config=config)
    print(f'Context: {result["context"]}\n\n')
    print(f'Answer: {result["answer"]}')
    return result["answer"].content, [result["context"][i] for i in range(len(result["context"]))]
    





if __name__ == "__main__":
    answer, context = run_graph("Write an function to calculate the sum of two numbers.")
    print('*' * 20)
    print('Answer: ', answer)
    if context:
        for doc in context:
            print(doc.metadata.get('source', None))






