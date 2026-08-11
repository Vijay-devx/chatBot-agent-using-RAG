import chromadb
from langchain.embeddings import init_embeddings
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool
import json
import os

DB_DIR_PATH="../chromadb"
db = chromadb.PersistentClient(DB_DIR_PATH)
COLLECTION="exploration-cbse-ix"
collection = db.get_collection(COLLECTION)

llm = init_chat_model(model="groq:openai/gpt-oss-120b", api_key=os.getenv("GROQ_API_KEY"))
embed_model = init_embeddings(model="ollama:nomic-embed-text")

SYSTEM_PROMPT = """
    You are a QA assistant over a knowledge base.
    You have access to a tool named retrieve_knowledge.

    Workflow:
    1. Always use the retrieve_knowledge tool before answering.
    2. Inspect all retrieved passages.
    3. Answer only from the retrieved passages.
    4. Cite every page that contributed to the answer.
    5. If the answer cannot be found in the retrieved passages, do not infer or use outside knowledge. Report the error in following format.
        Reference => No details available
        Answer => The provided knowledge base does not contain enough information to answer this question.

    When the answer is available, respond in this format:
        Reference => Page <page numbers of all referenced pages>
        Answer => <answer>
    """

@tool
def retrieve_knowledge(question: str) -> str:
    """
    Tool Name:
    retrieve_knowledge

    Description:
    Search the knowledge base for information relevant to the user's question.
    It returns the most relevant pages along with their metadata.
    If the returned pages do not contain the answer, do not make up an answer.

    :param question: string
    :returns retrieved knowledge in JSON format or "ERROR\n<information>"
    """
    try:
        question_embedding = embed_model.embed_query(question)
        search_results = collection.query(question_embedding, n_results=5)
        return json.dumps(search_results)
    except Exception as e:
        return f"ERROR\n{e}"


agent = create_agent(
    model=llm,
    tools=[
        retrieve_knowledge
    ],
    system_prompt = SYSTEM_PROMPT
)

while True:
    question = input("Ask Anything? ")
    if question == "exit":
        break
    USER_PROMPT = question
    result = agent.invoke({
        "messages": [
            {"role": "user", "content": USER_PROMPT}
        ]
    })
    # for msg in result["messages"]:
        # print(msg, "\n", "-" * 60)
    response = result["messages"][-1]
    print("Final Answer:", response.content)
