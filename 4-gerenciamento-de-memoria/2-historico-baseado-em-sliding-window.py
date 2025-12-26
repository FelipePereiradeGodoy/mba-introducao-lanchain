from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import trim_messages
from langchain_core.runnables import RunnableLambda

load_dotenv()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You're a helpful assistant that answers with a short joke when possible."),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])

llm = ChatOpenAI(model="gpt-5-nano", temperature=0.9)

def prepare_inputs(payload: dict) -> dict:
    raw_history = payload.get("raw_history", [])
    
    history_trimmed = trim_messages(
        raw_history,# Lista completa de mensagens a ser tratada, nesse caso o historico completo. Normalmente vem de MessageHistory.
        token_counter=len,#Função usada para "contar tokens". "len" = 1Token=1mensagem. Não é igual ao token de llm. Pode ter casos de personalizado.
        max_tokens=3,# Maximo de mensagens permitidas após o trim.
        strategy="last",# "last" = Mantém as mensagens mais recentes
        start_on="human",# "human" = Garante que o historico comece com uma mensagem do usuario.
        include_system=True,# Preserva mensagens do tipo "system"
        allow_partial=False,# Não permite cortar mensagens pela metade.
    )

    return {
        "input": payload.get("input", ""),
        "history": history_trimmed
    }

prepare = RunnableLambda(prepare_inputs)
chain = prepare | prompt | llm

session_store: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]


conversational_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="raw_history"
)

config = {"configurable": {"session_id": "demo-session"}}

resp1 = conversational_chain.invoke({"input": "My name is Felipe. Just reply with 'OK'. Don't mention my name unless I ask you to."}, config=config)
print("Assistant:", resp1.content)
print("-"*30)

resp2 = conversational_chain.invoke({"input": "Tell me a one-sentence fun fact."}, config=config)
print("Assistant:", resp2.content)
print("-"*30)

#Se aumentar o max_tokens para 6, então ele voltar a enxergar o historico contendo o nome e vai responder o nome.
resp3 = conversational_chain.invoke({"input": "What is my name? You are allowed to mention my name now."}, config=config)
print("Assistant:", resp3.content)
print("-"*30)