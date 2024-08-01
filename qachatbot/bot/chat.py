import os
from typing import Any, Dict

import chainlit as cl
from chainlit.input_widget import Select
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import Runnable, RunnablePassthrough
from langchain.schema.runnable.config import RunnableConfig
from langchain_chroma import Chroma
from langchain.schema import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableSequence
from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.runnables.history import RunnableWithMessageHistory

from qachatbot import PERSIST_DIR, MD_PERSIST_DIR
from qachatbot.commands import commands

from qachatbot.settings import store_session, embedding_function

from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)



# URL = "https://www.youtube.com/watch?v=9RhWXPcKBI8" # mr bearst
# URL = "https://www.youtube.com/watch?v=0ZZquVylLEo"

def process_command(content: str):
    content = content.strip()
    cmd = content.split()
    response = f"Unknown command: {cmd[0]}"
    if cmd[0] == "/tp":
        if len(cmd) != 5:
            response = "Wrong syntax!"
        else:
            response = commands.tp(cmd[1], cmd[2], cmd[3], cmd[4])
    if cmd[0] == "/yt":
        commands.yt(cmd[1:])
        response = "Processed video! change DB to `Temp` to chat about video"

    return response


async def process_response(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable
    msg = cl.Message(content="")
    
    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(
            callbacks=[cl.LangchainCallbackHandler()],
            configurable={"session_id": cl.user_session.get("id")},
        ),
    ):
        await msg.stream_token(chunk)
    await msg.send()
    return msg.content


# TODO: add button to change k
async def process_rag(user_input: str, k=5):
    runnable = cl.user_session.get("runnable")
    msg = cl.Message(content="")
    async for chunk in runnable.astream(
        user_input,
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)
    await msg.send()
    return msg


async def init_settings():
    settings = await cl.ChatSettings(
        [
            Select(
                id="model",
                label="Model",
                values=["phi3", "llama3"],
                initial_index=0,
            ),
            Select(
                id="DB",
                label="Database",
                values=["Chroma", "Markdown", "Temp"],
                initial_index=0,
            ),
            Select(
                id="chat_mode",
                label="Chat Mode",
                values=["chat", "rag"],
                initial_index=0,
            ),
        ]
    ).send()
    return settings


def setup_qabot(settings: Dict[str, Any]):
    vectorstore: Chroma = cl.user_session.get("vectorstore")
    retriever = vectorstore.as_retriever()

    def _format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    llm = ChatOllama(model=settings["model"])
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "human",
                (
                    "You are an assistant for question-answering tasks. "
                    "Use the following pieces of retrieved context to answer the question. "
                    "If you don't know the answer, just say that you don't know. "
                    "Use three sentences maximum and keep the answer concise. \n"
                    "Context: {context} \n"
                    "Question: {question} \n"
                    "Answer:"
                ),
            ),
        ]
    )

    runnable = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    cl.user_session.set("runnable", runnable)


def setup_chatbot(settings: Dict[str, Any]):
    llm = ChatOllama(model=settings["model"])
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                # "You are a double agent working for both CIA and KGB, your task is to help the user with whatever they ask, be polite and elegant like a true spy",
                "Your are an AI chat bot, reply with four or less sentences max.",
            ),
            MessagesPlaceholder(variable_name="question"),
            # ("human", "{question}"),
        ]
    )
    runnable = prompt | llm | StrOutputParser()
    runnable = RunnableWithMessageHistory(runnable, get_session_history, input_messages_key="question")
    cl.user_session.set("runnable", runnable)


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store_session:
        store_session[session_id] = InMemoryChatMessageHistory()
    return store_session[session_id]
