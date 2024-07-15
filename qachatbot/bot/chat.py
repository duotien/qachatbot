import os

import chainlit as cl
from chainlit.input_widget import Select
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from qachatbot import PERSIST_DIR, PROJECT_DIR
from qachatbot.commands import commands


def process_command(content: str):
    # TODO: refactor into a function called `process_command()`
    content = content.strip()
    cmd = content.split()
    response = f"Unknown command: {cmd[0]}"
    if cmd[0] == "/tp":
        if len(cmd) != 5:
            response = "Wrong syntax!"
        else:
            response = commands.tp(cmd[1], cmd[2], cmd[3], cmd[4])
    return response


async def process_response(message: cl.Message, chat_history):
    runnable = cl.user_session.get("runnable")  # type: Runnable
    # runnable.invoke({"question": message.content, "chat_history": chat_history})
    msg = cl.Message(content="")
    async for chunk in runnable.astream(
        {"question": message.content, "chat_history": chat_history},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)
    await msg.send()
    return msg.content


# TODO: add button to change k
def process_rag(user_input: str, k=3):
    vectorstore: Chroma = cl.user_session.get("vectorstore")
    retriever = vectorstore.as_retriever()
    context = retriever.invoke(user_input, k=k)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    context = format_docs(context)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "human",
                (
                    "You are an assistant for question-answering tasks. "
                    "Use the following pieces of retrieved context to answer the question. "
                    "If you don't know the answer, just say that you don't know. "
                    "Use three sentences maximum and keep the answer concise. \n"
                    "Question: {question} \n"
                    "Context: {context} \n"
                    "Answer:"
                ),
            ),
        ]
    )
    message = prompt.invoke({"context": context, "question": user_input})
    llm = cl.user_session.get("llm")
    message = llm.invoke(message)
    return message


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
                values=["Chroma"],
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


class VectorStoreManager:
    def __init__(self, embedding_function) -> None:
        self.embedding_function = embedding_function
        self._chromadb = None

    @property
    def chromadbd(self):
        if self._chromadb is None:
            self._chromadb = Chroma(
                persist_directory=PERSIST_DIR,
                embedding_function=self.embedding_function,
            )
        return self._chromadb
