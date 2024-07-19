import os
from typing import Any, Dict

from PIL import Image
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

from qachatbot import PERSIST_DIR, PROJECT_DIR
from qachatbot.bot.vision import convert_to_base64
from qachatbot.commands import commands


def process_command(content: str):
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
async def process_rag(user_input: str, k=3):
    runnable = cl.user_session.get("runnable")
    msg = cl.Message(content="")
    async for chunk in runnable.astream(
        user_input,
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)
    await msg.send()
    return msg


async def process_uploaded(message):
    for element in message.elements:
        if type(element) == cl.File:
            print("[DEBUG] You uploaded a File")
            pass
        if type(element) == cl.Image:
            print("[DEBUG] You uploaded an Image")
            image_b64 = Image.open(element.path)
            image_b64 = image_b64.convert("RGB")
            image_b64 = convert_to_base64(image_b64)

            runnable = cl.user_session.get("runnable")  # type: Runnable
            msg = cl.Message(content="")
            async for chunk in runnable.astream(
                {"question": message.content, "image": image_b64},
                config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
            ):
                await msg.stream_token(chunk)
            await msg.send()
            return msg.content
    pass


async def init_settings():
    settings = await cl.ChatSettings(
        [
            Select(
                id="model",
                label="Model",
                values=["phi3", "llama3", "llava-phi3"],
                initial_index=2,
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
                values=["chat", "rag", "chat-vision"],
                initial_index=2,
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
                    "Question: {question} \n"
                    "Context: {context} \n"
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
                "You are a AI god name Akashic, you answer questions with simple answers and no funny stuff, only answers short, focus on result",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    runnable = prompt | llm | StrOutputParser()
    cl.user_session.set("runnable", runnable)


def setup_chatbot2(settings):
    def _prompt_func(data):
        text = data["question"]
        image = data["image"]

        image_part = {
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{image}",
        }
        text_part = {"type": "text", "text": text}

        content_parts = []

        content_parts.append(image_part)
        content_parts.append(text_part)

        return [HumanMessage(content=content_parts)]

    llm = ChatOllama(model=settings["model"])

    runnable = _prompt_func | llm | StrOutputParser()
    cl.user_session.set("runnable", runnable)




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
