import os
from typing import Any, Dict
import requests

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
import PIL.Image

from qachatbot import PERSIST_DIR, MD_PERSIST_DIR
from qachatbot.bot.vision import convert_to_base64
from qachatbot.commands import commands

from qachatbot.settings import store_session

from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)

BASE_CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Your are an AI chat bot, reply with four or less sentences max.",
        ),
        MessagesPlaceholder(variable_name="question"),
    ]
)

BASE_RAG_PROMPT = ChatPromptTemplate.from_messages(
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


async def process_response(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable
    response = cl.Message(content="")

    async for chunk in runnable.astream(
        {
            "question": message.content,
        },
        config=RunnableConfig(
            callbacks=[cl.LangchainCallbackHandler()],
            configurable={"session_id": cl.user_session.get("id")},
        ),
    ):
        await response.stream_token(chunk)
    await response.send()
    return response.content


async def process_response_with_vision(message: cl.Message):
    llm_has_clip = cl.user_session.get("llm_has_clip")
    response = cl.Message(content="")
    if not llm_has_clip:
        response.content = "Sorry, I cannot handle image."
        await response.send()
        return response
    await process_uploaded(message)


# TODO: add button to change k
async def process_rag(user_input: str, k=5):
    runnable = cl.user_session.get("runnable")
    response = cl.Message(content="")
    async for chunk in runnable.astream(
        user_input,
        config=RunnableConfig(
            callbacks=[cl.LangchainCallbackHandler()],
            configurable={"session_id": cl.user_session.get("id")},
        ),
    ):
        await response.stream_token(chunk)
    await response.send()
    return response


# currently limited to 1 file
async def process_uploaded(message):
    for element in message.elements:
        if type(element) == cl.File:
            print("[DEBUG] You uploaded a File")
            await cl.Message(
                content="You uploaded a file, but I cannot process it yet."
            ).send()

        if type(element) == cl.Image:
            print("[DEBUG] You uploaded an Image")
            image = PIL.Image.open(element.path)
            image = image.convert("RGB")
            image = convert_to_base64(image)

            runnable = cl.user_session.get("runnable")  # type: Runnable
            response = cl.Message(content="")
            async for chunk in runnable.astream(
                input={
                    "question": message.content,
                    "image": image,
                },
                config=RunnableConfig(
                    callbacks=[cl.LangchainCallbackHandler()],
                    configurable={"session_id": cl.user_session.get("id")},
                ),
            ):
                await response.stream_token(chunk)
            await response.send()
    return response.content


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
                values=["Chroma", "Markdown"],
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
    prompt = BASE_RAG_PROMPT
    runnable = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    cl.user_session.set("runnable", runnable)


def setup_chatbot(settings: Dict[str, Any]):
    llm = ChatOllama(model=settings["model"])

    has_clip = model_has_clip(llm)
    prompt = BASE_CHAT_PROMPT
    if has_clip:
        prompt = _base_prompt_func

    runnable = prompt | llm | StrOutputParser()
    runnable = RunnableWithMessageHistory(
        runnable, get_session_history, input_messages_key="question"
    )
    cl.user_session.set("runnable", runnable)
    cl.user_session.set("llm_has_clip", has_clip)


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store_session:
        store_session[session_id] = InMemoryChatMessageHistory()
    return store_session[session_id]


def model_has_clip(llm: ChatOllama) -> bool:
    ollama_llm_model_name = llm.model if ":" in llm.model else f"{llm.model}:latest"
    tags = requests.request("GET", url=f"{llm.base_url}/api/tags").json()
    for model_info in tags["models"]:
        if ollama_llm_model_name == model_info["model"]:
            if "clip" in model_info["details"]["families"]:
                return True
    return False


def _base_prompt_func(data: Dict[str, Any]):
    prompt = BASE_CHAT_PROMPT
    if "image" in data:
        content_parts = []
        text_part = {
            "type": "text",
            "text": data["question"],
        }
        if data["image"] != "":
            image_part = {
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{data['image']}",
            }
            content_parts.append(image_part)
        else:
            text_part += (
                ". (the user added an image but you cannot process it, "
                "respond with 4 sentences maximum to inform the user)"
            )

        content_parts.append(text_part)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are an AI assistant, you reply the user with "
                        "concise and easy to understand answers."
                    ),
                ),
                MessagesPlaceholder(variable_name="question", optional=True),
                HumanMessage(content=content_parts),
            ]
        )
        print(f"[DEBUG] {content_parts}")

    print(f"[DEBUG] {prompt}")

    return prompt
