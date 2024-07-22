import os
from typing import Any, Dict

import chainlit as cl
import requests
from chainlit.input_widget import Select, TextInput
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable, RunnablePassthrough
from langchain.schema.runnable.config import RunnableConfig
from langchain_chroma import Chroma
from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableSequence
from PIL import Image

from qachatbot import MD_PERSIST_DIR, PERSIST_DIR
from qachatbot.bot.vision import convert_to_base64
from qachatbot.commands import commands

BASE_CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a AI god name Akashic, you answer questions "
                "with simple answers and no funny stuff, only answers short, focus on result"
            ),
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
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


# TODO: add history when chatting with image
async def process_response_with_vision(message: cl.Message):
    llm_has_clip = cl.user_session.get("llm_has_clip")
    response = cl.Message(content="")
    if not llm_has_clip:
        response.content = "Sorry, this model cannot handle image."
        await response.send()
        return response.content

    await process_uploaded(message)


async def process_response(message: cl.Message, chat_history):
    runnable = cl.user_session.get("runnable")  # type: Runnable

    # runnable.invoke({"question": message.content, "chat_history": chat_history})
    response = cl.Message(content="")
    async for chunk in runnable.astream(
        {
            "question": message.content,
            "chat_history": chat_history,
            # "image": None,
        },
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await response.stream_token(chunk)
    await response.send()
    return response.content


# TODO: add button to change k
async def process_rag(message: str, k=5):
    runnable = cl.user_session.get("runnable")
    response = cl.Message(content="")
    async for chunk in runnable.astream(
        message.content,
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
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
            image = Image.open(element.path)
            image = image.convert("RGB")
            image = convert_to_base64(image)

            runnable = cl.user_session.get("runnable")  # type: Runnable
            response = cl.Message(content="")
            async for chunk in runnable.astream(
                input={
                    "question": message.content,
                    "chat_history": [],
                    "image": image,
                },
                config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
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
            # TextInput(
            #     id="system_prompt",
            #     label="System Prompt",
            #     initial="",
            # )
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
        cl.user_session.set("llm-has-clip", True)
        prompt = _base_prompt_func

    runnable = prompt | llm | StrOutputParser()

    cl.user_session.set("runnable", runnable)
    cl.user_session.set("llm_has_clip", has_clip)


# def setup_chatbot_with_vision(settings):
#     llm = ChatOllama(model=settings["model"])
#     has_clip = model_has_clip()

#     _prompt_func = lambda data: _base_vision_prompt_func(data, has_clip=has_clip)
#     runnable = _prompt_func | llm | StrOutputParser()
#     cl.user_session.set("runnable", runnable)


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
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                HumanMessage(content=content_parts),
            ]
        )
        print(f"[DEBUG] {content_parts}")

    print(f"[DEBUG] {prompt}")

    return prompt
