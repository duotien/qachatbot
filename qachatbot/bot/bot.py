import os
from typing import Any, Dict

import chainlit as cl
import requests
from chainlit.input_widget import Select
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_chroma import Chroma
from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory

from qachatbot.settings import store_session

BASE_CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Your are an AI chat bot, reply with four or less sentences max.",
        ),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
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


def _base_prompt_func(data):
    prompt = BASE_CHAT_PROMPT
    if "image" in data:
        # TODO: This create a large overhead, because the image is being generated twice
        description = get_vision_description(data["image"])
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
                ("ai", description),
                ("human", "{question}"),
            ]
        )

    return prompt


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

    llm = ChatOllama(model=settings["model"], temperature=0)
    prompt = BASE_RAG_PROMPT
    runnable = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    cl.user_session.set("llm", llm)
    cl.user_session.set("runnable", runnable)
    # TODO: add runnable with history
    cl.user_session.set("runnable_with_history", runnable)


def setup_chatbot(settings: Dict[str, Any]):
    llm = ChatOllama(model=settings["model"], temperature=0)

    has_clip = model_has_clip(llm)
    prompt = BASE_CHAT_PROMPT
    if has_clip:
        prompt = _base_prompt_func

    runnable = prompt | llm | StrOutputParser()
    runnable_with_history = RunnableWithMessageHistory(
        runnable,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )
    cl.user_session.set("llm", llm)
    cl.user_session.set("runnable", runnable)
    cl.user_session.set("runnable_with_history", runnable_with_history)
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


def get_vision_description(image):
    llm: ChatOllama = cl.user_session.get("llm")
    content_parts = []
    text_part = {
        "type": "text",
        "text": (
            "You are an assistant tasked with summarizing images for retrieval. ",
            "These summaries will be embedded and used to retrieve the raw image. ",
            "Give a concise summary of the image that is well optimized for retrieval.",
        ),
    }
    image_part = {
        "type": "image_url",
        "image_url": f"data:image/jpeg;base64,{image}",
    }
    content_parts.append(text_part)
    content_parts.append(image_part)
    prompt = [HumanMessage(content=content_parts)]
    description = llm.invoke(prompt)
    return description.content
