import os

import chainlit as cl
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_chroma import Chroma
from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain.schema import StrOutputParser

from qachatbot.bot.chat import (
    init_settings,
    process_command,
    process_rag,
    process_response,
)

from qachatbot.settings import vectorstore_manager, chat_history


@cl.on_chat_start
async def on_chat_start():
    settings = await init_settings()
    await setup_agent(settings)
    print("A new chat session has started!")


@cl.on_chat_end
def on_chat_end():
    print("The user disconnected!")


@cl.on_message
async def on_message(message: cl.Message):
    content = message.content

    response = f"Received: {content}"

    if content.startswith("/"):
        # TODO: refactor into a function called `process_command()`
        response = process_command(content)
        await cl.Message(response).send()

    else:  # add chatbot here
        # TODO: them nut chuyen mode
        chat_mode = cl.user_session.get("chat_mode")
        if chat_mode == "chat":
            try:
                ai_response = await process_response(message, chat_history)
                # print(ai_response)
                chat_history.append(HumanMessage(content=message.content))
                chat_history.append(AIMessage(content=ai_response))

            except Exception as e:
                print(e)
                await cl.Message(response).send()
        if chat_mode == "rag":
            response = process_rag(message.content)
            msg = cl.Message(content=response.content)
            await msg.send()


@cl.on_settings_update
async def setup_agent(settings):
    model = ChatOllama(model=settings["model"])
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
    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("llm", model)
    cl.user_session.set("runnable", runnable)
    cl.user_session.set("chat_mode", settings["chat_mode"])
    
    match (cl.user_session.get("DB")):
        case "Chroma": 
            cl.user_session.set("vectorstore", vectorstore_manager.chromadbd)
        case _:
            # TODO: add another database here
            cl.user_session.set("vectorstore", vectorstore_manager.chromadbd)
            
