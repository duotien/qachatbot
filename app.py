import os

import chainlit as cl
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import StrOutputParser
from langchain_chroma import Chroma
from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain_core.messages import AIMessage, HumanMessage

from qachatbot.bot.chat import (
    init_settings,
    process_command,
    process_rag,
    process_response,
    setup_chatbot,
    setup_qabot,
)
from qachatbot.settings import (
    chat_history,
    vectorstore_manager,
)


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
            response = await process_rag(message.content)
            # todo: do something with response or remove it


@cl.on_settings_update
async def setup_agent(settings):
    # print("settings:", settings)
    chat_mode = settings["chat_mode"]
    cl.user_session.set("chat_mode", chat_mode)
    # print("DB Mode", cl.user_session.get("DB"))
    match settings['DB']:
        case "Chroma":
            cl.user_session.set("vectorstore", vectorstore_manager.chromadb)
        case "Markdown":
            cl.user_session.set("vectorstore", vectorstore_manager.markdown_chromadb)
        case _:
            # TODO: add another database here
            cl.user_session.set("vectorstore", vectorstore_manager.chromadb)

    match chat_mode:
        case "chat":
            setup_chatbot(settings)
        case "rag":
            setup_qabot(settings)
        case _:
            setup_chatbot(settings)

