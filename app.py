import os
import chainlit as cl
import asyncio
from chainlit.input_widget import Select, Switch, Slider
from qachatbot.bot.chat import process_command, process_response

from langchain.chat_models.ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.schema import StrOutputParser
from langchain_chroma import Chroma
from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings

# from langchain.schema.runnable import Runnable
# from langchain.schema.runnable.config import RunnableConfig

PROJECT_DIR = os.path.dirname(__file__)
PERSIST_DIR = os.path.join(PROJECT_DIR, ".chroma")

chat_history = []


@cl.on_chat_start
async def on_chat_start():
    async def init_setting():
        settings = await cl.ChatSettings(
            [
                Select(
                    id="Model",
                    label="Model",
                    values=["phi3", "llama3"],
                    initial_index=0,
                )
            ]
        ).send()
        return settings

    settings = asyncio.run(init_setting())
    md = settings["Model"]
    model = ChatOllama(model=md)
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
    cl.user_session.set("prompt", prompt)
    cl.user_session.set("runnable", runnable)

    embedding_function = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR, embedding_function=embedding_function
    )
    cl.user_session.set("vectorstore", vectorstore)
    cl.user_session.set("chat_mode", "rag")

    print("A new chat session has started!")


@cl.on_chat_end
def on_chat_end():
    print("The user disconnected!")


# TODO: add button to change k
def invoke_rag(user_input: str, k=3):
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
            response = invoke_rag(message.content)
            msg = cl.Message(content=response.content)
            await msg.send()


@cl.on_settings_update
async def setup_agent(settings):
    # print("on_settings_update", settings)
    md = settings["Model"]
    model = ChatOllama(model=md)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a double agent working for both CIA and KGB, your task is to help the user with whatever they ask, be polite and elegant like a true spy",
            ),
            ("human", "{question}"),
        ]
    )
    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)


if __name__ == "__main__":
    from chainlit.cli import run_chainlit, config

    config.run.watch = True
    config.run.headless = True
    config.run.debug = False
    run_chainlit(__file__)
