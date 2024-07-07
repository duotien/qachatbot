import chainlit as cl
from qachatbot.commands import commands

from langchain.chat_models.ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig

@cl.on_chat_start
def on_chat_start():
    model = ChatOllama(model="phi3")
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
    print("A new chat session has started!")


@cl.on_chat_end
def on_chat_end():
    print("The user disconnected!")


@cl.on_message
async def on_message(message: cl.Message):
    content = message.content

    response = ""

    if content.startswith("/"):
        # TODO: refactor into a function called `process_command()`
        content = content.strip()
        cmd = content.split()
        if cmd[0] == "/tp":
            if len(cmd) != 5:
                response = "Wrong syntax!"
            else:
                response = commands.tp(cmd[1], cmd[2], cmd[3], cmd[4])
    else: # add chatbot here
        runnable = cl.user_session.get("runnable")  # type: Runnable

        msg = cl.Message(content="")

        async for chunk in runnable.astream(
            {"question": message.content},
            config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
        ):
            await msg.stream_token(chunk)

        await msg.send()
        # response = f"Recieved: {content}"

    await cl.Message(response).send()


if __name__ == "__main__":
    from chainlit.cli import run_chainlit, config

    config.run.watch = True
    config.run.headless = True
    config.run.debug = False
    run_chainlit(__file__)

