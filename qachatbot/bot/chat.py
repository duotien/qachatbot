import chainlit as cl
from qachatbot.commands import commands

from langchain.schema.runnable.config import RunnableConfig
from langchain.schema.runnable import Runnable

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
        # print(chunk)
        await msg.stream_token(chunk)
    await msg.send()
    return msg.content
        # response = f"Recieved: {content}"

