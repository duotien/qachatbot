import chainlit as cl
from qachatbot.commands import commands

@cl.on_chat_start
def on_chat_start():
    print("A new chat session has started!")

@cl.on_chat_end
def on_chat_end():
    print("The user disconnected!")

@cl.on_message
async def on_message(message: cl.Message):
    response = message.content
    response2 = ""
    # /help 
    # /tp @p 12 13 14

    if response[0] == '/':
        response2 = "this is a cmd line !"
        response = response.strip()
        a = response.split(" ")
        await cl.Message(a).send()
        if a[0] == "/tp":
            if len(a) != 5:
                response2 = "Wrong syntax!"
            else:
                response2 = commands.tp(a[1], a[2], a[3], a[4])
    await cl.Message(response2).send()