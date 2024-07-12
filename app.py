import chainlit as cl
from qachatbot.commands import commands
from qachatbot.bot.chat import process_command, process_response

from langchain.chat_models.ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
# from langchain.schema.runnable import Runnable
# from langchain.schema.runnable.config import RunnableConfig


from chainlit.input_widget import Select, Switch, Slider

@cl.on_chat_start
async def on_chat_start():
    settings = await init_setting()
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
    
    else: # add chatbot here
        try:
            await process_response(message)
        except Exception as e:
            print(e)
            await cl.Message(response).send()


if __name__ == "__main__":
    from chainlit.cli import run_chainlit, config

    config.run.watch = True
    config.run.headless = True
    config.run.debug = False
    run_chainlit(__file__)


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

@cl.on_settings_update
async def setup_agent(settings):
    # print("on_settings_update", settings)
    md = settings['Model']
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
