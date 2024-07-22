from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain.document_loaders import WebBaseLoader
from qachatbot.settings import embedding_function


def tp(name, x, y, z):
    return f"teleported {name} to {x}, {y}, {z} !"


def ingest(x):
    web_paths = (x,)
    loader = WebBaseLoader(
        web_paths=web_paths,
    )

    docs = loader.load()

    import re

    def clean_text(text: str):
        cleaned = re.sub("\n+", "\n", text)
        cleaned = re.sub("\t+", "\t", cleaned)
        cleaned = cleaned.strip()
        return cleaned

    for doc in docs:
        doc.page_content = clean_text(doc.page_content)
    print(docs)

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)

    # TODO: xu ly phan nay sau !

    # vectorstore = Chroma.from_documents(documents=all_splits, embedding=embedding_function)

    # # retrieve
    # retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # prompt = ChatPromptTemplate.from_messages(
    #     [
    #         (
    #             "human",
    #             (
    #                 "You are an assistant for question-answering tasks. "
    #                 "Use the following pieces of retrieved context to answer the question. "
    #                 "If you don't know the answer, just say that you don't know. "
    #                 "Use three sentences maximum and keep the answer concise. \n"
    #                 "Question: {question} \n"
    #                 "Context: {context} \n"
    #                 "Answer:"
    #             ),
    #         ),
    #     ]
    # )
    # example_messages = prompt.invoke({
    # "context": "filler context",
    # "question": "filler question"
    # }).to_messages()
    # print(example_messages[0].content)

    return docs
