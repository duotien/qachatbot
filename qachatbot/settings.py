# This file handle global variables and objects
import chainlit as cl
from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings
from qachatbot.utils.vectorstore import VectorStoreManager

embedding_function = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vectorstore_manager = VectorStoreManager(embedding_function)
store_session = {}
