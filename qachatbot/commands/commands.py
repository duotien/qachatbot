import json
from typing import List
import yt_dlp
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chainlit as cl
from langchain_chroma import Chroma
from langchain_core.documents import Document
from qachatbot.settings import (
    vectorstore_manager,
    embedding_function
)

# class MyYoutubeDL(yt_dlp.YoutubeDL):
#     def download(self, url_list):
#         """Download a given list of URLs."""
#         url_list = variadic(url_list)  # Passing a single URL is a common mistake
#         outtmpl = self.params['outtmpl']['default']
#         if (len(url_list) > 1
#                 and outtmpl != '-'
#                 and '%' not in outtmpl
#                 and self.params.get('max_downloads') != 1):
#             raise SameFileError(outtmpl)

#         for url in url_list:
#             self.__download_wrapper(self.extract_info)(
#                 url, force_generic_extractor=self.params.get('force_generic_extractor', False))

#         return self._download_retcode

def yt(URL):
    DOWNLOAD_DIR = "download"
    ydl_opts = {
        "format": "bestvideo[height<=360][ext=mp4][vcodec^=avc]+worstaudio[ext=m4a][language=en]/best[ext=mp4]/best",
        "writesubtitles": True,
        "writeautomaticsub": True,
        # "listsubtitles": True,
        # "subtitlesformat": "srv3",
        "subtitleslangs": ["en"],
        "skip_download": True,
        "outtmpl": "{download}/%(id)s.%(ext)s".format(download=DOWNLOAD_DIR)
    }
    if type(URL) is list:
        URL = URL[0]

    print(f"[DEBUG]: {URL}")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(URL, download=True)
            video_id = info["id"]
            filename = f"{DOWNLOAD_DIR}/{video_id}.en.vtt"

            content = "Cannot read file"
            with open(filename, "r") as fp:
                content = fp.read()
    except Exception as e:
        print(e)
        ydl_opts["format"] = "bestvideo[height<=360][ext=mp4][vcodec^=avc]+worstaudio[ext=m4a]/best[ext=mp4]/best"
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(URL, download=True)
            video_id = info["id"]
            filename = f"{DOWNLOAD_DIR}/{video_id}.en.vtt"

            content = "Cannot read file"
            with open(filename, "r") as fp:
                content = fp.read()

    doc = Document(page_content=content)
    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=1000, chunk_overlap=200, add_start_index=True
    # )
    # all_splits = text_splitter.split_documents([doc])

    user_id = cl.user_session.get("id")
    vectorstore: Chroma = vectorstore_manager.temp_chromadb[user_id]
    vectorstore.add_documents([doc])
    vectorstore.delete_collection

def tp(name, x, y, z):
    return f"teleported {name} to {x}, {y}, {z} !"
