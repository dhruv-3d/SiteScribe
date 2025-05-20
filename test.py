import os

from langchain_community.document_loaders.html_bs import BSHTMLLoader
from langchain_community.document_transformers import MarkdownifyTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownTextSplitter

local_source_path = "local_data"
list_of_sources = []


# make a list of all files from the local directory which will be used for doing RAG
for file in os.listdir(local_source_path):
    if file.endswith('.html'):
        file_path = os.path.abspath(os.path.join(local_source_path, file))
        list_of_sources.append(file_path)


# for src in list_of_sources:
src_loader = BSHTMLLoader(list_of_sources[1])
docs = src_loader.load()

md = MarkdownifyTransformer(strip=["a", "svg", "script"])
converted_docs = md.transform_documents(docs)

print(len(converted_docs[0].page_content))

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
text_splitter = MarkdownTextSplitter()
splits = text_splitter.split_documents(converted_docs)

