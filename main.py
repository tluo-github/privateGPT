import copy
import asyncio
import requests

from fastapi import FastAPI, Request
from llama_cpp import Llama
from sse_starlette import EventSourceResponse


from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import LlamaCpp

from chromadb.config import Settings
from langchain import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')



app = FastAPI()

@app.get("/")
async def hello():
    return {"hello":"wold"}

@app.get("/model")
async def model():
    print("Loading model...")
    llm = Llama(model_path=model_path)
    stream = llm(
        "Question: 请介绍一下中国首都  Answer: ",
        max_tokens=100,
        stop=["Q:", "\n"],
        echo=True,
    )
    result = copy.deepcopy(stream)
    return {"result":result}


@app.get("/jokes")
async def jokes(request: Request):
    def get_messages():
        url = "https://official-joke-api.appspot.com/random_ten"
        response = requests.get(url)
        if response.status_code == 200:
            jokes = response.json()
            messages = []
            for joke in jokes:
                setup = joke['setup']
                punchline = joke['punchline']
                message = f"{setup} {punchline}"
                messages.append(message)
            return messages
        else:
            return None

    async def sse_event():
        while True:
            if await request.is_disconnected():
                break

            for message in get_messages():
                yield {"data": message}

            await asyncio.sleep(1)

    return EventSourceResponse(sse_event())


@app.get("/llama")
async def llama(request: Request):
    print("Loading model...")
    llm = Llama(model_path=model_path)
    stream = llm(
        "Question: 请介绍一下中国首都 Answer: ",
        max_tokens=100,
        stop=["\n", " Q:"],
        stream=True,
    )

    async def async_generator():
        for item in stream:
            yield item

    async def server_sent_events():
        async for item in async_generator():
            if await request.is_disconnected():
                break

            result = copy.deepcopy(item)
            text = result["choices"][0]

            yield {"data": text}

    return EventSourceResponse(server_sent_events())


CHROMA_SETTINGS = Settings(
    chroma_db_impl='duckdb+parquet',
    persist_directory=persist_directory,
    anonymized_telemetry=False
)

embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)

retriever = db.as_retriever(search_kwargs={"k": 3})

refine_prompt_template = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n"
    "这是原始问题: {question}\n"
    "已有的回答: {existing_answer}\n"
    "现在还有一些文字，（如果有需要）你可以根据它们完善现有的回答。"
    "\n\n"
    "{context_str}\n"
    "\\nn"
    "请根据新的文段，进一步完善你的回答。\n\n"
    "### Response: "
)

initial_qa_template = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n"
    "以下为背景知识：\n"
    "{context_str}"
    "\n"
    "请根据以上背景知识, 回答这个问题：{question}。\n\n"
    "### Response: "
)

refine_prompt = PromptTemplate(
    input_variables=["question", "existing_answer", "context_str"],
    template=refine_prompt_template,
)
initial_qa_prompt = PromptTemplate(
    input_variables=["context_str", "question"],
    template=initial_qa_template,
)
chain_type_kwargs = {"question_prompt": initial_qa_prompt, "refine_prompt": refine_prompt}

callbacks = []
llm = LlamaCpp(
    model_path=model_path,
    n_ctx=model_n_ctx,
    callbacks=callbacks,
    verbose=False,
    n_threads=8,
    n_gpu_layers=30,
)
print("Model loaded!")
qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="refine",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs)
print("QA loaded!")

@app.get("/static")
async def static(request: Request):

    res = qa("李白的诗是什么风格?")
    answer, docs = res['result'], res['source_documents']
    return {"answer":answer,"docs":docs}

@app.get("/luotao")
async def luotao(request: Request):
    callbacks = [StreamingStdOutCallbackHandler()]
    llm = LlamaCpp(
        model_path=model_path,
        n_ctx=model_n_ctx,
        callbacks=callbacks,
        verbose=False,
        n_threads=30)

    print("Model loaded!")
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="refine",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs)
    print("QA loaded!")

    res = qa("李白的诗是什么风格?")
    answer, docs = res['result'], res['source_documents']

    async def async_generator():
        for item in callbacks:
            yield item

    async def server_sent_events():
        async for item in async_generator():
            if await request.is_disconnected():
                break

            result = copy.deepcopy(item)

            yield {"data": result}

    return EventSourceResponse(server_sent_events())
