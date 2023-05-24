#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import os
import argparse

load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from constants import CHROMA_SETTINGS

def main():
    # Parse the command line arguments
    args = parse_arguments()
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    # Prepare the LLM
    if model_type == "LlamaCpp":
        llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, callbacks=callbacks, verbose=False,n_threads=8)
    elif model_type == "GPT4All":
        llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', callbacks=callbacks, verbose=False)
    else:
        print(f"Model {model_type} not supported!")
        exit;



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

    from langchain import PromptTemplate
    refine_prompt = PromptTemplate(
        input_variables=["question", "existing_answer", "context_str"],
        template=refine_prompt_template,
    )
    initial_qa_prompt = PromptTemplate(
        input_variables=["context_str", "question"],
        template=initial_qa_template,
    )
    chain_type_kwargs = {"question_prompt": initial_qa_prompt, "refine_prompt": refine_prompt}
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="refine",
        retriever=retriever, return_source_documents= not args.hide_source,
        chain_type_kwargs=chain_type_kwargs)


    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break

        # Get the answer from the chain
        res = qa(query)
        answer, docs = res['result'], [] if args.hide_source else res['source_documents']

        # Print the result
        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(answer)

        # Print the relevant sources used for the answer
        for document in docs:
            print("\n> " + document.metadata["source"] + ":")
            print(document.page_content)

def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()


if __name__ == "__main__":
    main()
