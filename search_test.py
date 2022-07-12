import haystack
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.utils import clean_wiki_text, convert_files_to_docs, fetch_archive_from_http, print_answers
from haystack.nodes import FARMReader, TransformersReader
from haystack.utils import launch_es
from haystack.document_stores import InMemoryDocumentStore
from haystack.pipelines import ExtractiveQAPipeline
from pprint import pprint
from haystack.pipelines import ExtractiveQAPipeline
from haystack.nodes import TfidfRetriever
import sys


def main():

    document_store = InMemoryDocumentStore()

    #launch_es()

    #document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="document")

    doc_dir = "data-civil-code"
    s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt1.zip"
    fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

    docs = convert_files_to_docs(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)

    #print(docs[:3])
    # Now, let's write the docs to our DB.
    document_store.write_documents(docs)

    retriever = TfidfRetriever(document_store=document_store)
    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)


    pipe = ExtractiveQAPipeline(reader, retriever)

    prediction = pipe.run(
        query="In which cases payments of social insurance will be included into the amount of the repairable damage?", params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}}
    )

    original_stdout = sys.stdout
    with open('answer.txt', 'w') as f:
        sys.stdout = f # Change the standard output to the file we created.
        print(prediction)
  #  print_answers(prediction, details="minimum")


if __name__ == "__main__":
    main()