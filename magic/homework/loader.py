from langchain_community.document_loaders import PDFPlumberLoader

from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings import HuggingFaceEmbeddings
from os import environ





loader = PDFPlumberLoader("alexnet.pdf")
docs = loader.load()

print("Loaded documents: ", len(docs))
print("First document content: ", docs[0].page_content)
print("First document metadata: ", docs[0].metadata)




text_splitter = SemanticChunker(HuggingFaceEmbeddings())
documents = text_splitter.split_documents(docs)

print("Number of chunks created: ", len(documents))

for i in range(len(documents)):
    print()
    print(f"CHUNK : {i+1}")
    print(documents[i].page_content)

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Instantiate the embedding model
embedder = HuggingFaceEmbeddings()

# Create the vector store 
vector = FAISS.from_documents(documents, embedder)

retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})
retrieved_docs = retriever.invoke("How can you reduce overfitting?")

from langchain_community.llms import Ollama

# Define llm
llm = Ollama(model="mistral")

from langchain.chains import RetrievalQA
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate

prompt = """
1. Use the following pieces of context to answer the question at the end.
2. If you don't know the answer, just say that "I don't know" but don't make up an answer on your own.\n
3. Keep the answer crisp and limited to 3,4 sentences.

Context: {context}

Question: {question}

Helpful Answer:"""


QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt) 

llm_chain = LLMChain(
                  llm=llm, 
                  prompt=QA_CHAIN_PROMPT, 
                  callbacks=None, 
                  verbose=True)

document_prompt = PromptTemplate(
    input_variables=["page_content", "source"],
    template="Context:\ncontent:{page_content}\nsource:{source}",
)

combine_documents_chain = StuffDocumentsChain(
                  llm_chain=llm_chain,
                  document_variable_name="context",
                  document_prompt=document_prompt,
                  callbacks=None,
              )

qa = RetrievalQA(
                  combine_documents_chain=combine_documents_chain,
                  verbose=True,
                  retriever=retriever,
                  return_source_documents=True,
              )
result = qa.invoke("How can you reduce overfitting?")
print(result)