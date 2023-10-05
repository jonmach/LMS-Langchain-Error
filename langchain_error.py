from langchain.chains.summarize import load_summarize_chain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import LLMChain
from langchain.chains import ReduceDocumentsChain
from langchain.chains import MapReduceDocumentsChain
from langchain.text_splitter import TokenTextSplitter
from langchain.document_loaders import WebBaseLoader
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

import openai

openai.api_base='http://localhost:1234/v1'

llm = OpenAI(openai_api_key='...', max_tokens=2000, model="text-davinci-003", temperature=0.3)

def summariseDoc (llm):

    # Map Chain
    map_template = """
        ### Instruction
        You are an expert in summarizing documents.

        Write a summary of the following content:
        
        <<<{content}>>>

        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.

        Summary:\n
        ###Response:
        """

    map_prompt = PromptTemplate.from_template(map_template)
    map_chain = LLMChain(prompt=map_prompt, llm=llm)

    # Reduce Chain
    reduce_template = """
        ### Instruction

        The following is a set of summaries:

        <<<{doc_summaries}>>>
        
        Provide a summary of the above summaries with all the key details.  If you feel like you don't 
        have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.

        Summary:\n
        ###Response:
        """
    
    reduce_prompt = PromptTemplate.from_template(reduce_template)
    reduce_chain = LLMChain(prompt=reduce_prompt, llm=llm)
    
    stuff_chain = StuffDocumentsChain(llm_chain=reduce_chain, document_variable_name="doc_summaries")
    
    reduce_chain = ReduceDocumentsChain(combine_documents_chain=stuff_chain, token_max = 2000)

    # Map Reduce Chain
    map_reduce_chain = MapReduceDocumentsChain(
        llm_chain=map_chain,
        document_variable_name="content",
        reduce_documents_chain=reduce_chain
    )
    
    # Load Content from web
    loader = WebBaseLoader('https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/')
    source_docs = loader.load()
    splitter = TokenTextSplitter(chunk_size=2000)
    docs = splitter.split_documents(source_docs)
    
    summary = map_reduce_chain(docs)
    return summary


#doc = summariseDoc(llm, 'SourceFiles/AI-Reflections-2019.pdf')
doc = summariseDoc(llm=llm)
print(doc)
