# Transaction Attribute Extraction Workflow using LangGraph

from langgraph.graph import StateGraph, END
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import Dict, List
import os

# === SETUP ===
os.environ["OPENAI_API_KEY"] = "sk-..."  # Your API key here

# === SCHEMA ===
ATTRIBUTE_SCHEMA = [
    {"name": "governing_law", "description": "Jurisdiction or governing law of the agreement"},
    {"name": "termination_notice_period", "description": "Minimum days required for termination notice"},
    {"name": "payment_terms", "description": "When and how payments are to be made"},
    # Add more attributes as needed
]

# === STEP 1: QUERY DECOMPOSITION AGENT ===
def query_decomposer(state: Dict):
    queries = []
    for attr in ATTRIBUTE_SCHEMA:
        queries.append({
            "attribute": attr["name"],
            "query": f"What is the {attr['description']}?"
        })
    state["queries"] = queries
    return state

# === STEP 2: DOCUMENT INGESTION + CHUNKING ===
def ingest_and_chunk(state: Dict):
    loader = PyMuPDFLoader(state["document_path"])
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    state["chunks"] = chunks
    return state

# === STEP 3: VECTOR STORE SETUP ===
def embed_chunks(state: Dict):
    embedding_model = OpenAIEmbeddings()
    vectordb = FAISS.from_documents(state["chunks"], embedding_model)
    state["vectordb"] = vectordb
    return state

# === STEP 4: RELEVANT CHUNK RETRIEVER ===
def retrieve_chunks(state: Dict):
    retriever = state["vectordb"].as_retriever(search_kwargs={"k": 5})
    query_chunks = {}
    for q in state["queries"]:
        docs = retriever.get_relevant_documents(q["query"])
        query_chunks[q["attribute"]] = docs
    state["query_chunks"] = query_chunks
    return state

# === STEP 5: ATTRIBUTE EXTRACTION AGENT ===
def extract_attributes(state: Dict):
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    extracted_values = {}
    for attr, docs in state["query_chunks"].items():
        context = "\n\n".join([d.page_content for d in docs])
        prompt = ChatPromptTemplate.from_template(
            """
            Extract the value for the attribute: {attr} from the below context:
            ----
            {context}
            ----
            Answer:
            """
        )
        chain = prompt | llm
        answer = chain.invoke({"attr": attr, "context": context})
        extracted_values[attr] = answer.content.strip()
    state["extracted_values"] = extracted_values
    return state

# === STEP 6: VALIDATION AGENT (Optional Rule-based Checks) ===
def validate_attributes(state: Dict):
    validated = {}
    for k, v in state["extracted_values"].items():
        if not v or "not found" in v.lower():
            validated[k] = {"value": v, "status": "uncertain"}
        else:
            validated[k] = {"value": v, "status": "confident"}
    state["validated_attributes"] = validated
    return state

# === GRAPH WORKFLOW ===
graph = StateGraph()
graph.add_node("query_decomposer", query_decomposer)
graph.add_node("ingest_and_chunk", ingest_and_chunk)
graph.add_node("embed_chunks", embed_chunks)
graph.add_node("retrieve_chunks", retrieve_chunks)
graph.add_node("extract_attributes", extract_attributes)
graph.add_node("validate_attributes", validate_attributes)

# === EDGES ===
graph.set_entry_point("query_decomposer")
graph.add_edge("query_decomposer", "ingest_and_chunk")
graph.add_edge("ingest_and_chunk", "embed_chunks")
graph.add_edge("embed_chunks", "retrieve_chunks")
graph.add_edge("retrieve_chunks", "extract_attributes")
graph.add_edge("extract_attributes", "validate_attributes")
graph.add_edge("validate_attributes", END)

app = graph.compile()

# === EXECUTION ===
example_state = {
    "document_path": "./agreements/sample_agreement.pdf"
}

result = app.invoke(example_state)
print(result["validated_attributes"])



from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain

def query_decomposer(state: Dict):
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.3)

    # HyDE-style prompt
    hyde_prompt = PromptTemplate.from_template(
        """
        You are an expert legal assistant. Given the attribute: "{attribute_name}", and its description: "{attribute_description}",
        generate a hypothetical answer as if it were found in a real-world agreement document.

        Format your answer in complete sentence(s) using realistic legal phrasing.

        Hypothetical Answer:
        """
    )

    hyde_chain = LLMChain(llm=llm, prompt=hyde_prompt)

    queries = []
    for attr in ATTRIBUTE_SCHEMA:
        hypothetical_answer = hyde_chain.run({
            "attribute_name": attr["name"],
            "attribute_description": attr["description"]
        })
        enriched_query = f"{attr['description']}. For example, {hypothetical_answer.strip()}"
        queries.append({
            "attribute": attr["name"],
            "query": enriched_query
        })

    state["queries"] = queries
    return state




### Extractor

# Transaction Attribute Extraction Workflow using LangGraph

from langgraph.graph import StateGraph, END
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.llm import LLMChain
from typing import Dict, List
import os

# === SETUP ===
os.environ["OPENAI_API_KEY"] = "sk-..."  # Your API key here

# === SCHEMA ===
ATTRIBUTE_SCHEMA = [
    {"name": "governing_law", "description": "Jurisdiction or governing law of the agreement"},
    {"name": "termination_notice_period", "description": "Minimum days required for termination notice"},
    {"name": "payment_terms", "description": "When and how payments are to be made"},
    # Add more attributes as needed
]

# === STEP 1: QUERY DECOMPOSITION AGENT with HyDE and Multi-query ===
def query_decomposer(state: Dict):
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.3)
    hyde_prompt = PromptTemplate.from_template(
        """
        Given the attribute: "{attribute_name}" and description: "{attribute_description}",
        generate a realistic legal-style hypothetical answer and three different ways of asking for this attribute in a legal document.

        Format:
        Hypothetical Answer: <...>
        Queries:
        1. <...>
        2. <...>
        3. <...>
        """
    )

    hyde_chain = LLMChain(llm=llm, prompt=hyde_prompt)
    queries = []

    for attr in ATTRIBUTE_SCHEMA:
        output = hyde_chain.run({
            "attribute_name": attr["name"],
            "attribute_description": attr["description"]
        })

        # Parse output (simple heuristic split)
        parts = output.strip().split("Queries:")
        hypo = parts[0].replace("Hypothetical Answer:", "").strip()
        query_lines = parts[1].strip().split("\n") if len(parts) > 1 else []
        for q in query_lines:
            q_text = q.strip("1234567890. ")
            enriched_query = f"{attr['description']}. For example: {hypo}. Query: {q_text}"
            queries.append({"attribute": attr["name"], "query": enriched_query})

    state["queries"] = queries
    return state

# === STEP 2: DOCUMENT INGESTION + CHUNKING ===
def ingest_and_chunk(state: Dict):
    loader = PyMuPDFLoader(state["document_path"])
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    state["chunks"] = chunks
    return state

# === STEP 3: VECTOR STORE SETUP ===
def embed_chunks(state: Dict):
    embedding_model = OpenAIEmbeddings()
    vectordb = FAISS.from_documents(state["chunks"], embedding_model)
    state["vectordb"] = vectordb
    return state

# === STEP 4: RELEVANT CHUNK RETRIEVER ===
def retrieve_chunks(state: Dict):
    retriever = state["vectordb"].as_retriever(search_kwargs={"k": 5})
    query_chunks = {}
    for q in state["queries"]:
        docs = retriever.get_relevant_documents(q["query"])
        if q["attribute"] not in query_chunks:
            query_chunks[q["attribute"]] = []
        query_chunks[q["attribute"]].extend(docs)
    # Deduplicate chunks per attribute
    for attr in query_chunks:
        seen = set()
        unique_docs = []
        for doc in query_chunks[attr]:
            if doc.page_content not in seen:
                unique_docs.append(doc)
                seen.add(doc.page_content)
        query_chunks[attr] = unique_docs
    state["query_chunks"] = query_chunks
    return state

# === STEP 5: ATTRIBUTE EXTRACTION AGENT ===
def extract_attributes(state: Dict):
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    extracted_values = {}
    for attr, docs in state["query_chunks"].items():
        context = "\n\n".join([d.page_content for d in docs])
        prompt = ChatPromptTemplate.from_template(
            """
            Extract the value for the attribute: {attr} from the below context:
            ----
            {context}
            ----
            Answer:
            """
        )
        chain = prompt | llm
        answer = chain.invoke({"attr": attr, "context": context})
        extracted_values[attr] = answer.content.strip()
    state["extracted_values"] = extracted_values
    return state

# === STEP 6: VALIDATION AGENT (Optional Rule-based Checks) ===
def validate_attributes(state: Dict):
    validated = {}
    for k, v in state["extracted_values"].items():
        if not v or "not found" in v.lower():
            validated[k] = {"value": v, "status": "uncertain"}
        else:
            validated[k] = {"value": v, "status": "confident"}
    state["validated_attributes"] = validated
    return state

# === GRAPH WORKFLOW ===
graph = StateGraph()
graph.add_node("query_decomposer", query_decomposer)
graph.add_node("ingest_and_chunk", ingest_and_chunk)
graph.add_node("embed_chunks", embed_chunks)
graph.add_node("retrieve_chunks", retrieve_chunks)
graph.add_node("extract_attributes", extract_attributes)
graph.add_node("validate_attributes", validate_attributes)

# === EDGES ===
graph.set_entry_point("query_decomposer")
graph.add_edge("query_decomposer", "ingest_and_chunk")
graph.add_edge("ingest_and_chunk", "embed_chunks")
graph.add_edge("embed_chunks", "retrieve_chunks")
graph.add_edge("retrieve_chunks", "extract_attributes")
graph.add_edge("extract_attributes", "validate_attributes")
graph.add_edge("validate_attributes", END)

app = graph.compile()

# === EXECUTION ===
example_state = {
    "document_path": "./agreements/sample_agreement.pdf"
}

result = app.invoke(example_state)
print(result["validated_attributes"])


### Agent

# Transaction Attribute Extraction Workflow using LangGraph

from langgraph import StateGraph, END
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import Dict, List
import os

# === SETUP ===
os.environ["OPENAI_API_KEY"] = "sk-..."  # Your API key here

# === SCHEMA ===
ATTRIBUTE_SCHEMA = [
    {"name": "governing_law", "description": "Jurisdiction or governing law of the agreement"},
    {"name": "termination_notice_period", "description": "Minimum days required for termination notice"},
    {"name": "payment_terms", "description": "When and how payments are to be made"},
    # Add more attributes as needed
]

# === STEP 1: QUERY DECOMPOSITION AGENT ===
def query_decomposer(state: Dict):
    queries = []
    for attr in ATTRIBUTE_SCHEMA:
        queries.append({
            "attribute": attr["name"],
            "query": f"What is the {attr['description']}?"
        })
    state["queries"] = queries
    return state

# === STEP 2: DOCUMENT INGESTION + CHUNKING ===
def ingest_and_chunk(state: Dict):
    loader = PyMuPDFLoader(state["document_path"])
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    state["chunks"] = chunks
    return state

# === STEP 3: VECTOR STORE SETUP ===
def embed_chunks(state: Dict):
    embedding_model = OpenAIEmbeddings()
    vectordb = FAISS.from_documents(state["chunks"], embedding_model)
    state["vectordb"] = vectordb
    return state

# === STEP 4: RELEVANT CHUNK RETRIEVER ===
def retrieve_chunks(state: Dict):
    retriever = state["vectordb"].as_retriever(search_kwargs={"k": 5})
    query_chunks = {}
    for q in state["queries"]:
        docs = retriever.get_relevant_documents(q["query"])
        query_chunks[q["attribute"]] = docs
    state["query_chunks"] = query_chunks
    return state

# === STEP 5: ATTRIBUTE EXTRACTION AGENT ===
def extract_attributes(state: Dict):
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    extracted_values = {}
    for attr, docs in state["query_chunks"].items():
        context = "\n\n".join([d.page_content for d in docs])
        prompt = ChatPromptTemplate.from_template(
            """
            Extract the value for the attribute: {attr} from the below context:
            ----
            {context}
            ----
            Answer:
            """
        )
        chain = prompt | llm
        answer = chain.invoke({"attr": attr, "context": context})
        extracted_values[attr] = answer.content.strip()
    state["extracted_values"] = extracted_values
    return state

# === STEP 6: VALIDATION AGENT USING LLM ===
def validate_attributes(state: Dict):
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    validated = {}
    for attr, value in state["extracted_values"].items():
        context = "\n\n".join([d.page_content for d in state["query_chunks"][attr]])
        prompt = ChatPromptTemplate.from_template(
            """
            Based on the following extracted answer and the original context, determine if the answer is confidently correct.

            Attribute: {attr}
            Extracted Answer: {value}

            Context:
            ----
            {context}
            ----

            Answer with either:
            - CONFIDENT: [reason]
            - UNCERTAIN: [reason]
            """
        )
        chain = prompt | llm
        answer = chain.invoke({"attr": attr, "value": value, "context": context})
        validated[attr] = {"value": value, "status": answer.content.strip()}
    state["validated_attributes"] = validated
    return state

# === GRAPH WORKFLOW ===
graph = StateGraph()
graph.add_node("query_decomposer", query_decomposer)
graph.add_node("ingest_and_chunk", ingest_and_chunk)
graph.add_node("embed_chunks", embed_chunks)
graph.add_node("retrieve_chunks", retrieve_chunks)
graph.add_node("extract_attributes", extract_attributes)
graph.add_node("validate_attributes", validate_attributes)

# === EDGES ===
graph.set_entry_point("query_decomposer")
graph.add_edge("query_decomposer", "ingest_and_chunk")
graph.add_edge("ingest_and_chunk", "embed_chunks")
graph.add_edge("embed_chunks", "retrieve_chunks")
graph.add_edge("retrieve_chunks", "extract_attributes")
graph.add_edge("extract_attributes", "validate_attributes")
graph.add_edge("validate_attributes", END)

app = graph.compile()

# === EXECUTION ===
example_state = {
    "document_path": "./agreements/sample_agreement.pdf"
}

result = app.invoke(example_state)
print(result["validated_attributes"])
