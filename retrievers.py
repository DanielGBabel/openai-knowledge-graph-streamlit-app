from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Neo4jVector
from typing import Tuple, List, Optional
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.graphs import Neo4jGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_openai import ChatOpenAI
import streamlit as st
import os


# NEO4J_URI= st.secrets["NEO4J_URI"]
# NEO4J_USERNAME= st.secrets["NEO4J_USERNAME"]
# NEO4J_PASSWORD= st.secrets["NEO4J_PASSWORD"]

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Neo4jVector
import os

# Declare global variables for the retrievers
# typical_rag = None
# parent_vectorstore = None
# hypothetic_question_vectorstore = None
# summary_vectorstore = None

graph = Neo4jGraph();
llm = ChatOpenAI(temperature=0.2, model_name="gpt-4-0125-preview")
vector_index = Neo4jVector.from_existing_graph(
        OpenAIEmbeddings(),
        search_type="hybrid",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding"
        
)       
class Entities(BaseModel):
    """Identifying information about entities."""

    names: List[str] = Field(
        ...,
        description="All the person, organization, or business entities that "
        "appear in the text",
    )

def get_entity_chain():
    
    graph.query(
    "CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")

    # Extract entities from text


    prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are extracting any kind of domain entities from the text. (Please consider that some questions can be in spanish)",
        ),
        (
            "human",
            "Use the given format to extract information from the following"
            "input: {question}",
        ),
    ]
    )

    entity_chain = prompt | llm.with_structured_output(Entities)
    
    return entity_chain



def generate_full_text_query(input: str) -> str:
    """
    Generate a full-text search query for a given input string.

    This function constructs a query string suitable for a full-text search.
    It processes the input string by splitting it into words and appending a
    similarity threshold (~2 changed characters) to each word, then combines
    them using the AND operator. Useful for mapping entities from user questions
    to database values, and allows for some misspelings.
    """
    full_text_query = ""
    words = [el for el in remove_lucene_chars(input).split() if el]
    for word in words[:-1]:
        full_text_query += f" {word}~2 AND"
    full_text_query += f" {words[-1]}~2"
    return full_text_query.strip()


def structured_retriever(question: str) -> str:
    """
    Collects the neighborhood of entities mentioned
    in the question
    """
   
    entity_chain = get_entity_chain()
    result = ""
    entities = entity_chain.invoke({"question": question})
    print('my question: ', question)
    print('my entities: ', entities)
    for entity in entities.names:
        response = graph.query(
            """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
            YIELD node,score
            CALL {
              MATCH (node)-[r:!MENTIONS]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
              UNION
              MATCH (node)<-[r:!MENTIONS]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
            }
            RETURN output LIMIT 50
            """,
            {"query": generate_full_text_query(entity)},
        )
        result += "\n".join([el['output'] for el in response])
    return result



def final_retriever(question: str):
    print(f"Search query: {question}")
    structured_data = structured_retriever(question)
    unstructured_data = [el.page_content for el in vector_index.similarity_search(question)]
    final_data = f"""Structured data:
    {structured_data}
    Unstructured data:
    {"#Document ". join(unstructured_data)}
    """
    return final_data


def initialize_retrievers(openai_api_key):
    # global typical_rag, parent_vectorstore, hypothetic_question_vectorstore, summary_vectorstore

    os.environ["OPENAI_API_KEY"] = openai_api_key

    # NEO4J_URI= st.secrets["NEO4J_URI"]
    # NEO4J_USERNAME= st.secrets["NEO4J_USERNAME"]
    # NEO4J_PASSWORD= st.secrets["NEO4J_PASSWORD"]
   
    # graph = Neo4jGraph(
    #     url=os.environ["NEO4J_URI"],
    #     username=os.environ["NEO4J_USERNAME"],
    #     password=os.environ["NEO4J_PASSWORD"])

    # Initialize typical_rag
    typical_rag = Neo4jVector.from_existing_index(
        OpenAIEmbeddings(), index_name="vector")

    # Initialize parent_vectorstore
    
    parent_query = """
    MATCH (node)<-[:HAS_CHILD]-(parent)
    WITH parent, max(score) AS score // deduplicate parents
    RETURN parent.text AS text, score, {} AS metadata LIMIT 1
    """
    parent_vectorstore = Neo4jVector.from_existing_index(
        OpenAIEmbeddings(),
        index_name="parent_document",
        retrieval_query=parent_query,
    )

    # Initialize hypothetic_question_vectorstore
    hypothetic_question_query = """
    MATCH (node)<-[:HAS_QUESTION]-(parent)
    WITH parent, max(score) AS score // deduplicate parents
    RETURN parent.text AS text, score, {} AS metadata
    """
    hypothetic_question_vectorstore = Neo4jVector.from_existing_index(
        OpenAIEmbeddings(),
        index_name="hypothetical_questions",
        retrieval_query=hypothetic_question_query,
    )

    # Initialize summary_vectorstore
    summary_query = """
    MATCH (node)<-[:HAS_SUMMARY]-(parent)
    WITH parent, max(score) AS score // deduplicate parents
    RETURN parent.text AS text, score, {} AS metadata
    """
    summary_vectorstore = Neo4jVector.from_existing_index(
        OpenAIEmbeddings(),
        index_name="summary",
        retrieval_query=summary_query,
    )

    return typical_rag, parent_vectorstore, hypothetic_question_vectorstore, summary_vectorstore

