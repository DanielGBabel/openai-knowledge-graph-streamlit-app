from langchain_openai import OpenAIEmbeddings
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.graphs import Neo4jGraph
from streamlit_agraph import agraph, Node, Edge, Config
from neo4j import GraphDatabase
import os
from openai import OpenAI

# Function to process the query and return a response
def process_query(query):
    # Use GraphCypherQAChain to get a Cypher query and a natural language response
    result = cypher_chain(query)
    intermediate_steps = result['intermediate_steps']
    final_answer = result['result']
    generated_cypher = intermediate_steps[0]['query']
    response_structured = final_answer
    
    # Fetch graph data using the Cypher query
    nodes, edges = fetch_graph_data(nodesType=None, relType=None, direct_cypher_query=generated_cypher, intermediate_steps=intermediate_steps)
    
    return response_structured, nodes, edges

# Function to fetch data from Neo4j
def fetch_graph_data(nodesType=None, relType=None, direct_cypher_query=None, intermediate_steps=None):
    # Use the direct Cypher query if provided
    if direct_cypher_query:
        context = intermediate_steps[1]['context']
        nodes, edges = process_graph_result(context)
    else:
        if nodesType or relType:
            # Construct the Cypher query based on selected filters
            cypher_query = construct_cypher_query(nodesType, relType)
            with GraphDatabase.driver(os.environ["NEO4J_URI"], 
                                    auth=(os.environ["NEO4J_USERNAME"], 
                                            os.environ["NEO4J_PASSWORD"])).session() as session:
                result = session.run(cypher_query)
                nodes, edges = process_graph_result_select(result)
    
    return nodes, edges


# Function to construct the Cypher query based on selected filters
def construct_cypher_query(node_types, rel_types):
    # Create a list of MATCH clauses for node types
    node_clauses = []
    for node_type in node_types:
        node_clauses.append(f"(p:{node_type})-[r]->(n) ")

    # Create a list of WHERE clauses for relationship types
    rel_clauses = []
    for rel_type in rel_types:
        rel_clauses.append(f"type(r)='{rel_type}' ")

    # Combine the clauses into one Cypher query
    if rel_clauses:
        rel_match = " OR ".join(rel_clauses)
        query = f"MATCH {' OR '.join(node_clauses)} WHERE {rel_match} RETURN p, r, n"
    else:
        query = f"MATCH {' OR '.join(node_clauses)} RETURN p, r, n"
    
    return query

def process_graph_result(result):
    nodes = []
    edges = []
    node_names = set()  # This defines node_names to track unique nodes

    for record in result: 
        # Process nodes
        p_name = record['p.name']
        o_name = record['o.name']

        # Add nodes if they don't already exist
        if p_name not in node_names:
            nodes.append(Node(id=p_name, label=p_name, size=5, shape="circle"))
            node_names.add(p_name)
        if o_name not in node_names:
            nodes.append(Node(id=o_name, label=o_name, size=5, shape="circle"))
            node_names.add(o_name)

        # Process edges
        relationship_label = record['type(r)']
        edges.append(Edge(source=p_name, target=o_name, label=relationship_label))

    return nodes, edges

def process_graph_result_select(result):
    nodes = []
    edges = []
    node_names = set()  # This defines node_names to track unique nodes

    for record in result: 
        # Process nodes
        p = record['p']
        n = record['n']
        p_name = p['name']
        n_name = n['name']

       # Add nodes if they don't already exist
        if p_name not in node_names:
            nodes.append(Node(id=p_name, label=p_name, size=5, shape="circle"))
            node_names.add(p_name)
        if n_name not in node_names:
            nodes.append(Node(id=n_name, label=n_name, size=5, shape="circle"))
            node_names.add(n_name)

        # Process edges, include the date in the label if it exists
        r = record['r']
        relationship_label = r.type
        if 'date' in r:
            relationship_label = f"{r.type} ({r['date']})"
        edges.append(Edge(source=p_name, target=n_name, label=relationship_label))
    
    return nodes, edges

# from langchain.agents import initialize_agent
st.title("Santander Generales.\nSEGURO MI HOGAR SANTANDER.\nMI PÓLIZA")

NEO4J_URI= st.secrets["NEO4J_URI"]
NEO4J_USERNAME= st.secrets["NEO4J_USERNAME"]
NEO4J_PASSWORD= st.secrets["NEO4J_PASSWORD"]

graph = Neo4jGraph(
    url=os.environ["NEO4J_URI"],
    username=os.environ["NEO4J_USERNAME"],
    password=os.environ["NEO4J_PASSWORD"])

# Fetch the unique node types and relationship types for sidebar filters
node_types = ['__Entity__','Document']
relationship_types = []


with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="langchain_search_api_key_openai", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"

def combine_contexts(structured, unstructured, client):
    
    messages = [{'role': 'system', 'content': 'You are an assistant of an advanced retrieval augmented system,\
                 who prioritizes accuracy and is very context-aware.\
                 Pleass summarize text from the following and generate\
                 a comprehensive, logical and context_aware answer.'},
                {'role': 'user', 'content': structured + unstructured}]
    completion = client.chat.completions.create(model="gpt-4-turbo",
                                                messages=messages,
                                                temperature=0)
    response = completion.choices[0].message.content
    
    return response

# Initialize OpenAI API key and Chat model
if openai_api_key:
    client = OpenAI(api_key=openai_api_key)
    os.environ["OPENAI_API_KEY"] = openai_api_key
    from retrievers import final_retriever
   
# Chat interface
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hola, hazme una pregunta sobre tu póliza de seguro."}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="Pregunta sobre tu poliza de seguro"):
    if not openai_api_key:
        st.error("Please add your OpenAI API key to continue.")
    else:
        # Display response
        # Initialize the GraphCypherQAChain from chain.py
        from langchain.chains import GraphCypherQAChain
        cypher_chain = GraphCypherQAChain.from_llm(
            cypher_llm=ChatOpenAI(temperature=0, model_name='gpt-4-turbo', api_key=openai_api_key),
            qa_llm=ChatOpenAI(temperature=0, api_key=openai_api_key),
            graph=graph,
            verbose=True,
            return_intermediate_steps=True
)
        from langchain_community.vectorstores import Neo4jVector
        from chain import invoke_chain
       
        
        # Update session state with new message
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        # Process the query and return the response
        config = Config(height=600, width=800, directed=True, nodeHighlightBehavior=True, highlightColor="#F7A7A6")
        final_ans = invoke_chain(question=prompt,chat_history=st.session_state.messages)
        st.session_state.messages.append({"role": "assistant", "content": final_ans})
        st.chat_message("assistant").write(final_ans)

