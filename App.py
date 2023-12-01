import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
from neo4j import GraphDatabase
import os

os.environ["NEO4J_URI"] = st.secrets["NEO4J_URI"]
os.environ["NEO4J_USERNAME"] =st.secrets["NEO4J_USERNAME"]
os.environ["NEO4J_PASSWORD"] = st.secrets["NEO4J_PASSWORD"]

import openai
# from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from neo4j_advanced_rag.neo4j_vector import VectorTool
from neo4j_advanced_rag.neo4j_cypher import GraphTool
from neo4j_advanced_rag.history import get_graph_history, save_graph_history
import os
import importlib

# Initialize the LLM
llm = ChatOpenAI()

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

# Function to fetch data from Neo4j
def fetch_graph_data(driver, nodesType, relType):
    cypher_query = construct_cypher_query(nodesType, relType)
    with driver.session() as session:
        result = session.run(cypher_query)
        nodes = []
        edges = []
        node_names = set()  # To keep track of unique node names

        for record in result:
            # Process nodes
            p = record['p']
            n = record['n']
            p_name = p['name']
            n_name = n['name']

            # Add nodes if they don't already exist
            if p_name not in node_names:
                nodes.append(Node(id=p_name, label=p_name, size=25, shape="circle"))
                node_names.add(p_name)

            if n_name not in node_names:
                nodes.append(Node(id=n_name, label=n_name, size=25, shape="circle"))
                node_names.add(n_name)

            # Process edges, include the date in the label if it exists
            r = record['r']
            relationship_label = r.type
            if 'date' in r:
                relationship_label = f"{r.type} ({r['date']})"
            edges.append(Edge(source=p_name, target=n_name, label=relationship_label))

        return nodes, edges

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="langchain_search_api_key_openai", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
openai.api_key = openai_api_key

def get_neo4j_connection():
    return GraphDatabase.driver(url, auth=(username, password))
# Initialize the Neo4j connection
driver = get_neo4j_connection()

# Render the graph
st.title('Neo4j Graph Visualization')
# uri= st.secrets["uri"]
# username= st.secrets["username"]
# password= st.secrets["password"]

# Fetch the unique node types and relationship types for sidebar filters
node_types = ['Person', 'Organization', 'Group', 'Topic']
relationship_types = [
    'BELONGS_TO', 'FORMER_CEO_OF', 'CEO_OF', 'FORMER_MEMBER_OF', 'CURRENT_MEMBER_OF','REMAIN_MEMBER_OF', 'SCHEDULES_CALL_WITH',
    'QUESTIONED_FIRING_SAM', 'FOUNDED_BY', 'INVESTED_IN', 'CONSIDERS_BOARD_SEAT', 'FORMER_CTO_OF', 'INFORMED_OF_FIRING', 'FIRED_AS_CEO',
    'ALL_HANDS_MEETING', 'RESIGNS_FROM', 'APPOINTED_INTERIM_CEO', 'JOINS_MICROSOFT', 'THREATEN_TO_RESIGN', 'CONSIDERS_MERGER_WITH',
    'IN_TALKS_WITH_BOARD', 'RETURNS_AS_CEO', 'RETURNS_TO', 'CONSIDERS_BOARD_SEAT', 'AIMS_TO_DEVELOP_AGI_WITH', 'QUESTIONED_FIRING_SAM',
    'FOUNDED_BY', 'INVESTED_IN', 'DEMOTED_FROM', 'RELEASES_HIRING_STATEMENT', 'HIRED_BY', 'REGRETS_FIRING','MENTIONS', 'EXPLAINS_DECISIONS', 'DESCRIBES', 'FORMER_PRESIDENT']

st.sidebar.header('Filters')
selected_node_types = st.sidebar.multiselect('Node Types', node_types, default=node_types)
selected_relationship_types = st.sidebar.multiselect('Relationship Types', relationship_types, default=relationship_types)

# Sidebar filters for node and relationship types
selected_node_types = st.sidebar.multiselect('Node Types', node_types, default=node_types)
selected_relationship_types = st.sidebar.multiselect('Relationship Types', relationship_types, default=relationship_types)

# Initialize state variables and check for changes in selections
if 'prev_node_types' not in st.session_state:
    st.session_state.prev_node_types = selected_node_types
if 'prev_relationship_types' not in st.session_state:
    st.session_state.prev_relationship_types = selected_relationship_types

# Update graph if selections change
if (selected_node_types != st.session_state.prev_node_types or 
    selected_relationship_types != st.session_state.prev_relationship_types):
    st.session_state.prev_node_types = selected_node_types
    st.session_state.prev_relationship_types = selected_relationship_types
    # Construct and fetch new graph data
    cypher_query = construct_cypher_query(selected_node_types, selected_relationship_types)
    with driver.session() as session:
        nodes, edges = fetch_graph_data(session, cypher_query)
    # Define the configuration for the graph visualization
    config = Config(height=600, width=800, directed=True, nodeHighlightBehavior=True, highlightColor="#F7A7A6")
    # Render the graph using agraph with the specified configuration
    agraph(nodes=nodes, edges=edges, config=config)

# Chat interface
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hi there, what can I help you with?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="Type your query..."):
    if not openai_api_key:
        st.error("Please add your OpenAI API key to continue.")
    else:
        user_id = '1'  # Example user ID
        session_id = '1'  # Example session ID
        # Process input through agent
        from neo4j_advanced_rag.agent import AgentInput, agent_executor
        response = agent_executor.invoke(AgentInput(input=prompt, user_id=user_id, session_id=session_id))
        retriever_module = importlib.import_module("neo4j_advanced_rag.retrievers")
        from neo4j_advanced_rag.neo4j_vector import VectorTool
        graph_tool = GraphTool()
        vector_tool = VectorTool(openai_api_key=openai_api_key)
        # Now you can use functions or classes from retriever.py
        # For example, retriever_module.some_function(openai_api_key)
        cypher_query, nl_response = graph_tool._run(prompt, user_id, session_id)
        # Display response
        st.write(response)

        # Update session state with new message
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": response})