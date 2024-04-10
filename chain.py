from operator import itemgetter

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel
from langchain.schema.output_parser import StrOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    ConfigurableField,
    RunnableParallel, 
)

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from typing import Tuple, List, Optional
import os

from retrievers import final_retriever , llm

# from retrievers import (
#     hypothetic_question_vectorstore,
#     parent_vectorstore,
#     summary_vectorstore,
#     typical_rag,
# )

# Add typing for input
class Question(BaseModel):
    question: str


def initialize_chain(openai_api_key, typical_rag, parent_vectorstore, hypothetic_question_vectorstore, summary_vectorstore):
    os.environ["OPENAI_API_KEY"] = openai_api_key
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    model = ChatOpenAI()

    retriever = typical_rag.as_retriever().configurable_alternatives(
        ConfigurableField(id="strategy"),
        default_key="typical_rag",
        parent_strategy=parent_vectorstore.as_retriever(),
        hypothetical_questions=hypothetic_question_vectorstore.as_retriever(),
        summary_strategy=summary_vectorstore.as_retriever(),
    )

    chain = (
        RunnableParallel(
            {
                "context": itemgetter("question") | retriever,
                "question": itemgetter("question"),
            }
        )
        | prompt
        | model
        | StrOutputParser()
    )

    chain = chain.with_types(input_type=Question)

    return chain


def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer
    
def invoke_chain(question: str, chat_history: List[Tuple[str, str]]): 
    # Condense a chat history and follow-up question into a standalone question
    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question,
    in its original language.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""  # noqa: E501
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

   

    _search_query = RunnableBranch(
        # If input includes chat_history, we condense it with the follow-up question
        (
            RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                run_name="HasChatHistoryCheck"
            ),  # Condense follow-up question and chat into a standalone_question
            RunnablePassthrough.assign(
                chat_history=lambda x: _format_chat_history(x["chat_history"])
            )
            | CONDENSE_QUESTION_PROMPT
            | ChatOpenAI(temperature=0)
            | StrOutputParser(),
        ),
        # Else, we have no chat history, so just pass through the question
        RunnableLambda(lambda x : x["question"]),
    )
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    Dame toda la información completa.
    Answer:"""
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        RunnableParallel(
            {
                "context": _search_query | final_retriever,
                "question": RunnablePassthrough(),
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    result = chain.invoke({"question": question, "chat_history": chat_history})
    return result

