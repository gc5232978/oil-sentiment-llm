import os 
from dotenv import load_dotenv
from langchain_groq import ChatGroq 
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
load_dotenv()


GROQ_API_KEY = os.environ.get("GROQ_API_KEY")


uri = "sqlite:///database.db"
db = SQLDatabase.from_uri(uri)
llm = ChatGroq(model="llama3-70b-8192", temperature=0)


template = """
    Based on the table schema below, write a SQL query. No pre-amble. No backticks.
    {schema}

    Question: {question}
    SQL Query: """


prompt = ChatPromptTemplate.from_template(template)


def get_schema(_):
    return db.get_table_info()


chain = (
    RunnablePassthrough.assign(schema=get_schema)
    | prompt
    | llm.bind(stop=["\nSQLResult:"])
    | StrOutputParser()
)


template2 = """Based on the table schema below, question, sql query, and sql response, 
write a natural language response without a preamble:
{schema}

Question: {question}
SQL Query: {query}
SQL Response: {response}"""
prompt2 = ChatPromptTemplate.from_template(template2)


def run_query(query):
    return db.run(query)


chain2 = (
    RunnablePassthrough.assign(query=chain).assign(
        schema=get_schema,
        response=lambda vars: run_query(vars["query"]),
    )
    | prompt2
    | llm
)


user_question1 = input("Enter your query: ")
result = chain2.invoke({"question": user_question1})
print(result.content)
