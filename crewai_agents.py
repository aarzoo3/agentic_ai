from crewai import Agent
from crewai import Crew, Task
from crewai import LLM
from crewai_tools import VisionTool
from crewai_tools import SerperDevTool
from crewai_tools import PDFSearchTool
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from functools import wraps
import streamlit as st


app = FastAPI()


# loading api keys

import os
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
HF_API = os.getenv("HUGGING_FACE_API")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")



# defining inputs here 

pdf_file = r"C:\Users\lenovo\Downloads\UMD_Decision Letter.pdf"


st.title("Crew AI Chat Assistant")

query = st.text_input("Ask your question:")
# creating LLM

llm_mistral = LLM(
    api_key=MISTRAL_API_KEY,
    model="mistral/mistral-large-latest",
    temperature=0.5
)


# pdf reader tool

pdf_Reader_tool = PDFSearchTool(pdf = pdf_file,
    config=dict(
        llm=dict(
            provider="mistralai", # or any other llm of your choice.
            config=dict(
                model="mistral/mistral-large-latest",
         
            ),
        ),

        embedder=dict(
            provider="huggingface",
            config=dict(
                model="sentence-transformers/all-mpnet-base-v2",
            )
        )
    )
)



# Agent 1 - To check if the question is clear and precise and to rewrite it if necessary
query_check_agent  = Agent(
    role="Text Analyst",
    goal=f"Read and understand the question asked by the user that is {query}",
    backstory="You are an experienced analyst with attention to detail and understands the intent of questions. Also handles if there is any spelling mistake in the query",
    verbose=False,  # Enable logging for debugging
    llm = llm_mistral
)

query_check_task = Task(
    description=f"Understand this question's intent - {query}",
    expected_output="rewritten question in a more precise and clear manner",
    agent=query_check_agent
)

# Agent 2 - Web Searcher
internet_search_agent = Agent(
    role="Web Searcher",
    goal=f"Find the answer to this question asked by the user- {query}",
    backstory="You are an expert in web searching and can find the accurate information from reliable sources",
    verbose=False,  # Enable logging for debugging
    llm=llm_mistral,

    tools=[SerperDevTool()]
)
internet_search_task = Task(
    description=f"Find the answer to this question asked by the user - {query}",
    expected_output="answer to the question asked by the user also include the source of the information",
    agent=internet_search_agent
)

# Agent 3 - PDF Reader
pdf_reader_agent = Agent(
    role="PDF Reader",
    goal=f"Read and understand the PDF file at {pdf_file}and if the query which is - {query} is related to the PDF file then answer it",
    backstory="You are an expert in reading and understanding PDF files and can extract information from them",
    verbose=False,  # Enable logging for debugging
    llm=llm_mistral,
    tools=[pdf_Reader_tool]
)
pdf_reader_task = Task(
    description=f"Read the PDF file at {pdf_file} and answer the question - {query} if information available",
    expected_output="answer to the question asked by the user also include the source of the information",
    agent=pdf_reader_agent
)

# Agent 4 - The decision Maker

decision_maker_agent = Agent(
    role="Decision Maker",
    goal=f"Analyse the {query} and the pdf available, if not able to extract answer from it then use internet",
    backstory="You are an expert in decision making and can decide which agent has the most accurate answer to the question asked by the user",
    verbose=False,  # Enable logging for debugging
    llm=llm_mistral
)
decision_maker_task = Task(
    description=f"Analyse the {query} and the {pdf_file} available, if not able to extract answer from it then use internet",
    expected_output=f"A decision on which agent is better for the {query} ",
    agent=decision_maker_agent
)

# crew = Crew(
#      agents=[query_check_agent, internet_search_agent, pdf_reader_agent, decision_maker_agent],
#      tasks=[query_check_task, internet_search_task, pdf_reader_task, decision_maker_task],
# )
crew = Crew(
     agents=[query_check_agent, internet_search_agent, decision_maker_agent],
     tasks=[query_check_task, internet_search_task, decision_maker_task],
)

# result = crew.kickoff(
#         inputs={"question":query})
# print(result)


#def timetaken(func):
 #   @wraps(func)
  #  def wrapper(*args):
   #     start = time.time()
    #    results = func(*args)
     ##  print("time taken ->", end-start)
       # return results
   # return wrapper

#class QueryRequest(BaseModel):
    # query:str

# @app.post("/questions")
# def get_result(que:str):
#     try:
#         result = crew.kickoff(inputs={"question": que})
#     except Exception as e:
#         return {"error": str(e), "note": "Something went wrong while processing the query."}
#     return result




if query:
    st.write("Thinking... 💭")
    # Call your agent
    result = crew.kickoff(inputs={"question": query})
    st.markdown("### Response:")
    st.write(result)

