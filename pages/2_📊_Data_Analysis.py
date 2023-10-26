from langchain.agents import ConversationalChatAgent, AgentExecutor
from langchain.tools import DuckDuckGoSearchRun
from langchain.utilities import WikipediaAPIWrapper

from langchain.agents import AgentType
from langchain.agents import create_pandas_dataframe_agent
from langchain.callbacks.streamlit import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI

from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.memory import ConversationBufferMemory

from langchain.tools.python.tool import PythonAstREPLTool
from langchain.agents import initialize_agent, Tool
from langchain.chains import LLMMathChain
import pandas as pd
import os
import re
import plotly.io as pio
import streamlit as st
import plotly.express as px

import openai
from dotenv import load_dotenv , find_dotenv
from pathlib import Path

pio.templates.default = "plotly"

file_formats = {
    "csv": pd.read_csv,
    "xls": pd.read_excel,
    "xlsx": pd.read_excel,
    "xlsm": pd.read_excel,
    "xlsb": pd.read_excel,
}


dotenv_path = Path('path/to/.env')
_ = load_dotenv(dotenv_path=dotenv_path) # find_dotenv()
openai.api_key = os.getenv('OPEN_API_KEY') # os.eniron('OPEN_API_KEY')

openai_api_key = openai.api_key  # st.sidebar.text_input("OpenAI API Key", type="password")

if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.")
    st.stop()

def clear_submit():
    """
    Clear the Submit Button State
    Returns:

    """
    st.session_state["submit"] = False


@st.cache_data(ttl="2h")
def load_data(uploaded_file):
    try:
        ext = os.path.splitext(uploaded_file.name)[1][1:].lower()
    except:
        ext = uploaded_file.split(".")[-1]
    if ext in file_formats:
        return file_formats[ext](uploaded_file)
    else:
        st.error(f"Unsupported file format: {ext}")
        return None


st.set_page_config(page_title="Chat with Data", page_icon="📊")
st.title("📊 Chat with Data")

uploaded_file = st.file_uploader(
    "Upload a Data file",
    type=list(file_formats.keys()),
    help="Various File formats are Supported",
    on_change=clear_submit,
)

if not uploaded_file:
    st.warning(
        "This app uses LangChain's `PythonAstREPLTool` which is vulnerable to arbitrary code execution. Please use caution in deploying and sharing this app."
    )

if uploaded_file:
    df = load_data(uploaded_file)
    df_columns = df.columns.to_list()  # print column names of df




memory = ConversationBufferMemory(
     chat_memory=StreamlitChatMessageHistory(key="langchain_messages"), return_messages=True, memory_key="chat_history", output_key="output")

if "messages" not in st.session_state or st.sidebar.button("Clear conversation history"):
    #st.session_state["messages"].clear()
    st.session_state.messages = []
    memory.clear()
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
    st.session_state.steps = {}


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input(placeholder="What is this data about?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    

    query_prompt = f"""Given a pandas dataframe: 'df' with schema {df_columns}\
            Based on this dataframe, answer this question: {prompt}"""

      
    plot_prompt = f"""You are a helpful assistant that is expert in writing Python code\
        using plotly.express,Given a pandas dataframe: df with schema {df_columns}\
        Based on this dataframe, write a python code to {prompt} from previous data,\
        The solution should be given using plotly.express, Do not use matplotlib.\
        the output python code must be in the following format ```python ....```"""


llm = ChatOpenAI(
    temperature=0, model="gpt-3.5-turbo-0613", openai_api_key=openai_api_key, streaming=True
)

python = PythonAstREPLTool(locals={"df": df})  # set access of python_repl tool to the dataframe


# create calculator tool
calculator = LLMMathChain.from_llm(llm=llm, verbose=True)

wikipedia = WikipediaAPIWrapper()
search = DuckDuckGoSearchRun()

def extract_python_code(text):
    """Extracts Python code from a string response."""
    # Use a regex pattern to match content between triple backticks
    code_pattern = r"```python(.*?)```"
    match = re.search(code_pattern, text, re.DOTALL)

    if match:
        # Extract the matched code and strip any leading/trailing whitespaces
        return match.group(1).strip()
    return None



agent_analytics_tool = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    handle_parsing_errors=True
)

tools = [Tool(
    name='Analytics_Tool',
    func=agent_analytics_tool.run,
    description=f"""Useful tool for data analysis, answer questions about data stored in pandas dataframe 'df'\
                    Do n't use this tool if the question is not related to the data in the pandas dataframe 'df'.
                """
                ),
    Tool(
        name="Plot_Visualization",
        func=python.run,
        description=f"""
                Useful for when you need to show charts or visualize plot about data stored in pandas dataframe 'df'.
                Run python pandas operations on 'df' to help you get the right answer.
                """
    ),
    Tool(
    name='wikipedia',
    func= wikipedia.run,
    description="Useful for when you need to look up a topic, country or person on wikipedia"
),

Tool(
    name='DuckDuckGo_Search',
    func= search.run,
    description="Useful for when you need to do a search on the internet to find information that another tool can't find. be specific with your input."
),
    Tool(
        name="Calculator",
        func=calculator.run,
        description=f"""
                    Useful when you need to do math operations or arithmetic.
                    """),
]

agent_kwargs = {
    'prefix': f"""You are friendly AI assistant. You are tasked to assist on questions related to dataset stored in in pandas dataframe 'df'. You have access to the following tools:"""}

agent = initialize_agent(tools,
                         llm,
                         agent=AgentType.OPENAI_FUNCTIONS,#CONVERSATIONAL_REACT_DESCRIPTION, #ZERO_SHOT_REACT_DESCRIPTION,
                         verbose=True,
                         agent_kwargs=agent_kwargs,
                         handle_parsing_errors=True,
                         return_intermediate_steps=True,
                         memory=memory,
                         max_iterations=5)



with st.chat_message("assistant"):
    st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
    
    plot_flag = False
    
    if prompt:
        if  "plot" in prompt:
            plot_flag = True
            #response = executor({"input": plot_prompt, "chat_history": []})
            response = agent({"input": plot_prompt, "chat_history": []})
            # st.session_state.messages.append({"role": "user", "content": query_system_prompt + query_prompt})
        else:
            response = agent({"input": query_prompt, "chat_history": []})
        #st.session_state.messages.append({"role": "assistant", "content": response["output"]})

        if plot_flag:
            code = extract_python_code(response['output'])
            print(code)
            if code is None:
                st.warning(
                    "Couldn't find data to plot in the chat. "
                    "Check if the number of tokens is too low for the data at hand. "
                    "I.e. if the generated code is cut off, this might be the case.",
                    icon="🚨"
                )
            else:
                code = code.replace("fig.show()", "")
                code += """st.plotly_chart(fig, theme='streamlit', use_container_width=True)"""  # noqa: E501
                st.write(f"""```python 
                         {code}""")
                exec(code)
                st.session_state.messages.append({"role": "assistant", "content":f"""```python 
                                                  {code}"""})
                
        else:
            st.write(response["output"])
            st.session_state.messages.append({"role": "assistant", "content": response["output"]})