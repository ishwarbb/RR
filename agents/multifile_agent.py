from langchain_openai import ChatOpenAI
from langchain_experimental.tools import PythonAstREPLTool
from langchain_experimental.utilities import PythonREPL
from langchain_core.messages import (
    BaseMessage, 
    HumanMessage,
    ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import functools 
from typing import Annotated
from langchain_core.tools import tool
from langchain_core.messages import AIMessage
from langchain_community.callbacks.manager import get_openai_callback
import time
import pandas as pd 
from langgraph.checkpoint.sqlite import SqliteSaver
import operator 
from typing import Annotated, Sequence, TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from tenacity import retry, stop_after_attempt
# Either agent can decide to end
from typing import Literal
import os
import tempfile
import shutil
import glob
import json
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 20)

# Declaring tools
repl = PythonREPL()

def ensure_code_format(args):
    # Load the JSON string into a dictionary
    try:
        # Try to load the JSON string into a dictionary
        args_dict = json.loads(args)
        
        # Check if the dictionary has a "code" key
        if "code" in args_dict:
            return args
        else:
            # Assume the dictionary has only one key which is the code itself
            # Get that key
            code_key = list(args_dict.keys())[0]
            
            # Create the new dictionary in the desired format
            new_args_dict = {"code": code_key}
            
            # Convert the new dictionary back to a JSON string
            new_args = json.dumps(new_args_dict)
            
            return new_args
    except json.JSONDecodeError:
        print("invalid args, changing to correct format...")

        # If it's not a JSON string, assume it's direct code
        # Create the new dictionary in the desired format
        new_args_dict = {"code": args}
        
        # Convert the new dictionary back to a JSON string
        new_args = json.dumps(new_args_dict)
        
        return new_args

def parse_tool_call(additional_kwargs):
    parsed_tool_calls = []
    for tool_call in additional_kwargs['tool_calls']:
        function_name = tool_call['function']['name']
        function_arguments = tool_call['function']['arguments']
        
        # Parse the arguments as JSON to extract the code
        try:
            arguments_dict = json.loads(function_arguments)
            code = arguments_dict.get('code', function_arguments)
        except json.JSONDecodeError:
            # If not JSON, use the function_arguments directly
            code = function_arguments
        
        parsed_tool_call = {
            'name': function_name,
            'args': arguments_dict,
            'id': tool_call['id']
        }
        
        parsed_tool_calls.append(parsed_tool_call)
    
    return parsed_tool_calls

@tool
@retry(stop=stop_after_attempt(5))
def python_repl(
    code: Annotated[str, "The python code to execute."],
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user.
    Always give the code in the required JSON format."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"\nSuccessfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return (
        result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
    )

# util function to create an agent
def create_agent(llm, tools, system_message: str):
    """creates a basic agent

    Args:
        llm (_type_): _description_
        tools (_type_): _description_
        system_message (str): _description_

    Returns:
        _type_: llm with tools and system prompt, basically an agent
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                " Use the provided tools to answer the question."
                " If you have the answer to the users, finish by prefixing your response with FINAL ANSWER."
                " Think step by step for solving the task."
                " Make sure to finish the task by learning from your mistakes, keep trying different approaches until you reach the answer."
                " You have access to the following tools: {tool_names}.\n{system_message}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    return prompt | llm.bind_tools(tools)

def agent_node(state, agent, name):
    """returns an agent node for the graph
        agent node processes the messages in the state, and appends a message.
        
    Args:
        state (_type_): _description_
        agent (_type_): _description_
        name (_type_): _description_

    Returns:
        _type_: _description_
    """
    result = agent.invoke(state)

    if isinstance(result, ToolMessage):
        print(ToolMessage)
        pass
    else: 
        result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
        try:
            if result.additional_kwargs:            
                for tool_call in result.additional_kwargs['tool_calls']:
                    if tool_call['function']['name'] == 'python_repl':
                        args = tool_call['function']['arguments']
                        corrected_args = ensure_code_format(args)
                        tool_call['function']['arguments'] = corrected_args
            if result.invalid_tool_calls:
                result.invalid_tool_calls = []
            result.tool_calls = parse_tool_call(result.additional_kwargs)
        except Exception as e:
            print(e) 
            pass
    return {
        "messages": [result],
        "sender": name,
    }

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str


def router(state) -> Literal["call_tool", "__end__", "continue"]:
    # This is the router
    messages = state["messages"]
    last_message = messages[-1]
    # print(messages)
    if last_message.tool_calls:
        # The previous agent is invoking a tool
        return "call_tool"
    if "FINAL ANSWER" in last_message.content:
        # Any agent decided the work is done
        return "__end__"
    return "continue"

# globals
SERVICE_URL = "http://prod0-intuitionx-llm-router.sprinklr.com:80"

class Multifile_Agent: 
    def __init__(self):
        llm = ChatOpenAI(model = "gpt-4o", api_key="null", temperature=0, base_url=SERVICE_URL)
        self.llm = llm
        self.graph = None
        self.datasets = None
        self.file_paths = []
        self.file_names = []
        self.dir_path = None
        self.repl = PythonREPL()
        self.memory = None
        self.config = {"configurable": {"thread_id": "1"}, "recursion_limit": 20}
    
    def restart_memory(self):
        if self.memory:
            self.memory.conn.close()
            print("Memory closed.")

        self.memory = SqliteSaver.from_conn_string(":memory:")
            
    def load_datasets(self, dfs, file_names):
        self.datasets = dfs
        self.file_names = []
        # Create a persistent temporary directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            for i, df in enumerate(self.datasets):
                if isinstance(df, dict):  # Handle multiple sheets
                    for sheet_name, sheet_df in df.items():
                        # Construct a unique file path for each sheet
                        file_path = os.path.join(temp_dir, f"{file_names[i]}_{sheet_name}.csv")
                        self.file_names.append(f"{file_names[i]}_{sheet_name}.csv")
                        print(file_path)
                        sheet_df.to_csv(file_path, index=False)
                        self.file_paths.append(file_path)
                else:
                    file_path = os.path.join(temp_dir, file_names[i])
                    print(file_path)
                    df.to_csv(file_path, index=False)
                    self.file_paths.append(file_path)
                    self.file_names.append(file_names[i])
            self.dir_path = temp_dir
            # Print confirmation
            print(f"Datasets loaded successfully to: {self.dir_path}.")
        except Exception as e:
            print(f"An error occurred: {e}")
            shutil.rmtree(temp_dir)  # Cleanup in case of an error
            raise e
        
        
        # Return the file paths
        return self.file_paths
        

    def create_graph(self):
        self.memory = SqliteSaver.from_conn_string(":memory:")
        
        sysmsg = ""
        if self.dir_path: 
            sysmsg = f"""
            All the datasets you need are in the directory: `{self.dir_path}`. \
            The file names are as follows: `{self.file_names}`. \
            Use python code to load this data and answer the user's question. \
            the output is read through stdout, so use print to output anything. \
            If you are generating any sort of visualizations, store them in the same directory.
            """
    
        doer_agent = create_agent(
            self.llm, 
            [python_repl],
            system_message = sysmsg
        )

        doer_node = functools.partial(agent_node, agent=doer_agent, name="doer")
        
        tools = [python_repl]
        tool_node = ToolNode(tools)
        
        workflow = StateGraph(AgentState)

        workflow.add_node("doer", doer_node)
        workflow.add_node("call_tool", tool_node)

        workflow.add_conditional_edges(
            "doer",
            router,
            {"continue": "doer", "call_tool": "call_tool", "__end__": END},
        )

        workflow.add_conditional_edges(
            "call_tool",
            lambda x: x["sender"],
            {
                # "thinker": "thinker",
                "doer": "doer",
            },
        )
        workflow.set_entry_point("doer")
        self.graph = workflow.compile(checkpointer=self.memory)

        return self.graph
    
    def run(self, query):
        """Run a query through the framework"""
        
        # new graph everytime to restart the memory
        self.graph = self.create_graph()
        
        preinput = "Think step by step before answering the question. Feel free to look at certain rows or columns if you ever get stuck. Question: "

        inputs = {"messages": [HumanMessage(content= preinput + str(query))]}
        with get_openai_callback() as cb:
            start = time.time()
            for s in self.graph.stream(inputs, self.config, stream_mode="values"):
                message = s["messages"][-1]
                if isinstance(message, tuple):
                    print(message)
                else:
                    message.pretty_print()
            answer = message.content
            end = time.time()
            time_taken = end-start
        
        print(f"time taken = {time_taken}")
        print(f"input tokens = {cb.prompt_tokens}")
        print(f"output tokens = {cb.completion_tokens}")
        print(f"number of requests = {cb.successful_requests}")
        print(f"cost = {cb.total_cost}")
        
        # close the session
        self.restart_memory()
        return time_taken, answer, cb.prompt_tokens, cb.completion_tokens, cb.successful_requests, cb.total_cost



# Usage example
if __name__ == "__main__":
    agent = Multifile_Agent()
    dir_path = "/Users/vamshi.bonagiri/Desktop/intern-2024/multi_agent_framework/multifile/olist/data"
    file_paths = glob.glob(os.path.join(dir_path, '*.csv'))
    dataframes = [pd.read_csv(file) for file in file_paths]
    file_names = [os.path.basename(file_path) for file_path in file_paths]

    agent.load_datasets(dataframes, file_names)
    graph = agent.create_graph()
    
    while True:
        user_input = input("Enter your query or 'exit' to stop: ")
        try: 
            if user_input.lower() == 'exit':
                break
            preinput = "Think step by step before answering the question. Feel free to look at certain rows or columns if you ever get stuck. Question: "
            inputs = {"messages": [HumanMessage(content=preinput + user_input)]}
            
            for s in graph.stream(inputs, agent.config, stream_mode="values"):
                message = s["messages"][-1]
                if isinstance(message, tuple):
                    print(message)
                else:
                    message.pretty_print()

        except Exception as e:
            response = f"Sorry, I'm just a version 1, looks like I broke due to the error: {e}"
            print(response)
            agent.restart_memory()
            graph = agent.create_graph()