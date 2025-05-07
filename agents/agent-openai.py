import requests
from PyPDF2 import PdfReader
from autogen.coding import DockerCommandLineCodeExecutor, CodeBlock
import tempfile
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    BaseMessage, 
    HumanMessage,
    AIMessage
)
from langchain_core.tools import tool
from tenacity import retry, stop_after_attempt

repo = input("github link: ")

repo_response = requests.get(repo)
if repo_response.status_code == 200:
    temp_dir = tempfile.TemporaryDirectory()
    executor = DockerCommandLineCodeExecutor(
        image="ubuntu",
        timeout=420,
        work_dir=temp_dir.name,
    )

    blk0 = CodeBlock(
        code="apt-get update",
        language="bash"
    )

    blk1 = CodeBlock(
        code="apt-get install -y python3 pip",
        language="bash"
    )

    blk2 = CodeBlock(
        code="python3 --version",
        language="bash"
    )

    blk3 = CodeBlock(
        code="pip install json",
        language="bash"
    )

    result = executor.execute_code_blocks(code_blocks=[blk0])
    print(result)
    result = executor.execute_code_blocks(code_blocks=[blk1])
    print(result)
    result = executor.execute_code_blocks(code_blocks=[blk2])
    print(result)
    result = executor.execute_code_blocks(code_blocks=[blk3])
    print(result)

    model = ChatOpenAI(model="gpt-4o", temperature=0, api_key="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")

    from langchain_core.tools import tool

    #TODO openai agents - take a sneak peek into the library

    @tool
    @retry(stop=stop_after_attempt(10))
    def docker_ubuntu_terminal(
        code: str
    ):
        """This is an ubuntu command line running in a docker container, use this to execute command line instructions. use apt-get install -y to install packages. Write and execute python code by writing to and executing a file. Use pip install to install python packages, for example, to install the json package, the command is: 'pip install json' never use exclamation marks when issuing commands like git or pip or any other command, use pip instead of !pip and git instead of !git."""
        try:
            blk = CodeBlock(
                code=code,
                language="sh"
            )
            result = executor.execute_code_blocks(code_blocks=[blk])
            result_str = f"\nSuccessfully executed code:\n```sh\n{code}\n```\nStdout: {result}"
            return (
            result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
            )
        except Exception as e:
            return f"Failed to execute.: {repr(e)}"

    tools = [docker_ubuntu_terminal]

    from langgraph.prebuilt import create_react_agent

    graph = create_react_agent(model, tools=tools)

    def print_stream(stream):
        for s in stream:
            message = s["messages"][-1]
            if isinstance(message, tuple):
                print(message)
            else:
                message.pretty_print()

    inputs = {"messages": [("user", f"clone this repository {repo}. The repository is associated with a research project, follow the instructions in the readme file to run the code and reproduce the results.")]}

    print_stream(graph.stream(inputs, stream_mode="values"))
