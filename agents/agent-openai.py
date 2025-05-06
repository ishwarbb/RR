import requests
from PyPDF2 import PdfReader
from google import genai
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

def download_arxiv_pdf(arxiv_url, output_path):
    response = requests.get(arxiv_url)
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print(f"PDF downloaded successfully and saved to {output_path}")
    else:
        print(f"Failed to download PDF. Status code: {response.status_code}")

arxiv_url = input("Enter the arXiv URL: ")
output_path = f"{arxiv_url[-14:-4]}.pdf"
download_arxiv_pdf(arxiv_url, output_path)

text = ""

reader = PdfReader(output_path)
number_of_pages = len(reader.pages)

for n in range(number_of_pages):
    page = reader.pages[n]
    text += page.extract_text()

# print(text)

#TODO ✅ check if this is robust to detect wrong links - yes it is 
client = genai.Client(api_key="google-can-suck-my-dick")
response = client.models.generate_content(
    model="gemini-2.0-flash", contents=f"I will give you a text of a research paper, if they have provided a link to a github repo of their work in the paper, give it to me, if there is no such repo link provided in the paper, reply with 'no repo found', paper: {text}"
)

repo = response.text.replace("\n", '').replace("`", "").lower()

if repo != "no repo found":
    try:
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

            model = ChatOpenAI(model="gpt-4o", temperature=0, api_key="fuck-microsoft-i-hate")

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
        else:
            print("no")
    except requests.exceptions.RequestException as e:
        print(f"Failed to reach the repo URL: {e}")
else:
    print("no repo found")