import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama

model = ChatOllama(model="llama3.2")
from langchain_core.messages import HumanMessage

def to_human_messages(mcp_prompt):
    return [HumanMessage(content=msg.content.text) for msg in mcp_prompt.messages if msg.role == "user"]

async def main():
    server_params = StdioServerParameters(
        command="python",
        args=["study_assistant_server.py"],  # Your MCP server file
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await load_mcp_tools(session)
            agent = create_react_agent(model, tools)

            # TOOL: GPA Calculation
            query1 = {"messages": "Calculate GPA for grades [3.5, 4.0, 3.0] with credits [3, 4, 2]."}

            # RESOURCE: Definition lookup
            term = "photosynthesis"
            resource_uri = f"definition://{term}"
            resource_instance = await session.read_resource(resource_uri)
            resource_content = resource_instance.contents[0].text  # extract only the actual value

            # PROMPT: Summarize a long text
            sample_chapter = """
Photosynthesis is the process by which green plants and some bacteria use sunlight to synthesize nutrients from carbon dioxide and water. It usually involves the green pigment chlorophyll and generates oxygen as a byproduct.
"""
            prompt3 = await session.get_prompt("summarize_chapter", arguments={"chapter_text": sample_chapter})
            response3 = await model.ainvoke(to_human_messages(prompt3))

            # PROMPT: Explain a concept
            concept = "entropy"
            prompt4 = await session.get_prompt("explain_concept", arguments={"concept": concept})
            response4 = await model.ainvoke(to_human_messages(prompt4))

            # Run queries and print results
            print(f"\n--- Query 1: Tool - GPA Calculation ---")
            res1 = await agent.ainvoke(query1)
            for m in res1["messages"]:
                m.pretty_print()

            print(f"\n--- Query 2: Resource - Definition Lookup ---")
            print(f"Definition of {term}:\n{resource_content}")

            print(f"\n--- Query 3: Prompt - Summarize Chapter ---")
            print(response3.content)

            print(f"\n--- Query 4: Prompt - Explain Concept ---")
            print(response4.content)

if __name__ == "__main__":
    asyncio.run(main())
