import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama


model = ChatOllama(model="llama3.2")

async def main():
    server_params = StdioServerParameters(
        command="python",
        args=["study_assistant_server.py"],  # Make sure this matches the server file
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            agent = create_react_agent(model, tools)

            # 🔍 Test cases

            # TOOL: GPA Calculation
            query1 = {"messages": "Calculate GPA for grades [3.5, 4.0, 3.0] with credits [3, 4, 2]."}

            # RESOURCE: Definition lookup
            query2 = {"messages": "What is the definition of osmosis?"}

            # PROMPT: Summarize a long text
            sample_chapter = """
Photosynthesis is the process by which green plants and some bacteria use sunlight to synthesize nutrients from carbon dioxide and water. It usually involves the green pigment chlorophyll and generates oxygen as a byproduct.
"""
            query3 = {"messages": f"Summarize this chapter: {sample_chapter}"}

            # PROMPT: Explain a concept
            query4 = {"messages": "Explain entropy in simple language."}

            # Run queries
            for idx, query in enumerate([query1, query2, query3, query4], start=1):
                print(f"\n--- Query {idx} ---")
                res = await agent.ainvoke(query)
                for m in res['messages']:
                    m.pretty_print()

if __name__ == "__main__":
    asyncio.run(main())
