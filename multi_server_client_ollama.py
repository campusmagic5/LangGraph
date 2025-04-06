# pip install langchain-mcp-adapters
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

import asyncio

model = ChatOllama(model="llama3.2")  # Or any available Ollama model

async def main():
    async with MultiServerMCPClient({
        "strings": {
            "command": "python",
            "args": ["string_tools_server.py"],  # Update path
            "transport": "stdio",
        },
        "datetime": {
            "command": "python",
            "args": ["datetime_tools_server.py"],  # Update path
            "transport": "stdio",
        }
    }) as client:
        agent = create_react_agent(model, client.get_tools())

        query_1 = {"messages": "Reverse the string 'LangGraph and MCP are cool!'"}
        query_2 = {"messages": "What is the current date and time?"}
        query_3 = {"messages": "How many days until 2025-12-31?"}

        res1 = await agent.ainvoke(query_1)
        # print("\n🌀 Reversed String:", res1)
        for m in res1['messages']:
                m.pretty_print()

        res2 = await agent.ainvoke(query_2)
        # print("\n🕒 Current DateTime:", res2)
        for m in res2['messages']:
                m.pretty_print()

        res3 = await agent.ainvoke(query_3)
        # print("\n📅 Days Until 2025-12-31:", res3)
        for m in res3['messages']:
                m.pretty_print()

if __name__ == "__main__":
    asyncio.run(main())