from fastmcp import FastMCP

mcp = FastMCP("StudyAssistant")

# ----------------------------
# 🧠 TOOLS – actionable functions
# ----------------------------

@mcp.tool()
def calculate_gpa(grades: list[float], credits: list[int]) -> float:
    """Calculate GPA from grades and credits."""
    total_quality_points = sum(g * c for g, c in zip(grades, credits))
    total_credits = sum(credits)
    return round(total_quality_points / total_credits, 2)

# ----------------------------
# 📚 RESOURCES – static lookups
# ----------------------------

@mcp.resource("definition://{term}")
def subject_definitions(term: str) -> str:
    """Returns subject definitions for common academic terms."""
    dictionary = {
        "osmosis": "Osmosis is the movement of water molecules through a semi-permeable membrane from a region of low solute concentration to high solute concentration.",
        "photosynthesis": "Photosynthesis is the process by which green plants convert sunlight into chemical energy.",
        "entropy": "Entropy is a measure of disorder or randomness in a system.",
    }
    return dictionary.get(term.lower(), "Definition not found.")

# ----------------------------
# ✏️ PROMPTS – LLM-guided tasks
# ----------------------------

@mcp.prompt()
def summarize_chapter(chapter_text: str) -> str:
    return f"Summarize the following textbook chapter into simple bullet points:\n\n{chapter_text}"

@mcp.prompt()
def explain_concept(concept: str) -> str:
    return f"Explain the concept of '{concept}' in simple language for a high school student."

# ----------------------------
# 🚀 Start MCP server
# ----------------------------

if __name__ == "__main__":
    mcp.run()
