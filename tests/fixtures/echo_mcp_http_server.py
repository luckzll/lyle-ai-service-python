import os

from mcp.server.fastmcp import FastMCP


server = FastMCP(
    "Lyle HTTP MCP Test",
    host="127.0.0.1",
    port=int(os.environ["MCP_TEST_PORT"]),
    json_response=True,
)


@server.tool()
def add(a: int, b: int) -> int:
    """Add two numbers for MCP client integration tests."""
    return a + b


if __name__ == "__main__":
    server.run(transport="streamable-http")
