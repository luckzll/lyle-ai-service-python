from mcp.server.fastmcp import FastMCP


server = FastMCP("Lyle MCP Test")


@server.tool()
def echo(message: str) -> str:
    """Return a message for MCP client integration tests."""
    return f"echo:{message}"


if __name__ == "__main__":
    server.run()
