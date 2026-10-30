from mcp.server.fastmcp import FastMCP

server = FastMCP("Lyle State Isolation Test")
counter = 0


@server.tool()
def next_value() -> str:
    """Increment this process's state without modifying any files or external system."""
    global counter
    counter += 1
    return str(counter)


if __name__ == "__main__":
    server.run()
