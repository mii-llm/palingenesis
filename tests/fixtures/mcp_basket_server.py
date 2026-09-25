"""A small stateless MCP server for the RL tests: a basket behind an explicit handle (the
spec's stateful-tools pattern), a failing tool and a grader.

    python tests/fixtures/mcp_basket_server.py                   # stdio
    python tests/fixtures/mcp_basket_server.py http 8765         # Streamable HTTP on :8765/mcp
"""

import sys
import uuid

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

server = MCPServer("basket")
PRICES = {"apple": 2, "pear": 3}
BASKETS: dict[str, list[str]] = {}


@server.tool()
def create_basket(budget: int = 10) -> dict:
    """Create an empty basket. Baskets expire when the server restarts."""
    basket_id = f"bsk_{uuid.uuid4().hex[:8]}"
    BASKETS[basket_id] = []
    return {"basket_id": basket_id}


@server.tool()
def add_item(basket_id: str, sku: str) -> str:
    """Add an item to the basket.

    Args:
        basket_id: The basket.
        sku: apple or pear.
    """
    if sku not in PRICES:
        raise ToolError(f"unknown sku {sku!r}: choose apple or pear")  # its message reaches the policy
    BASKETS[basket_id].append(sku)
    return f"added {sku}; {len(BASKETS[basket_id])} item(s)"


@server.tool()
def total(basket_id: str) -> int:
    """The basket's total price."""
    return sum(PRICES[s] for s in BASKETS[basket_id])


@server.tool()
def grade(basket_id: str, answer: str) -> dict:
    """Hidden from the policy: 1 when the basket totals 5 and the answer says so."""
    correct = sum(PRICES[s] for s in BASKETS[basket_id]) == 5
    return {"reward": float(correct and "5" in answer)}


if __name__ == "__main__":
    if sys.argv[1:2] == ["http"]:
        server.run("streamable-http", port=int(sys.argv[2]))
    else:
        server.run()
