from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode


@tool
def check_database_inventory(product_id: str) -> str:
    """Check the real-time stock availability level for a specific product ID in the warehouse."""
    # Mock inventory data
    db_mock = {
        "prod-01": "12 items remaining",
        "prod-02": "Out of Stock",
        "prod-03": "85 items remaining",
    }
    status = db_mock.get(
        product_id.lower(), "Product ID not found in inventory catalog."
    )
    print(f"\n[TOOL EXECUTION] Checked database inventory for {product_id}: {status}")
    return f"Inventory Status for {product_id}: {status}"


@tool
def send_email_to_customer(customer_email: str, content: str) -> str:
    """Send an official update notification email to a specific customer email address."""
    print(
        f"\n[TOOL EXECUTION] Dispatching email to {customer_email} with content: '{content[:30]}...'"
    )
    return f"Success: Email dispatched cleanly to {customer_email}."


# Combine defined tools into a single list
tools_list = [check_database_inventory, send_email_to_customer]
tools_node = ToolNode(tools_list)
