import asyncio
from typing import Any, Dict
import pandas as pd
from langchain_core.messages import HumanMessage

# 1. Official Documentation Imports
from phoenix.evals import create_classifier, evaluate_dataframe
from phoenix.evals.llm import LLM

# Import your FastAPI app directly
from simple.main_gemini import app


async def execute_graph_for_eval(
    graph_app: Any, prompt: str, session_id: str
) -> Dict[str, Any]:
    """
    Executes the LangGraph inside the active lifespan and extracts the model's output string.
    """
    config = {"configurable": {"thread_id": session_id}}
    inputs = {"messages": [HumanMessage(content=prompt)]}

    final_state = await graph_app.ainvoke(inputs, config=config)
    messages = final_state.get("messages", [])
    ai_response = messages[-1].content if messages else ""
    return {"output": ai_response}


async def main():
    # 2. Prepare your Golden Test Dataset matching the input variables
    TEST_DATASET = [
        {"input": "Can you check stock for prod-01?"},
        {"input": "Tell the user that Vietnam's area is big."},
        {"input": "Send an update email to test@domain.com telling them hello."},
    ]

    print("[EVAL ENGINE] Initializing FastAPI Lifespan context...")
    results = []

    # 3. Securely manage your runtime backend with the app router context
    async with app.router.lifespan_context(app):
        graph_app = app.state.graph_app
        print("[EVAL ENGINE] Lifespan active. Running graph scenarios...")

        for i, test_case in enumerate(TEST_DATASET):
            session_id = f"eval_session_{i}"
            run_output = await execute_graph_for_eval(
                graph_app, test_case["input"], session_id
            )

            # Save inputs and outputs matching the prompt template format fields
            results.append(
                {
                    "input": test_case["input"],
                    "output": run_output["output"],
                }
            )

    # Convert execution logs into the required evaluation DataFrame
    df = pd.DataFrame(results)

    print("\n[EVAL ENGINE] Initializing Judge LLM via Phoenix Adapter Interface...")

    # 4. Instantiate the Gemini Judge using the official provider contract
    # Phoenix matches your GOOGLE_API_KEY environment variable behind the scenes
    judge_llm = LLM(provider="google", model="gemini-3.1-flash-lite")

    # 5. Create your custom Relevance Evaluator using create_classifier
    relevance_evaluator = create_classifier(
        name="relevance",
        prompt_template="Is the response relevant to the user query?\n\nQuery: {input}\nResponse: {output}",
        llm=judge_llm,
        choices={"relevant": 1.0, "irrelevant": 0.0},
    )

    print("[EVAL ENGINE] Executing automated dataframe evaluations...")

    # Run the evaluation over the accumulated dataframe logs
    results_df = evaluate_dataframe(
        dataframe=df,
        evaluators=[relevance_evaluator],
    )

    print("\n================ EVALUATION RESULTS MATRIX ================")
    print(results_df)


if __name__ == "__main__":
    asyncio.run(main())
