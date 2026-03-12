import asyncio
from typing import Any
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import AzureChatOpenAI
from pydantic import SecretStr
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient

BATCH_SYSTEM_PROMPT = """You are a user search assistant. Your task is to find users from the provided list that match the search criteria.

INSTRUCTIONS:
1. Analyze the user question to understand what attributes/characteristics are being searched for
2. Examine each user in the context and determine if they match the search criteria
3. For matching users, extract and return their complete information
4. Be inclusive - if a user partially matches or could potentially match, include them

OUTPUT FORMAT:
- If you find matching users: Return their full details exactly as provided, maintaining the original format
- If no users match: Respond with exactly "NO_MATCHES_FOUND"
- If uncertain about a match: Include the user with a note about why they might match"""

FINAL_SYSTEM_PROMPT = """You are a helpful assistant that provides comprehensive answers based on user search results.

INSTRUCTIONS:
1. Review all the search results from different user batches
2. Combine and deduplicate any matching users found across batches
3. Present the information in a clear, organized manner
4. If multiple users match, group them logically
5. If no users match, explain what was searched for and suggest alternatives"""

USER_PROMPT = """## USER DATA:
{context}

## SEARCH QUERY: 
{query}"""


class TokenTracker:
    def __init__(self):
        self.total_tokens = 0
        self.batch_tokens = []

    def add_tokens(self, tokens: int):
        self.total_tokens += tokens
        self.batch_tokens.append(tokens)

    def get_summary(self):
        return {
            'total_tokens': self.total_tokens,
            'batch_count': len(self.batch_tokens),
            'batch_tokens': self.batch_tokens
        }


llm_client = AzureChatOpenAI(
    temperature=0.0,
    azure_deployment='gpt-4o',
    azure_endpoint=DIAL_URL,
    api_key=SecretStr(API_KEY),
    api_version=""
)

token_tracker = TokenTracker()


def join_context(context: list[dict[str, Any]]) -> str:
    context_str = ""
    for user in context:
        context_str += f"User:\n"
        for key, value in user.items():
            context_str += f"  {key}: {value}\n"
        context_str += "\n"
    return context_str


async def generate_response(system_prompt: str, user_message: str) -> str:
    print("Processing...")
    # 1. Create messages array
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message)
    ]
    # 2. Generate response
    response = await llm_client.ainvoke(messages)
    # 3. Get token usage
    total_tokens = response.response_metadata.get('token_usage', {}).get("total_tokens", 0)
    # 4. Track tokens
    token_tracker.add_tokens(total_tokens)
    # 5. Print response info
    print(response.content)
    print(f"Tokens used in this call: {total_tokens}")
    # 6. Return response
    return response.content


async def main():
    print("Query samples:")
    print(" - Do we have someone with name John that loves traveling?")

    user_question = input("> ").strip()
    if user_question:
        print("\n--- Searching user database ---")

        # 1. Get all users
        all_users = UserClient().get_all_users()

        # 2. Split users into batches of 100
        batch_size = 100
        user_batches = [all_users[i:i + batch_size] for i in range(0, len(all_users), batch_size)]

        # 3. Prepare async tasks for each batch
        tasks = []
        for user_batch in user_batches:
            tasks.append(generate_response(
                system_prompt=BATCH_SYSTEM_PROMPT,
                user_message=USER_PROMPT.format(
                    context=join_context(user_batch),
                    query=user_question
                )
            ))

        # 4. Run all batch tasks in parallel
        batch_results = await asyncio.gather(*tasks)

        # 5. Filter out NO_MATCHES_FOUND responses
        relevant_results = [r for r in batch_results if "NO_MATCHES_FOUND" not in r]

        # 5. If matches found, run final aggregation
        if relevant_results:
            combined_results = "\n\n".join(relevant_results)
            await generate_response(
                system_prompt=FINAL_SYSTEM_PROMPT,
                user_message=f"SEARCH RESULTS:\n{combined_results}\n\nORIGINAL QUERY: {user_question}"
            )
        else:
            # 6. No matches found
            print("No users found matching your query.")

        # 7. Print token usage summary
        summary = token_tracker.get_summary()
        print(f"\n--- Token Usage Summary ---")
        print(f"Total tokens used: {summary['total_tokens']}")
        print(f"Number of batches processed: {summary['batch_count']}")
        print(f"Tokens per batch: {summary['batch_tokens']}")


if __name__ == "__main__":
    asyncio.run(main())