import asyncio
from typing import Any, Optional

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from pydantic import SecretStr, BaseModel, Field
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient

SYSTEM_PROMPT = """You are a RAG-powered assistant that groups users by their hobbies.

## Flow:
Step 1: User will ask to search users by their hobbies etc.
Step 2: Will be performed search in the Vector store to find most relevant users.
Step 3: You will be provided with CONTEXT (most relevant users, there will be user ID and information about user), and 
        with USER QUESTION.
Step 4: You group by hobby users that have such hobby and return response according to Response Format

## Response Format:
{format_instructions}
"""

USER_PROMPT = """## CONTEXT:
{context}

## USER QUESTION: 
{query}"""


llm_client = AzureChatOpenAI(
    temperature=0.0,
    azure_deployment='gpt-4o',
    azure_endpoint=DIAL_URL,
    api_key=SecretStr(API_KEY),
    api_version=""
)


class GroupingResult(BaseModel):
    hobby: str = Field(description="Hobby. Example: football, painting, horsing, photography, bird watching...")
    user_ids: list[int] = Field(description="List of user IDs that have hobby requested by user.")


class GroupingResults(BaseModel):
    grouping_results: list[GroupingResult] = Field(description="List matching search results.")


def format_user_document(user: dict[str, Any]) -> str:
    return (
        f"User:\n"
        f"  id: {user.get('id')}\n"
        f"  About user: {user.get('about_me')}\n"
        f"---"
    )


class InputGrounder:
    def __init__(self, embeddings: AzureOpenAIEmbeddings, llm_client: AzureChatOpenAI):
        self.llm_client = llm_client
        self.embeddings = embeddings
        self.user_client = UserClient()
        self.vectorstore = None

    async def __aenter__(self):
        await self.initialize_vectorstore()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def initialize_vectorstore(self, batch_size: int = 50):
        """Initialize vectorstore with all current users."""
        print("🔍 Loading all users for initial vectorstore...")

        # 1. Get all users
        users = self.user_client.get_all_users()

        # 2. Prepare documents
        documents = [
            Document(id=str(user.get('id')), page_content=format_user_document(user))
            for user in users
        ]

        # 3. Split into batches of 100
        batch_size = 100
        batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]

        # 4. Setup vectorstore and add documents in parallel
        self.vectorstore = Chroma(
            collection_name="users",
            embedding_function=self.embeddings
        )
        tasks = [self.vectorstore.aadd_documents(batch) for batch in batches]
        await asyncio.gather(*tasks)
        print(f"✅ Vectorstore initialized with {len(documents)} users")

    async def retrieve_context(self, query: str, k: int = 100, score: float = 0.2) -> str:
        """Retrieve context, with optional automatic vectorstore update."""

        # 1. Update vectorstore with any new/deleted users
        await self._update_vectorstore()

        # 2. Similarity search
        results = await self.vectorstore.asimilarity_search_with_relevance_scores(query, k=k)

        # 3. Collect context parts
        context_parts = []

        # 4. Iterate and filter by score
        for doc, relevance_score in results:
            if relevance_score >= score:
                context_parts.append(doc.page_content)
                print(f"Retrieved (Score: {relevance_score:.3f}): {doc.page_content}")

        # 5. Return joined context
        return "\n\n".join(context_parts)

    async def _update_vectorstore(self):
        # 1. Get all current users
        users = self.user_client.get_all_users()

        # 2. Get all data from vectorstore
        vectorstore_data = self.vectorstore.get()

        # 3. Get set of ids from vectorstore
        vectorstore_ids_set = set(str(user_id) for user_id in vectorstore_data.get("ids", []))

        # 4. Prepare dict from retrieved users
        users_dict = {str(user.get('id')): user for user in users}

        # 5. Prepare set with current user ids
        users_ids_set = set(users_dict.keys())

        # 6. Find new user ids
        new_user_ids = users_ids_set - vectorstore_ids_set

        # 7. Find ids to delete
        ids_to_delete = vectorstore_ids_set - users_ids_set

        # 8. Delete removed users
        if ids_to_delete:
            print(f"🗑️ Removing {len(ids_to_delete)} deleted users from vectorstore")
            self.vectorstore.delete(list(ids_to_delete))

        # 9. Prepare new documents
        new_documents = [
            Document(id=user_id, page_content=format_user_document(users_dict[user_id]))
            for user_id in new_user_ids
        ]

        # 10. Add new documents if any
        if new_documents:
            print(f"➕ Adding {len(new_documents)} new users to vectorstore")
            await self.vectorstore.aadd_documents(new_documents)

    def augment_prompt(self, query: str, context: str) -> str:
        return USER_PROMPT.format(context=context, query=query)

    def generate_answer(self, augmented_prompt: str) -> GroupingResults:
        # 1. Create parser
        parser = PydanticOutputParser(pydantic_object=GroupingResults)

        # 2. Create messages — inject format_instructions directly to avoid template conflicts
        system_content = SYSTEM_PROMPT.replace(
            "{format_instructions}",
            parser.get_format_instructions()
        )
        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=augmented_prompt)
        ]

        # 3. Invoke and parse
        response = self.llm_client.invoke(messages)
        grouping_results: GroupingResults = parser.parse(response.content)

        # 4. Return
        return grouping_results


class OutputGrounder:
    def __init__(self):
        self.user_client = UserClient()

    async def ground_response(self, grouping_results: GroupingResults):
        # 1. Iterate through grouping results
        for grouping_result in grouping_results.grouping_results:
            # 2. Print hobby
            print(f"\n🎯 Hobby: {grouping_result.hobby}")
            # 3. Fetch and print users
            users = await self._find_users(grouping_result.user_ids)
            print(f"Users: {users}")

    async def _find_users(self, ids: list[int]) -> list[dict[str, Any]]:
        async def safe_get_user(user_id: int) -> Optional[dict[str, Any]]:
            try:
                return await self.user_client.aget_user(user_id)
            except Exception as e:
                if "404" in str(e):
                    print(f"User with ID {user_id} is absent (404)")
                    return None
                raise

        # 1. Gather all user fetches in parallel
        tasks = [safe_get_user(user_id) for user_id in ids]
        results = await asyncio.gather(*tasks)

        # 2. Filter out None values
        return [user for user in results if user is not None]


async def main():
    embeddings = AzureOpenAIEmbeddings(
        deployment='text-embedding-3-small-1',
        azure_endpoint=DIAL_URL,
        api_key=SecretStr(API_KEY),
        dimensions=384,
        check_embedding_ctx_length=False
    )
    output_grounder = OutputGrounder()

    async with InputGrounder(embeddings, llm_client) as rag:
        print("Query samples:")
        print(" - I need people who love to go to mountains")
        print(" - Find people who love to watch stars and night sky")
        print(" - I need people to go to fishing together")

        while True:
            user_question = input("> ").strip()
            if user_question.lower() in ['quit', 'exit']:
                break

            # 1. Retrieve context
            context = await rag.retrieve_context(user_question)

            # 2. Augment prompt
            augmented_prompt = rag.augment_prompt(user_question, context)

            # 3. Generate answer
            grouping_results = rag.generate_answer(augmented_prompt)

            # 4. Output grounding
            await output_grounder.ground_response(grouping_results)


if __name__ == "__main__":
    asyncio.run(main())