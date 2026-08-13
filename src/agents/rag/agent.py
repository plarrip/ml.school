import asyncio
import re
from pathlib import Path

import markdown
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.models.lite_llm import LiteLlm, LiteLLMClient
from google.adk.tools.tool_context import ToolContext
from langchain_community.vectorstores import FAISS
from litellm.exceptions import RateLimitError

from common.embeddings import CustomEmbeddingModel

from .prompts import FORMATTER_INSTRUCTIONS, RETRIEVER_INSTRUCTIONS

EMBEDDING_MODEL = "gemini/gemini-embedding-001"

MAX_RATE_LIMIT_RETRIES = 5
_RETRY_DELAY_PATTERN = re.compile(r'"retryDelay":\s*"(\d+)s"')


class RateLimitAwareLiteLLMClient(LiteLLMClient):
    """LiteLLM client that honors the Gemini API's suggested retry delay.

    Google's free-tier Gemini API returns a "retryDelay" hint in the error
    body when it rate-limits a request, but LiteLLM's built-in retry logic
    doesn't parse it and gives up too soon, so we parse it ourselves and
    sleep the exact suggested amount before retrying.
    """

    async def acompletion(self, model, messages, tools, **kwargs):
        for attempt in range(MAX_RATE_LIMIT_RETRIES):
            try:
                return await super().acompletion(model, messages, tools, **kwargs)
            except RateLimitError as error:
                match = _RETRY_DELAY_PATTERN.search(str(error))
                if match is None or attempt == MAX_RATE_LIMIT_RETRIES - 1:
                    raise
                await asyncio.sleep(int(match.group(1)) + 1)
        return None


def retrieve_content(tool_context: ToolContext, question: str) -> list[dict[str, str]]:  # noqa: ARG001
    """Retrieve documentation and reference materials to answer the question."""
    # Let's start by initializing embedding model we want to use.
    custom_embedding_model = CustomEmbeddingModel(model=EMBEDDING_MODEL)

    # We need to define the path where the vector store is located. To ensure the
    # code works regardless of where it's run from, we will use a path relative to
    # the location of this file.
    index_path = (
        Path(__file__).resolve().parents[3] / "data" / "index" / EMBEDDING_MODEL
    )

    # Now, we can load the vector store from disk. This vector store was created
    # by running the Indexing pipeline.
    vector_store = FAISS.load_local(
        str(index_path),
        custom_embedding_model,
        allow_dangerous_deserialization=True,
    )

    # Finally, we can run a similarity search to find the most relevant documents
    # related to the supplied question.
    results = vector_store.similarity_search(
        question,
        k=4,
    )

    return [
        {
            "file": result.metadata["file"],
            "content": result.page_content,
        }
        for result in results
    ]


def markdown_to_html(tool_context: ToolContext, text: str) -> str:  # noqa: ARG001
    """Convert the supplied Markdown text to HTML."""
    try:
        return markdown.markdown(
            text,
            extensions=["fenced_code", "tables", "codehilite", "toc", "sane_lists"],
        )
    except Exception:
        # If the conversion fails for any reason, we will just use the original
        # answer.
        return text


# def base_agent(model: str = "gemini/gemini-3.5-flash"):
def base_agent(model: str = "gemini/gemini-3.5-flash-lite"):
    """Create the Retrieval-Augmented Generation agent."""
    retriever_agent = LlmAgent(
        model=LiteLlm(model=model, llm_client=RateLimitAwareLiteLLMClient()),
        name="retriever",
        description="Answers user questions about the program.",
        instruction=RETRIEVER_INSTRUCTIONS,
        tools=[retrieve_content],
        output_key="answer_markdown",
    )

    formatter_agent = LlmAgent(
        model=LiteLlm(model=model, llm_client=RateLimitAwareLiteLLMClient()),
        name="formatter",
        description="Formats the answers coming from the retriever.",
        instruction=FORMATTER_INSTRUCTIONS,
        tools=[markdown_to_html],
        output_key="answer_html",
    )

    return SequentialAgent(
        name="workflow",
        sub_agents=[retriever_agent, formatter_agent],
        description=(
            "Executes a sequence of retrieval and formatting steps to answer questions."
        ),
    )


root_agent = base_agent()
