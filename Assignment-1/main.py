# ============================================================
#  AI Text Transformer - Assignment 1
#  Uses: DeepSeek API | LangChain | PromptTemplate | StrOutputParser
# ============================================================

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ─── Load environment variables ────────────────────────────
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# ─── 1. Initialize the DeepSeek Model via OpenAI-compatible API ─
model = ChatOpenAI(
    model="deepseek-chat",                        # DeepSeek's chat model
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com/v1",       # DeepSeek's endpoint
    temperature=0.7,
)

# ─── 2. Create a PromptTemplate ─────────────────────────────
prompt_template = PromptTemplate(
    input_variables=["paragraph"],
    template="""
You are an expert writing assistant. Analyze the paragraph below and respond in plain text only.
Do NOT use JSON, markdown code blocks, or any special formatting characters.

Paragraph:
{paragraph}

Provide your response in this exact format:

SUMMARY:
Write a 3 to 4 line summary of the paragraph here.

TONE:
Write only one word here — either Formal, Casual, or Technical.

IMPROVED VERSION:
Write a clearly improved version of the paragraph here, fixing grammar, flow, and clarity.
""".strip()
)

# ─── 3. Create StrOutputParser ──────────────────────────────
parser = StrOutputParser()

# ─── 4. Chain: prompt | model | parser ──────────────────────
chain = prompt_template | model | parser


# ─── 5. Run the Application ─────────────────────────────────
def transform_text(paragraph: str) -> str:
    """Pass a paragraph through the chain and return plain string output."""
    result = chain.invoke({"paragraph": paragraph})
    return result


def main():
    print("=" * 60)
    print("        AI TEXT TRANSFORMER (Powered by DeepSeek)")
    print("=" * 60)

    # ── Sample paragraph (you can change this) ──────────────
    sample_paragraph = """
    the meeting was like super long and we talked about a lot of stuff. 
    john said we should maybe do better with our reports and stuff coz the boss 
    wasn't happy. we gotta fix things by next week or there could be problems. 
    everyone agreed it was not great and we need to step up our game going forward.
    """.strip()

    print("\n📄 INPUT PARAGRAPH:\n")
    print(sample_paragraph)
    print("\n" + "-" * 60)
    print("⏳ Analyzing with DeepSeek...\n")

    # ── Invoke the chain ────────────────────────────────────
    output = transform_text(sample_paragraph)

    print("✅ TRANSFORMATION RESULT:\n")
    print(output)
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()