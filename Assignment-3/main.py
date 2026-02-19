"""
Assignment 3: Product Review Analyzer
Using PydanticOutputParser + DeepSeek API
"""

import os
from typing import List

from dotenv import load_dotenv

# Pydantic
from pydantic import BaseModel, Field, ValidationError

# LangChain
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate

# ─────────────────────────────────────────────
# 1. LOAD ENVIRONMENT VARIABLES
# ─────────────────────────────────────────────
load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not DEEPSEEK_API_KEY:
    raise EnvironmentError(
        "❌ DEEPSEEK_API_KEY not found! Please add it to your .env file."
    )


# ─────────────────────────────────────────────
# 2. DEFINE PYDANTIC MODEL
# ─────────────────────────────────────────────
class ReviewAnalysis(BaseModel):
    """Structured schema for product review analysis."""

    sentiment: str = Field(
        description="Overall sentiment of the review. Must be one of: 'Positive', 'Negative', or 'Neutral'."
    )
    rating: int = Field(
        description="Estimated rating on a scale of 1 to 5 based on the review content.",
        ge=1,
        le=5,
    )
    key_features: List[str] = Field(
        description="List of key product features mentioned or praised in the review."
    )
    improvement_suggestions: List[str] = Field(
        description="List of specific suggestions for improving the product based on the review."
    )


# ─────────────────────────────────────────────
# 3. INITIALIZE DeepSeek via OpenAI-Compatible API
# ─────────────────────────────────────────────
llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=DEEPSEEK_API_KEY,
    openai_api_base="https://api.deepseek.com/v1",
    temperature=0.0,  # deterministic output for structured parsing
)


# ─────────────────────────────────────────────
# 4. SET UP PydanticOutputParser
# ─────────────────────────────────────────────
parser = PydanticOutputParser(pydantic_object=ReviewAnalysis)


# ─────────────────────────────────────────────
# 5. BUILD THE PROMPT TEMPLATE
# ─────────────────────────────────────────────
prompt_template = PromptTemplate(
    input_variables=["review"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
    template="""
You are an expert product review analyst. Analyze the following product review and extract structured insights.

Product Review:
\"\"\"{review}\"\"\"

Instructions:
- Determine the overall sentiment (Positive, Negative, or Neutral).
- Estimate a rating from 1 (very poor) to 5 (excellent) based on the review.
- List all key features of the product that are mentioned or appreciated.
- List all improvement suggestions or complaints that could help improve the product.

{format_instructions}

Important:
- Return ONLY the JSON object. Do NOT include markdown code blocks or extra text.
- Ensure rating is an integer between 1 and 5.
- Ensure sentiment is exactly one of: Positive, Negative, Neutral.
""",
)


# ─────────────────────────────────────────────
# 6. CORE ANALYSIS FUNCTION WITH ERROR HANDLING
# ─────────────────────────────────────────────
def analyze_review(review_text: str) -> ReviewAnalysis | None:
    """
    Analyzes a product review and returns a validated ReviewAnalysis object.

    Args:
        review_text: The raw product review string.

    Returns:
        ReviewAnalysis object or None if parsing fails.
    """
    print("\n" + "=" * 60)
    print("📦 PRODUCT REVIEW ANALYZER")
    print("=" * 60)
    print(f"\n📝 Review:\n{review_text}\n")

    try:
        # Step 1: Format the prompt
        formatted_prompt = prompt_template.format(review=review_text)

        # Step 2: Send to DeepSeek LLM
        print("🔄 Sending to DeepSeek API...")
        raw_response = llm.invoke(formatted_prompt)
        raw_content = raw_response.content
        print(f"\n🤖 Raw LLM Output:\n{raw_content}\n")

        # Step 3: Parse and validate with Pydantic
        print("✅ Parsing and validating output...")
        result = parser.parse(raw_content)
        return result

    except ValidationError as ve:
        print(f"\n❌ Pydantic Validation Error:\n{ve}")
        print("💡 The LLM returned data that doesn't match the expected schema.")
        return None

    except ValueError as ve:
        print(f"\n❌ Parsing Error:\n{ve}")
        print("💡 The LLM output could not be parsed into the expected format.")
        return None

    except Exception as e:
        print(f"\n❌ Unexpected Error:\n{type(e).__name__}: {e}")
        return None


# ─────────────────────────────────────────────
# 7. DISPLAY RESULTS NICELY
# ─────────────────────────────────────────────
def display_results(analysis: ReviewAnalysis | None) -> None:
    """Prints the structured analysis in a readable format."""
    if analysis is None:
        print("\n⚠️  No structured output could be generated.")
        return

    print("\n" + "─" * 60)
    print("📊 STRUCTURED ANALYSIS RESULTS")
    print("─" * 60)

    # Sentiment with emoji
    sentiment_emoji = {"Positive": "😊", "Negative": "😞", "Neutral": "😐"}
    emoji = sentiment_emoji.get(analysis.sentiment, "🔍")
    print(f"\n  {emoji}  Sentiment     : {analysis.sentiment}")

    # Rating with stars
    stars = "⭐" * analysis.rating + "☆" * (5 - analysis.rating)
    print(f"  {stars}  Rating        : {analysis.rating}/5")

    # Key Features
    print(f"\n  🔑 Key Features ({len(analysis.key_features)} found):")
    for i, feature in enumerate(analysis.key_features, 1):
        print(f"      {i}. {feature}")

    # Improvement Suggestions
    print(f"\n  💡 Improvement Suggestions ({len(analysis.improvement_suggestions)} found):")
    if analysis.improvement_suggestions:
        for i, suggestion in enumerate(analysis.improvement_suggestions, 1):
            print(f"      {i}. {suggestion}")
    else:
        print("      None — reviewer had no suggestions!")

    print("\n" + "─" * 60)


# ─────────────────────────────────────────────
# 8. SAMPLE REVIEWS TO TEST
# ─────────────────────────────────────────────
SAMPLE_REVIEWS = [
    # Review 1: Positive with minor complaints
    """
    I absolutely love this wireless headphone! The sound quality is crystal clear and the bass
    is deep without being overpowering. Battery life is incredible — I got 28 hours on a single charge.
    The noise cancellation works perfectly on my daily commute. Build quality feels premium.
    My only gripe is that the carrying case feels cheap for the price, and the ear cushions
    could be softer for extended wear. Also, the companion app crashes occasionally on Android.
    Overall, highly recommend this to anyone looking for quality audio.
    """,

    # Review 2: Negative review
    """
    Extremely disappointed with this product. The laptop overheats after just 30 minutes of use,
    making it impossible to work on demanding tasks. The keyboard feels mushy and unresponsive,
    and the trackpad is horribly inaccurate. Battery barely lasts 3 hours despite claiming 10.
    Customer support was unhelpful when I raised these issues. The only saving grace is the
    bright display, but that's not enough to justify the price. Avoid this product.
    """,

    # Review 3: Neutral/Mixed review
    """
    The coffee maker does its job — makes decent coffee in about 4 minutes. 
    The design is sleek and fits well on my counter. However, it's quite loud during brewing
    which is annoying in the morning. The carafe leaks a bit when pouring.
    It's an okay product for the price range. Nothing extraordinary but gets the job done.
    """,
]


# ─────────────────────────────────────────────
# 9. MAIN ENTRY POINT
# ─────────────────────────────────────────────
def main():
    print("\n🚀 Starting Product Review Analyzer...\n")

    for idx, review in enumerate(SAMPLE_REVIEWS, 1):
        print(f"\n{'#' * 60}")
        print(f"  ANALYZING REVIEW {idx} of {len(SAMPLE_REVIEWS)}")
        print(f"{'#' * 60}")

        result = analyze_review(review.strip())
        display_results(result)

    print("\n✅ All reviews analyzed successfully!\n")


if __name__ == "__main__":
    main()