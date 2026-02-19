"""
extractor.py - Core resume extraction module
Uses DeepSeek API + LangChain JsonOutputParser
"""

import os
from typing import List
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

# Load .env file
load_dotenv()


# ──────────────────────────────────────────────────
# 1. Define the Output Schema using Pydantic
# ──────────────────────────────────────────────────
class ResumeInfo(BaseModel):
    """Schema defining the structure of extracted resume data."""
    
    name: str = Field(
        description="Full name of the candidate"
    )
    email: str = Field(
        description="Email address of the candidate"
    )
    skills: List[str] = Field(
        description="List of all technical and soft skills mentioned"
    )
    experience_years: int = Field(
        description="Total years of professional work experience as a whole number"
    )
    education: List[str] = Field(
        description="List of educational degrees and certifications with institutions"
    )


# ──────────────────────────────────────────────────
# 2. Initialize DeepSeek LLM
# ──────────────────────────────────────────────────
def get_llm() -> ChatOpenAI:
    """
    Initialize the DeepSeek LLM using OpenAI-compatible interface.
    DeepSeek exposes an OpenAI-compatible API at api.deepseek.com
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")
    
    if not api_key:
        raise ValueError(
            "DEEPSEEK_API_KEY not found!\n"
            "Please add it to your .env file:\n"
            "DEEPSEEK_API_KEY=your_key_here"
        )
    
    llm = ChatOpenAI(
        model="deepseek-chat",       # DeepSeek's main model
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",  # DeepSeek API endpoint
        temperature=0                # 0 = deterministic, best for extraction
    )
    
    return llm


# ──────────────────────────────────────────────────
# 3. Build the Extraction Chain
# ──────────────────────────────────────────────────
def build_extraction_chain():
    """
    Builds a LangChain pipeline:
    ChatPromptTemplate | DeepSeek LLM | JsonOutputParser
    """
    llm = get_llm()
    
    # Initialize parser with the Pydantic schema
    # This auto-generates format instructions for the LLM
    parser = JsonOutputParser(pydantic_object=ResumeInfo)
    
    # Get auto-generated format instructions from the parser
    format_instructions = parser.get_format_instructions()
    
    # Build structured prompt
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert resume parser and information extraction specialist.

Your job is to carefully read a resume and extract specific structured information.

RULES:
- Extract ONLY information explicitly present in the resume
- For experience_years: calculate total work experience in whole years
- For skills: include all mentioned technologies, tools, languages, and soft skills
- For education: include degrees, certifications, and institutions
- If any field is missing, use: "" for strings, 0 for numbers, [] for lists

{format_instructions}

IMPORTANT: Return ONLY valid JSON. No explanations, no markdown, just JSON."""
        ),
        (
            "human",
            """Extract structured information from this resume:

==================== RESUME ====================
{resume_text}
================================================

Extract and return the JSON now:"""
        )
    ])
    
    # Inject format instructions as a partial variable
    prompt = prompt.partial(format_instructions=format_instructions)
    
    # Create the chain using LangChain's pipe operator
    # Flow: prompt → llm → parser
    chain = prompt | llm | parser
    
    return chain


# ──────────────────────────────────────────────────
# 4. Main Extraction Function
# ──────────────────────────────────────────────────
def extract_resume_info(resume_text: str) -> dict:
    """
    Extract structured information from raw resume text.
    
    Args:
        resume_text (str): Unstructured resume text
        
    Returns:
        dict: Structured resume data with fields:
              name, email, skills, experience_years, education
    """
    if not resume_text or not resume_text.strip():
        raise ValueError("Resume text cannot be empty!")
    
    chain = build_extraction_chain()
    
    try:
        print("Sending to DeepSeek API for extraction...")
        result = chain.invoke({"resume_text": resume_text})
        print("Extraction successful!")
        return result
        
    except Exception as e:
        print(f"Error during extraction: {type(e).__name__}: {e}")
        print("Returning default structure with error info...")
        
        # Return safe fallback structure on any error
        return {
            "name": "",
            "email": "",
            "skills": [],
            "experience_years": 0,
            "education": [],
            "extraction_error": str(e)
        }