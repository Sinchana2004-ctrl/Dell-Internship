"""
main.py - Resume Extractor Entry Point
Assignment 2: JsonOutputParser with DeepSeek API
"""

import json
from extractor import extract_resume_info


# ──────────────────────────────────────────────────
# Sample Resume Text (Unstructured)
# ──────────────────────────────────────────────────
SAMPLE_RESUME = """
John Doe
Software Engineer
Email: johndoe@email.com | Phone: +1-555-0101
LinkedIn: linkedin.com/in/johndoe

PROFESSIONAL SUMMARY
Experienced Software Engineer with 6 years of hands-on experience building
scalable web applications, distributed systems, and AI-powered tools.

TECHNICAL SKILLS
- Languages: Python, JavaScript, TypeScript, Java, SQL, Bash
- Frameworks: React, Node.js, FastAPI, Django, Flask, LangChain
- Cloud & DevOps: AWS (EC2, S3, Lambda), Docker, Kubernetes, Terraform
- Databases: PostgreSQL, MongoDB, Redis, Elasticsearch
- Tools: Git, GitHub Actions, Jira, Postman, VS Code

PROFESSIONAL EXPERIENCE

Senior Software Engineer | TechCorp Inc. | January 2022 – Present (3 years)
- Led backend development of real-time data pipeline handling 1M+ events/day
- Architected microservices reducing system latency by 35%
- Mentored 4 junior developers and conducted code reviews

Software Engineer | StartupXYZ | June 2019 – December 2021 (2.5 years)
- Built React dashboard adopted by 10,000+ enterprise customers
- Designed and implemented 15+ RESTful APIs using FastAPI and Django
- Reduced CI/CD pipeline time from 45 to 12 minutes

Junior Developer | WebAgency LLC | July 2018 – May 2019 (10 months)
- Developed client websites using JavaScript, HTML, CSS, and PHP
- Managed MySQL databases and performed performance optimization

EDUCATION

Bachelor of Science in Computer Science
State University, New York | Graduated: May 2018 | GPA: 3.8/4.0

AWS Certified Solutions Architect – Associate | Amazon Web Services | 2021
Google Professional Data Engineer Certification | Google Cloud | 2023
"""


def display_results(result: dict) -> None:
    """Pretty print the extracted resume information."""
    
    print("\n" + "=" * 60)
    print("        EXTRACTED RESUME INFORMATION")
    print("=" * 60)
    
    # Full JSON output
    print("\n--- FULL JSON OUTPUT ---")
    print(json.dumps(result, indent=2))
    
    # Summary view
    print("\n--- SUMMARY ---")
    print(f"Name         : {result.get('name', 'Not found')}")
    print(f"Email        : {result.get('email', 'Not found')}")
    print(f"Experience   : {result.get('experience_years', 0)} years")
    
    skills = result.get('skills', [])
    print(f"Skills ({len(skills)} total):")
    for skill in skills:
        print(f"  - {skill}")
    
    education = result.get('education', [])
    print(f"Education ({len(education)} entries):")
    for edu in education:
        print(f"  - {edu}")
    
    # Check for errors
    if 'extraction_error' in result:
        print(f"\nWARNING - Extraction Error: {result['extraction_error']}")
    
    print("=" * 60)


def main():
    """Main function to run the resume extractor."""
    
    print("=" * 60)
    print("  RESUME INFORMATION EXTRACTOR")
    print("  DeepSeek API + LangChain JsonOutputParser")
    print("=" * 60)
    
    # You can replace SAMPLE_RESUME with any resume text
    resume_text = SAMPLE_RESUME
    
    print(f"\nResume text length: {len(resume_text)} characters")
    
    # Extract information
    result = extract_resume_info(resume_text)
    
    # Display results
    display_results(result)


if __name__ == "__main__":
    main()