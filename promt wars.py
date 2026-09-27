import os
import json
import re

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pypdf import PdfReader
from dotenv import load_dotenv

import google.generativeai as genai


# --------------------------------------------------
# LOAD API KEY
# --------------------------------------------------

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is missing. Please add it to your .env file."
    )

genai.configure(api_key=API_KEY)


# --------------------------------------------------
# CREATE FASTAPI APP
# --------------------------------------------------

app = FastAPI(
    title="AI Student Workspace",
    description="Lecture PDF to Revision Notes and Quiz"
)


# --------------------------------------------------
# STATIC FILES
# --------------------------------------------------

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.get("/")
async def home():
    return FileResponse("static/index.html")


# --------------------------------------------------
# GENERATE NOTES + QUIZ
# --------------------------------------------------

@app.post("/generate")
async def generate(file: UploadFile = File(...)):

    # Check file
    if not file.filename.lower().endswith(".pdf"):
        return {
            "error": "Please upload a PDF file."
        }

    try:

        # ------------------------------------------
        # READ UPLOADED PDF
        # ------------------------------------------

        contents = await file.read()

        with open("uploaded.pdf", "wb") as pdf_file:
            pdf_file.write(contents)

        # ------------------------------------------
        # EXTRACT PDF TEXT
        # ------------------------------------------

        reader = PdfReader("uploaded.pdf")

        text = ""

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        # ------------------------------------------
        # CHECK TEXT
        # ------------------------------------------

        if not text.strip():

            return {
                "error": "Could not extract text from this PDF. Try a text-based PDF."
            }

        # Keep request manageable
        text = text[:30000]

        # ------------------------------------------
        # AI PROMPT
        # ------------------------------------------

        prompt = f"""
You are an AI-powered student study assistant.

Analyze the lecture material below and create useful study material.

LECTURE MATERIAL:
-----------------
{text}
-----------------

Create:

1. REVISION NOTES

Make concise but useful revision notes.

Include:
- Important topics
- Simple explanations
- Important definitions
- Important facts
- Key points to remember

2. PRACTICE QUIZ

Create exactly 5 multiple-choice questions.

Rules:
- Questions must come ONLY from the uploaded lecture material.
- Each question must have exactly 4 options.
- There must be exactly one correct answer.
- Make questions useful for exam preparation.
- Include easy and medium difficulty questions.

Return ONLY valid JSON.

Use exactly this structure:

{{
    "title": "Revision Notes",
    "notes": [
        {{
            "topic": "Topic name",
            "explanation": "Simple explanation",
            "key_points": [
                "Important point 1",
                "Important point 2"
            ]
        }}
    ],
    "quiz": [
        {{
            "question": "Question text",
            "options": [
                "Option A",
                "Option B",
                "Option C",
                "Option D"
            ],
            "answer": 0
        }}
    ]
}}

IMPORTANT:
The answer value must be the option index:
0 = first option
1 = second option
2 = third option
3 = fourth option
"""

        # ------------------------------------------
        # GEMINI
        # ------------------------------------------

        model = genai.GenerativeModel(
            "gemini-1.5-flash"
        )

        response = model.generate_content(prompt)

        result = response.text.strip()

        # ------------------------------------------
        # CLEAN AI RESPONSE
        # ------------------------------------------

        result = re.sub(
            r"```json\s*",
            "",
            result,
            flags=re.IGNORECASE
        )

        result = re.sub(
            r"```\s*",
            "",
            result
        )

        result = result.strip()

        # ------------------------------------------
        # CONVERT JSON
        # ------------------------------------------

        data = json.loads(result)

        # ------------------------------------------
        # RETURN RESULT
        # ------------------------------------------

        return data

    except json.JSONDecodeError:

        return {
            "error": "AI returned an invalid response. Please try again."
        }

    except Exception as e:

        return {
            "error": str(e)
        }


# --------------------------------------------------
# RUNNING MESSAGE
# --------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "running",
        "project": "AI Student Workspace"
    }