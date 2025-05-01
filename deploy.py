from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine
from llama_index.core import SQLDatabase
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.llms.openai import OpenAI
import openai
import os
from dotenv import load_dotenv

# === Load Environment Variables ===
load_dotenv()

# === Secure Config ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_PORT = os.getenv("DB_PORT", "3306")
SSL_CERT_PATH = os.getenv("SSL_CERT_PATH", "/home/site/wwwroot/certs/azure-cert.pem")



# === Initialize API keys ===
openai.api_key = OPENAI_API_KEY

# === SQLAlchemy Engine + LlamaIndex Adapter ===
engine = create_engine(
    f"mysql+pymysql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
    connect_args={"ssl_ca": SSL_CERT_PATH}
)
sql_db = SQLDatabase(engine)

# === FastAPI App ===
app = FastAPI()

# === Request Schema ===
class QueryRequest(BaseModel):
    question: str
    email: str
    role: str  # STUDENT | TEACHER | ADMIN

# === System Prompt Builder ===
def build_system_prompt(user_role: str, user_email: str) -> str:
    return f"""
You are a university assistant helping students, teachers, and admins interact with a QR-based class attendance system.

Only answer questions using the SQL data provided. Never guess or fabricate. If no data is found, say clearly: "No records found." 
If the question is not permitted based on the user's role, respond: "Access denied. You do not have permission to access this information."

User Identity:
- Role: {user_role}
- Email: {user_email}

---

🧑‍🎓 STUDENT ROLE (read-only, self-scope):
Students can access:
- Their enrolled classes
- Their sessions
- Their attendance
- Their QR scans and session topics

They are forbidden from:
- Other students' data
- Class-wide or system-wide stats
- Teacher or admin information

---

👨‍🏫 TEACHER ROLE:
Teachers can access:
- Their classes and sessions
- Attendance of students in their classes
- Join and attendance requests
- QR codes for their classes

They are forbidden from:
- Other teachers' or students' data
- Admin-level stats

---

🧑‍💼 ADMIN ROLE:
Admins have full access to:
- All classes, users, teachers, sessions
- Approvals, absences, and attendance records

They may not:
- View passwords or perform state-changing operations

---

📋 SQL Interpretation:
- Count absences using status='ABSENT'
- Use exact match for class names and emails
- For sessions, join klass → klass_student → class_session
- Resolve student/teacher via user.email

---

📌 Behavior:
- Never make up SQL fields
- Return "Access denied" if query exceeds permissions
- Assume timezone is Asia/Beirut

🗣️ Natural Responses:
- "No classes scheduled for today."
- "You currently have no absences."
- "No students were absent from that session."
- "There are no upcoming sessions."
"""

# === Main Endpoint ===
@app.post("/query")
async def query_rag(req: QueryRequest):
    try:
        prompt = build_system_prompt(req.role.upper(), req.email)
        llm = OpenAI(model="gpt-4o", temperature=0, system_prompt=prompt)
        query_engine = NLSQLTableQueryEngine(sql_database=sql_db, llm=llm)

        response = query_engine.query(req.question)
        return {"answer": str(response)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
