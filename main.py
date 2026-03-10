from fastapi import FastAPI, Depends, HTTPException, Path
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timedelta
from jose import jwt, JWTError
from dotenv import load_dotenv
import google.generativeai as genai
import httpx
import os
import time

app = FastAPI(title="AI Trade Opportunities API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment setup
load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

ALGORITHM = "HS256"
TOKEN_EXPIRE_MINUTES = 60

genai.configure(api_key=GEMINI_API_KEY)

# Global objects (optimized)
model = genai.GenerativeModel("gemini-2.5-flash")
client = httpx.AsyncClient(timeout=10)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

rate_store = {}

RATE_LIMIT = 5
RATE_WINDOW = 60

DUCKDUCKGO_URL = "https://api.duckduckgo.com/"

# Pydantic models
class AnalysisResponse(BaseModel):
    sector: str
    report_markdown: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


# JWT token creation
def create_token(username: str):
    expire = datetime.utcnow() + timedelta(minutes=TOKEN_EXPIRE_MINUTES)

    payload = {
        "sub": username,
        "exp": expire
    }

    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

    rate_store[username] = {
        "tokens": RATE_LIMIT,
        "time": time.time()
    }

    return token


# Validate JWT token
async def get_current_user(token: str = Depends(oauth2_scheme)):

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

        username = payload.get("sub")

        if not username:
            raise HTTPException(status_code=401, detail="Invalid token payload")

        return username

    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


# Rate limiting
def check_rate(user: str):

    if user not in rate_store:
        rate_store[user] = {"tokens": RATE_LIMIT, "time": time.time()}

    bucket = rate_store[user]
    now = time.time()

    if now - bucket["time"] > RATE_WINDOW:
        bucket["tokens"] = RATE_LIMIT
        bucket["time"] = now

    if bucket["tokens"] <= 0:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    bucket["tokens"] -= 1


# Fetch market news
async def fetch_market_news(sector: str):

    params = {
        "q": f"{sector} sector india market news",
        "format": "json"
    }

    try:
        response = await client.get(DUCKDUCKGO_URL, params=params)
        data = response.json()

        return data.get("AbstractText", "No data available")

    except httpx.HTTPError:
        return "Market data unavailable"


# AI analysis using Gemini
def analyze_with_gemini(sector: str, news: str):

    prompt = f"""
Analyze the {sector} sector in India using the following market data.

Market Data:
{news}

Provide structured insights including:
1. Market Trends
2. Trade Opportunities
3. Risks and Challenges
4. Investment Outlook
5. Data Reliability Note
"""

    try:
        response = model.generate_content(prompt)
        return getattr(response, "text", "Unexpected AI response format")

    except Exception as e:
        return f"AI analysis unavailable: {str(e)}"


# Generate markdown report
def generate_report(sector: str, analysis: str):

    date = datetime.utcnow().strftime("%Y-%m-%d")

    return f"""
# Market Analysis Report

## Sector
{sector}

## Date
{date}

## AI Market Analysis
{analysis}

## Disclaimer
AI generated insights. Not financial advice.
"""


# Auth routes
@app.get("/guest", response_model=TokenResponse)
async def guest():

    username = f"guest_{int(time.time())}"
    token = create_token(username)

    return {"access_token": token}


@app.post("/login", response_model=TokenResponse)
async def login():

    username = "user"
    token = create_token(username)

    return {"access_token": token}


# Main API
@app.get("/analyze/{sector}", response_model=AnalysisResponse)
async def analyze_sector(
    sector: str = Path(..., min_length=2, max_length=50),
    user: str = Depends(get_current_user)
):

    check_rate(user)

    news = await fetch_market_news(sector)

    analysis = analyze_with_gemini(sector, news)

    report = generate_report(sector, analysis)

    return AnalysisResponse(
        sector=sector,
        report_markdown=report
    )


@app.get("/")
def root():
    return {"message": "AI Trade Opportunities API running"}


# Close HTTP client on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    await client.aclose()
