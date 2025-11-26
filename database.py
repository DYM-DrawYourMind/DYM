from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL")

# 엔진 생성 (DB와의 연결 통로)
engine = create_engine(SQLALCHEMY_DATABASE_URL)

# 세션 생성 (실제 DB 작업을 수행하는 객체)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 모델들이 상속받을 기본 클래스
Base = declarative_base()

# Dependency (API에서 DB 세션을 쓰고 닫기 위함)
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()