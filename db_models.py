from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from database import Base

# 1. 사용자 테이블
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    kakao_id = Column(String, unique=True, index=True, nullable=False) # 카카오 고유 ID
    nickname = Column(String, nullable=True)
    email = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # 관계 설정 (User : Drawing = 1 : N)
    drawings = relationship("Drawing", back_populates="owner")

# 2. 그림 및 분석 결과 테이블
class Drawing(Base):
    __tablename__ = "drawings"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    
    drawing_type = Column(String, nullable=False) # 'HOUSE', 'TREE', 'PERSON'
    image_url = Column(String, nullable=False)    # 이미지 저장 경로
    
    # YOLO 객체 탐지 결과 저장 (핵심!)
    # 예: {"window": 2, "door": 1} 와 같은 JSON 데이터를 그대로 저장
    detection_result = Column(JSON, nullable=True) 
    
    # LLM 심리 분석 결과
    analysis_text = Column(Text, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # 관계 설정
    owner = relationship("User", back_populates="drawings")