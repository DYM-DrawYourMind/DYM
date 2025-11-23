from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime
from app.db import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    kakao_id = Column(String, unique=True, index=True, nullable=True)
    email = Column(String, unique=True, index=True, nullable=True)
    nickname = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "kakao_id": self.kakao_id,
            "email": self.email,
            "nickname": self.nickname,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
