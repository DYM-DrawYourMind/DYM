from database import engine
from sqlalchemy import text
import db_models # 설계도 가져오기

print("--- 🩺 진단 시작 ---")

# 1. 설계도 확인
tables = list(db_models.Base.metadata.tables.keys())
print(f"1. 설계도에서 발견된 테이블: {tables}")

if not tables:
    print("🚨 [치명적 오류] 설계도가 비어있습니다! db_models.py가 제대로 연결되지 않았습니다.")
    exit()

# 2. 테이블 생성 시도
print(f"2. DB 주소({engine.url})에 테이블 생성을 요청합니다...")
db_models.Base.metadata.drop_all(bind=engine) # 기존꺼 삭제
db_models.Base.metadata.create_all(bind=engine) # 새로 생성

# 3. 파이썬 입장에서 확인 (Self-Check)
print("3. 파이썬이 DB 내부를 직접 확인합니다...")
with engine.connect() as connection:
    result = connection.execute(text("SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname='public'"))
    created_tables = [row[0] for row in result]
    
    print(f"✅ DB 안에 실제로 생성된 테이블 목록: {created_tables}")
    
    if "drawings" in created_tables:
        print("🎉 파이썬 쪽에서는 생성이 확인되었습니다!")
    else:
        print("😱 파이썬이 명령을 내렸는데도 테이블이 안 생겼습니다. (매우 이상함)")