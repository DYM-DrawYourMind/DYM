# Draw Your Mind - AI 심리 분석 서비스 개발 가이드

이 문서는 **Draw Your Mind** 프로젝트의 백엔드(FastAPI + PostgreSQL + YOLOv5 + OpenAI) 개발 환경 세팅 및 실행 방법을 설명합니다.

## 🛠 1. 개발 환경 요구사항 (Prerequisites)

* **OS:** macOS / Windows / Linux
* **Python:** 3.10 이상 (3.13 권장)
* **Docker:** 필수 (PostgreSQL DB 실행용)
* **Git:** 필수

---

## 📂 2. 프로젝트 폴더 구조 (Directory Structure)

프로젝트를 클론받거나 생성할 때 아래 폴더 구조를 유지해야 합니다.

```text
Drawyourmind/
├── .venv/                  # Python 가상환경
├── weights/                # YOLO 모델 파일 (.pt) 저장소 (※폴더명 주의)
│   ├── House_best.pt
│   ├── Tree_best.pt
│   └── Person_best.pt
├── yolov5/                 # YOLOv5 엔진 (자동 생성 또는 git clone)
├── static/                 # 업로드된 이미지 저장소
│   └── images/
├── main.py                 # FastAPI 메인 실행 파일
├── ai_service.py           # AI (YOLO + LLM) 처리 로직
├── database.py             # DB 연결 설정
├── db_models.py            # DB 테이블 모델 정의 (※ models.py 아님)
├── init_db.py              # 테이블 초기 생성 스크립트
├── docker-compose.yml      # DB 컨테이너 설정
├── .env                    # 환경변수 (API Key, DB 주소 등)
└── requirements.txt        # 패키지 목록

```
---

## 3. 설치 및 세팅 순서
### 3-1 ) 가상환경 세팅 및 패키지 설치
* 1. 가상환경 생성
```python -m venv .venv```

* 2. 가상환경 활성화
  -  Mac/Linux:
```source .venv/bin/activate```
-  Windows:
    ```venv\Scripts\activate```

# 3. 필수 라이브러리 설치
```
pip install fastapi uvicorn sqlalchemy psycopg[binary] python-dotenv python-multipart httpx openai torch torchvision opencv-python pandas requests PyYAML tqdm seaborn ultralytics
```

## 3-2) yoloV5 설정
* 인터넷 연결 문제 방지를 위해 YOLOv5 레포지토리를 로컬에 복제합니다.

# 프로젝트 루트 폴더에서 실행

 ```
git clone [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
pip install -r yolov5/requirements.txt
 ```

 ## 3-3) 데이터베이스(PostgreSQL) 실행
 * Docker를 사용하여 DB를 백그라운드에서 실행

 ```
version: '3.8'

services:
  db:
    image: postgres:15
    container_name: draw_your_mind_db
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_USER=jaewoong
      - POSTGRES_PASSWORD=mypassword
      - POSTGRES_DB=postgres
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
 ```

 * 실행 명령어
 ```docker-compose up -d```

 ## 3-4) env 파일 설정
 ```
 # 1. DB 접속 정보 (Docker 설정과 일치해야 함)
# 형식: postgresql+psycopg://아이디:비밀번호@주소:포트/DB이름
DATABASE_URL=postgresql+psycopg://jaewoong:mypassword@localhost:5432/postgres

# 2. OpenAI API 키 (심리 분석용)
OPENAI_API_KEY=sk-여기에_당신의_키를_입력하세요

# 3. 카카오 로그인 설정 (Kakao Developers)
KAKAO_CLIENT_ID=여기에_REST_API_키를_입력하세요
  ```

## 3-5) DB 테이블 생성(초김화)
* 서버 실행 전, 테이블(users, drawings)을 생성하기 위해 초기화 스크립트를 한 번 실행합니다
```python init_db.py```    
성공 시: "✅ 성공! users, drawings 테이블이 생성되었습니다." 메시지 출력

## 3-6) 서버 실행
```uvicorn main:app --reload```
* Server URL: http://127.0.0.1:8000        

* API Docs (Swagger): https://www.google.com/search?q=http://127.0.0.1:8000/docs

