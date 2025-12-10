from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
import shutil
import os
import uuid
import httpx # 카카오 API 통신용
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import json

# 우리가 만든 모듈들
import db_models 
from database import engine, get_db
import ai_service

# 환경변수 로드
load_dotenv()
KAKAO_CLIENT_ID = os.getenv("KAKAO_CLIENT_ID")
# 카카오 개발자센터에 등록한 주소와 100% 일치해야 함
KAKAO_REDIRECT_URI = "http://127.0.0.1:8000/auth/kakao/callback"

# 신뢰도 임계값 설정 (환경변수로 설정 가능)
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))

# DB 테이블 생성
db_models.Base.metadata.create_all(bind=engine)

app = FastAPI()

# ==========================================
# 🟢 CORS 설정 (프론트엔드 연결)
# ==========================================
origins = [
    "http://localhost:3000",    # 리액트 로컬 주소
    "http://127.0.0.1:3000",
    "*"                         # (테스트용) 모든 곳 허용
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # 👈 위에서 정의한 주소들 허용
    allow_credentials=True,
    allow_methods=["*"],        # 모든 HTTP 메소드(GET, POST 등) 허용
    allow_headers=["*"],        # 모든 헤더 허용
)

# [중요] 이미지 저장 경로 설정 (절대 경로 사용)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOAD_DIR = os.path.join(STATIC_DIR, "images")

# 폴더가 없으면 생성
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 정적 파일(이미지) 제공 설정
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def read_root():
    return {"message": "Draw Your Mind API is running!"}


# ==========================================
# 🟡 1. 카카오 로그인 (인증)
# ==========================================

# 1-1. 사용자가 이 주소로 접속하면 카카오 로그인 페이지로 보냄
@app.get("/auth/kakao/login")
def kakao_login():
    url = (
        f"https://kauth.kakao.com/oauth/authorize"
        f"?client_id={KAKAO_CLIENT_ID}"
        f"&redirect_uri={KAKAO_REDIRECT_URI}"
        f"&response_type=code"
    )
    return RedirectResponse(url)


# 1-2. 카카오 로그인이 끝나면 여기로 돌아옴 (Callback)
@app.get("/auth/kakao/callback")
async def kakao_callback(code: str, db: Session = Depends(get_db)):
    # 1. 받은 '인증 코드'로 '액세스 토큰' 요청
    token_url = "https://kauth.kakao.com/oauth/token"
    data = {
        "grant_type": "authorization_code",
        "client_id": KAKAO_CLIENT_ID,
        "redirect_uri": KAKAO_REDIRECT_URI,
        "code": code,
    }
    
    async with httpx.AsyncClient() as client:
        token_res = await client.post(token_url, data=data)
        token_json = token_res.json()
        
        access_token = token_json.get("access_token")
        if not access_token:
            raise HTTPException(status_code=400, detail="카카오 토큰 발급 실패")

        # 2. '액세스 토큰'으로 '사용자 정보' 요청
        user_info_url = "https://kapi.kakao.com/v2/user/me"
        headers = {"Authorization": f"Bearer {access_token}"}
        
        user_res = await client.get(user_info_url, headers=headers)
        user_json = user_res.json()
        
        # 카카오에서 준 유저 정보 추출
        kakao_id = str(user_json.get("id"))
        properties = user_json.get("properties", {})
        kakao_account = user_json.get("kakao_account", {})
        
        nickname = properties.get("nickname")
        if not nickname:
            nickname = kakao_account.get("profile", {}).get("nickname")
        
        email = kakao_account.get("email")

        # 3. DB에 사용자 저장 (이미 있으면 조회, 없으면 생성)
        user = db.query(db_models.User).filter(db_models.User.kakao_id == kakao_id).first()
        
        if not user:
            # 신규 회원 가입
            user = db_models.User(
                kakao_id=kakao_id,
                nickname=nickname,
                email=email
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            print(f"🎉 신규 회원 가입: {nickname}")
        else:
            print(f"👋 기존 회원 로그인: {nickname}")

        # [핵심 변경] 로그인 성공 후 프론트엔드로 리다이렉트!
        # URL 뒤에 user_id를 붙여서 보냅니다. 프론트에서 이걸 잡아서 저장해야 합니다.
        return RedirectResponse(
            url=f"http://localhost:3000?user_id={user.id}&token={access_token}"
        )


# ==========================================
# 🟢 2. 그림 분석 (개선된 버전)
# ==========================================
@app.post("/analyze")
async def analyze_drawing(
    file: UploadFile = File(...),
    user_id: int = None, # (선택) 로그인했다면 유저 ID를 같이 보냄
    db: Session = Depends(get_db)
):
    """
    그림 업로드 → YOLO 탐지 (신뢰도 필터링) → Groq 심리 분석 (마크다운)
    """
    try:
        # 1. 파일 저장
        extension = file.filename.split(".")[-1]
        filename = f"{uuid.uuid4()}.{extension}"
        file_path = os.path.join(UPLOAD_DIR, filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 2. AI 전체 파이프라인 실행 (신뢰도 필터링 포함)
        analysis_result = ai_service.full_analysis_pipeline(
            image_path=file_path,
            confidence_threshold=CONFIDENCE_THRESHOLD
        )
        
        detection_result = analysis_result["detection"]
        psychology_result = analysis_result["psychology"]

        # ⭐ [로그 출력] 터미널에서 AI 분석 결과 확인하기
        print("\n" + "="*60)
        print("🔍 [객체 탐지 결과 - 필터링됨]:")
        print(json.dumps(detection_result["filtered_summary"], indent=2, ensure_ascii=False))
        print("\n🤖 [Groq 심리 분석 결과 - 마크다운]:")
        print(psychology_result["markdown"])
        print("="*60 + "\n")

        # 3. DB 저장
        new_drawing = db_models.Drawing(
            user_id=user_id,
            drawing_type="HTP_FULL", 
            image_url=f"/static/images/{filename}",
            detection_result=detection_result["filtered_summary"],  # 필터링된 결과만 저장
            analysis_text=psychology_result["markdown"]  # 마크다운 원본 저장
        )
        
        db.add(new_drawing)
        db.commit()
        db.refresh(new_drawing)

        # 4. 응답 반환
        return {
            "status": "success",
            "result": {
                "id": new_drawing.id,
                "user_id": new_drawing.user_id,
                "image_url": new_drawing.image_url,
                "result_image_url": detection_result.get("result_image_url"), # [추가] 박스 그려진 이미지
                
                # 탐지 결과 (필터링된 요약 + 상세 정보)
                "detection": {
                    "summary": detection_result["filtered_summary"],
                    "details": detection_result["details"],
                    "confidence_threshold": CONFIDENCE_THRESHOLD
                },
                
                # 심리 분석 결과 (마크다운 + 파싱된 구조)
                "psychology": {
                    "markdown": psychology_result["markdown"],  # 원본 마크다운
                    "overall_summary": psychology_result["overall_summary"],
                    "positive_traits": psychology_result["positive_traits"],
                    "negative_traits": psychology_result["negative_traits"],
                    "neutral_observations": psychology_result["neutral_observations"],
                    "recommendations": psychology_result["recommendations"]
                }
            }
        }
    
    except Exception as e:
        print(f"❌ Error in analyze_drawing: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"분석 중 오류가 발생했습니다: {str(e)}"
        )


# ==========================================
# 🔵 3. 조회 기능 (결과 페이지 & 마이페이지)
# ==========================================

# 3-1. 특정 분석 결과 조회 (결과 공유/다시보기용)
@app.get("/results/{drawing_id}")
def get_drawing_result(drawing_id: int, db: Session = Depends(get_db)):
    drawing = db.query(db_models.Drawing).filter(db_models.Drawing.id == drawing_id).first()
    
    if not drawing:
        raise HTTPException(status_code=404, detail="결과를 찾을 수 없습니다.")
    
    # analysis_text가 마크다운이므로 파싱해서 반환
    try:
        parsed_psychology = ai_service.parse_markdown_analysis(drawing.analysis_text)
    except:
        # 파싱 실패 시 원본만 반환
        parsed_psychology = {
            "markdown": drawing.analysis_text,
            "overall_summary": "",
            "positive_traits": [],
            "negative_traits": [],
            "neutral_observations": [],
            "recommendations": []
        }
        
    return {
        "id": drawing.id,
        "user_id": drawing.user_id,
        "image_url": drawing.image_url,
        "detection": drawing.detection_result,
        "psychology": parsed_psychology,
        "created_at": drawing.created_at
    }


# 3-2. 특정 유저의 그림 목록 조회 (마이페이지용)
@app.get("/users/{user_id}/drawings")
def get_user_drawings(user_id: int, db: Session = Depends(get_db)):
    drawings = db.query(db_models.Drawing).filter(db_models.Drawing.user_id == user_id).all()
    
    return {
        "user_id": user_id,
        "count": len(drawings),
        "drawings": [
            {
                "id": d.id,
                "image_url": d.image_url,
                "created_at": d.created_at
            }
            for d in drawings
        ]
    }


# ==========================================
# 🆕 4. 신뢰도 임계값 조정 API (선택적)
# ==========================================
@app.get("/settings/confidence-threshold")
def get_confidence_threshold():
    """현재 설정된 신뢰도 임계값 조회"""
    return {
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "description": "0.0 ~ 1.0 사이의 값. 높을수록 엄격한 필터링"
    }