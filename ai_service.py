import torch
import os
import pathlib
from pathlib import Path
import openai
from dotenv import load_dotenv

# .env 로드 (API Key 불러오기)
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# [Windows/Mac 경로 호환성]
pathlib.PosixPath = pathlib.Path

# 1. 모델 경로 설정 (weights 폴더)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATHS = {
    "house": os.path.join(BASE_DIR, "weights", "House_best.pt"),
    "tree": os.path.join(BASE_DIR, "weights", "Tree_best.pt"),
    "person": os.path.join(BASE_DIR, "weights", "Person_best.pt")
}

print("Loading AI Models... Please wait.")
models = {}

# 모델 로드 로직 (기존과 동일)
try:
    for key, path in MODEL_PATHS.items():
        if os.path.exists(path):
            models[key] = torch.hub.load('./yolov5', 'custom', path=path, source='local')
            print(f"✅ [{key}] model loaded successfully.")
        else:
            print(f"⚠️ [{key}] model not found at {path}")
except Exception as e:
    print(f"❌ Error loading models: {e}")


# 2. 통합 객체 탐지 함수 (기존과 동일)
def detect_full_htp(image_path: str):
    combined_summary = {} 
    combined_details = [] 

    for subject in ["house", "tree", "person"]:
        model = models.get(subject)
        if not model:
            combined_summary[subject] = {}
            continue

        results = model(image_path)
        df = results.pandas().xyxy[0]
        
        counts = df['name'].value_counts().to_dict()
        combined_summary[subject] = counts
        
        details = df.to_dict(orient="records")
        for item in details:
            item['subject_type'] = subject
        
        combined_details.extend(details)

    print(f"🔍 Combined Detection: {combined_summary}")
    return {"summary": combined_summary, "details": combined_details}


# 3. [핵심] LLM 심리 분석 함수 (ChatGPT 연동)
def analyze_psychology(detection_summary: dict):
    """
    YOLO 탐지 결과를 바탕으로 OpenAI API를 호출하여 심리 분석 보고서를 생성합니다.
    """
    
    # 1. 프롬프트 생성 (AI에게 역할 부여)
    system_prompt = """
    당신은 따뜻하고 전문적인 미술 심리 상담가입니다. 
    사용자가 그린 HTP(집, 나무, 사람) 그림에서 탐지된 객체 목록을 보고, 
    사용자의 심리 상태를 분석하여 부드러운 어조로 설명해주세요.
    
    - 전문 용어를 쓰되, 일반인도 이해하기 쉽게 풀어서 설명하세요.
    - 부정적인 내용보다는 긍정적이고 희망적인 조언을 덧붙여주세요.
    - 결과는 3~4문장의 요약된 줄글로 작성하세요.
    - 탐지된 객체가 별로 없으면 "그림이 단순한 것으로 보아..." 형태로 해석하세요.
    """

    user_prompt = f"""
    [탐지된 객체 데이터]
    - 집(House): {detection_summary.get('house', {})}
    - 나무(Tree): {detection_summary.get('tree', {})}
    - 사람(Person): {detection_summary.get('person', {})}
    
    위 데이터를 바탕으로 심리 분석 결과를 작성해주세요.
    """

    try:
        # 2. ChatGPT API 호출 (최신 gpt-3.5-turbo 또는 gpt-4 사용)
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo", # 또는 "gpt-4"
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7, # 창의성 조절
            max_tokens=500   # 길이 제한
        )
        
        # 3. 결과 텍스트 추출
        analysis_result = response.choices[0].message.content
        return analysis_result

    except Exception as e:
        print(f"❌ OpenAI API Error: {e}")
        return "죄송합니다. 현재 심리 분석 AI 모델 연결이 원활하지 않아 상세 분석을 제공할 수 없습니다. (기본 탐지 결과만 확인 가능)"