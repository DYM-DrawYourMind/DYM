import torch
import os
import pathlib
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import re
import cv2
import numpy as np

# .env 로드 (API Key 불러오기)
load_dotenv()

# Groq 클라이언트 설정
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("Groq_API_KEY")
)

# [Windows/Mac 경로 호환성]
pathlib.PosixPath = pathlib.Path

# 1. 모델 경로 설정 (weights 폴더)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATHS = {
    "house": os.path.join(BASE_DIR, "weights", "House_best.pt"),
    "tree": os.path.join(BASE_DIR, "weights", "Tree_best.pt"),
    "person": os.path.join(BASE_DIR, "weights", "Person_best.pt")
}

# 신뢰도 임계값 설정 (0.0 ~ 1.0)
CONFIDENCE_THRESHOLD = 0.75  # 75% 이상만 유효한 탐지로 인정

print("Loading AI Models... Please wait.")
models = {}

# 모델 로드 로직
try:
    for key, path in MODEL_PATHS.items():
        if os.path.exists(path):
            models[key] = torch.hub.load('./yolov5', 'custom', path=path, source='local')
            print(f"✅ [{key}] model loaded successfully.")
        else:
            print(f"⚠️ [{key}] model not found at {path}")
except Exception as e:
    print(f"❌ Error loading models: {e}")


#  OpenCV 전처리 함수
def preprocess_image(image_path: str):
    """
    YOLO 모델 입력 전 OpenCV 전처리 수행
    
    Args:
        image_path: 이미지 파일 경로
    
    Returns:
        numpy.ndarray: 전처리된 이미지 배열
    """
    # 1. 이미지 파일 읽기 (OpenCV는 BGR 포맷으로 읽음)
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"이미지를 불러올 수 없습니다: {image_path}")
    
    # 2. BGR → RGB 색상 공간 변환
    # OpenCV 기본 포맷은 BGR이지만 YOLO는 RGB 기준으로 동작하기 때문에 변환
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 3. GaussianBlur로 노이즈 제거
    # 사용자가 촬영해서 올린 그림 이미지에 노이즈가 섞여
    # YOLO가 불필요한 패턴을 객체로 오탐하는 문제를 방지
    # 커널 크기 (3, 3): 노이즈는 제거하되 객체 윤곽은 유지하는 수준
    image = cv2.GaussianBlur(image, (3, 3), 0)
    
    print(f"✅ [전처리 완료] 이미지 크기: {image.shape}, 포맷: RGB")
    
    return image


# 2. 신뢰도 필터링이 적용된 객체 탐지 함수
def detect_full_htp(image_path: str, confidence_threshold: float = CONFIDENCE_THRESHOLD):
    """
    HTP 객체 탐지 with 신뢰도 필터링
    
    Args:
        image_path: 이미지 경로
        confidence_threshold: 신뢰도 임계값 (기본 0.75)
    
    Returns:
        dict: {
            "summary": {객체별 개수},
            "details": [탐지 상세 정보],
            "filtered_summary": {신뢰도 필터링된 요약}
        }
    """
    combined_summary = {}
    combined_details = []
    filtered_summary = {}  # 신뢰도 높은 결과만

    # [추가] YOLO 입력 전 OpenCV 전처리 수행
    preprocessed_image = preprocess_image(image_path)

    for subject in ["house", "tree", "person"]:
        model = models.get(subject)
        if not model:
            combined_summary[subject] = {}
            filtered_summary[subject] = {}
            continue

        # [변경] image_path 대신 전처리된 numpy 배열을 YOLO에 입력
        results = model(preprocessed_image)
        df = results.pandas().xyxy[0]

        # 전체 탐지 결과
        counts = df['name'].value_counts().to_dict()
        combined_summary[subject] = counts

        # 신뢰도 필터링 적용
        # '전체(Whole)' 객체(class 0)에 대해서만 더 엄격한 기준 적용
        def strict_filter(row):
            if row['class'] == 0:  # 전체 객체 (집전체, 나무전체, 사람전체)
                return row['confidence'] >= max(0.75, confidence_threshold)
            else:
                return row['confidence'] >= confidence_threshold

        df_filtered = df[df.apply(strict_filter, axis=1)]

        # '전체(Whole)' 객체(class 0)가 없으면 부속 부위 무시
        has_whole_object = not df_filtered[df_filtered['class'] == 0].empty

        if not has_whole_object:
            print(f"⚠️ [{subject}] '전체(Whole)' 객체가 감지되지 않아 부속 부위 {len(df_filtered)}개를 무시합니다.")
            df_filtered = df_filtered.iloc[0:0]

        filtered_counts = df_filtered['name'].value_counts().to_dict()
        filtered_summary[subject] = filtered_counts

        # 상세 정보 (필터링된 것만)
        details = df_filtered.to_dict(orient="records")
        for item in details:
            item['subject_type'] = subject

        combined_details.extend(details)

        # 로그 출력
        total_detected = len(df)
        valid_detected = len(df_filtered)
        if total_detected > valid_detected:
            print(f"⚠️ [{subject}] {total_detected}개 탐지 → {valid_detected}개 유효 (신뢰도 {confidence_threshold*100}% 이상)")

    # 중복 탐지 제거 (NMS - Non-Maximum Suppression)
    # 서로 다른 모델(House vs Person)이 같은 위치를 탐지했을 때,
    # 신뢰도가 더 높은 것만 남기고 나머지는 제거
    def calculate_iomin(box1, box2):
        # Intersection over Minimum Area (IoMin) 계산
        # 포함 관계(큰 박스가 작은 박스를 삼키는 경우)를 감지하기 위함
        # 일반 IoU는 박스 크기 차이가 클 때 포함 관계를 잘 못 잡는 한계가 있어
        # 더 작은 박스의 면적을 기준으로 겹침을 계산하는 IoMin 방식을 직접 구현
        x1 = max(box1['xmin'], box2['xmin'])
        y1 = max(box1['ymin'], box2['ymin'])
        x2 = min(box1['xmax'], box2['xmax'])
        y2 = min(box1['ymax'], box2['ymax'])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1['xmax'] - box1['xmin']) * (box1['ymax'] - box1['ymin'])
        area2 = (box2['xmax'] - box2['xmin']) * (box2['ymax'] - box2['ymin'])

        min_area = min(area1, area2)
        if min_area == 0:
            return 0
        return intersection / min_area

    # 신뢰도 순으로 정렬 (높은게 우선)
    combined_details.sort(key=lambda x: x['confidence'], reverse=True)

    final_details = []
    for current in combined_details:
        is_duplicate = False
        for kept in final_details:
            iomin = calculate_iomin(current, kept)
            # IoMin이 0.6 이상이면 (상당 부분 겹치거나 포함됨),
            # 그리고 서로 다른 주제라면(House vs Person) 중복/오탐지로 간주하고 제거
            if iomin > 0.6 and current['subject_type'] != kept['subject_type']:
                is_duplicate = True
                print(f"🗑️ 중복/포함 제거: [{current['subject_type']}]{current['name']}({current['confidence']:.2f})가 "
                      f"[{kept['subject_type']}]{kept['name']}({kept['confidence']:.2f})와 겹침 (IoMin: {iomin:.2f})")
                break

        if not is_duplicate:
            final_details.append(current)

    combined_details = final_details

    # filtered_summary 재구성 (NMS 적용 후)
    filtered_summary = {"house": {}, "tree": {}, "person": {}}
    for det in combined_details:
        subj = det['subject_type']
        name = det['name']
        filtered_summary[subj][name] = filtered_summary[subj].get(name, 0) + 1

    result_image_url = None

    print(f"🔍 Combined Detection (Filtered & NMS): {filtered_summary}")

    return {
        "summary": combined_summary,
        "details": combined_details,
        "filtered_summary": filtered_summary,
        "result_image_url": result_image_url
    }


# 3. LLM 심리 분석 함수 - 필터링된 데이터 사용
def analyze_psychology(detection_summary: dict, filtered_summary: dict):
    """
    신뢰도 높은 YOLO 탐지 결과를 바탕으로 Groq API를 호출하여
    마크다운 형식의 심리 분석 보고서를 생성합니다.
    """

    system_prompt = """
당신은 HTP(집-나무-사람) 그림 심리 분석 전문가입니다.
분석 결과를 반드시 아래 마크다운 형식으로만 출력하세요.

# 전반적인 심리 상태
(2-3문장으로 요약)

## 😊 긍정적 특성
### 특성명 [강도: 85]
설명 (1-2문장)

### 특성명 [강도: 70]
설명 (1-2문장)

## 😔 주의가 필요한 부분
### 특성명 [강도: 60]
설명 (1-2문장)

## 📌 중립적 관찰
- 관찰 내용 1
- 관찰 내용 2

## 💡 심리적 조언
- 조언 1
- 조언 2

**규칙:**
- 강도(intensity)는 0-100 사이의 정수 (대괄호 안에 표시)
- 긍정적 특성: 강도가 높을수록 더 강함 (80-100: 매우 강함)
- 주의가 필요한 부분: 강도가 높을수록 더 주의 필요 (80-100: 심각)
- 각 섹션은 최소 1개, 최대 5개 항목
- 부정적인 내용도 희망적인 어조로 작성
- 전문 용어를 쓰되 이해하기 쉽게 설명
- 탐지된 객체가 없으면 "그림이 단순하거나 명확하지 않아..." 형태로 일반적인 해석 제공
"""

    house_info = filtered_summary.get('house', {})
    tree_info = filtered_summary.get('tree', {})
    person_info = filtered_summary.get('person', {})

    # 집 모델에서 발견된 나무/꽃 정보를 나무 정보에 합침
    if '나무' in house_info:
        tree_info['나무(집모델감지)'] = house_info['나무']
    if '꽃' in house_info:
        tree_info['꽃(집모델감지)'] = house_info['꽃']

    house_detected = bool(house_info)
    tree_detected = bool(tree_info)
    person_detected = bool(person_info)

    user_prompt = f"""
[고신뢰도 객체 탐지 결과]
- 집(House): {"탐지됨 " + str(house_info) if house_detected else "탐지 안됨"}
- 나무(Tree): {"탐지됨 " + str(tree_info) if tree_detected else "탐지 안됨"}
- 사람(Person): {"탐지됨 " + str(person_info) if person_detected else "탐지 안됨"}

주의: 위 결과는 신뢰도 75% 이상인 탐지만 포함되어 있습니다.

위 탐지 결과를 바탕으로 심리 분석을 마크다운 형식으로 출력하세요.
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.6,
            max_tokens=2000
        )

        markdown_result = response.choices[0].message.content
        parsed_data = parse_markdown_analysis(markdown_result)

        print("✅ Psychology Analysis Success (Markdown)")
        return parsed_data

    except Exception as e:
        print(f"❌ Groq API Error: {e}")
        return {
            "markdown": "현재 AI 분석 서비스에 연결할 수 없습니다.",
            "overall_summary": "분석을 생성할 수 없습니다.",
            "positive_traits": [],
            "negative_traits": [],
            "neutral_observations": [f"오류: {str(e)}"],
            "recommendations": ["잠시 후 다시 시도해주세요."]
        }


# 4. 마크다운 파싱 함수
def parse_markdown_analysis(markdown_text: str):
    """
    마크다운 텍스트를 파싱하여 구조화된 데이터로 변환
    """
    result = {
        "markdown": markdown_text,
        "overall_summary": "",
        "positive_traits": [],
        "negative_traits": [],
        "neutral_observations": [],
        "recommendations": []
    }

    lines = markdown_text.split('\n')
    current_section = None
    current_trait = None

    for line in lines:
        line = line.strip()

        if line.startswith('# 전반적인 심리 상태'):
            current_section = 'overall'
            continue
        elif '긍정적 특성' in line or '😊' in line:
            current_section = 'positive'
            continue
        elif '주의가 필요한' in line or '😔' in line:
            current_section = 'negative'
            continue
        elif '중립적 관찰' in line or '📌' in line:
            current_section = 'neutral'
            continue
        elif '심리적 조언' in line or '💡' in line:
            current_section = 'recommendations'
            continue

        if current_section == 'overall' and line and not line.startswith('#'):
            result['overall_summary'] += line + ' '

        elif current_section == 'positive' and line.startswith('###'):
            match = re.search(r'###\s*(.+?)\s*\[강도:\s*(\d+)\]', line)
            if match:
                trait_name = match.group(1).strip()
                intensity = int(match.group(2))
                current_trait = {
                    "trait": trait_name,
                    "description": "",
                    "intensity": intensity
                }

        elif current_section == 'positive' and current_trait and line and not line.startswith('#'):
            current_trait['description'] += line + ' '
            if line.endswith('.') or line.endswith('요'):
                result['positive_traits'].append(current_trait)
                current_trait = None

        elif current_section == 'negative' and line.startswith('###'):
            match = re.search(r'###\s*(.+?)\s*\[강도:\s*(\d+)\]', line)
            if match:
                trait_name = match.group(1).strip()
                intensity = int(match.group(2))
                current_trait = {
                    "trait": trait_name,
                    "description": "",
                    "intensity": intensity
                }

        elif current_section == 'negative' and current_trait and line and not line.startswith('#'):
            current_trait['description'] += line + ' '
            if line.endswith('.') or line.endswith('요'):
                result['negative_traits'].append(current_trait)
                current_trait = None

        elif current_section == 'neutral' and line.startswith('-'):
            result['neutral_observations'].append(line[1:].strip())

        elif current_section == 'recommendations' and line.startswith('-'):
            result['recommendations'].append(line[1:].strip())

    result['overall_summary'] = result['overall_summary'].strip()

    return result


# 5. 전체 분석 파이프라인 함수
def full_analysis_pipeline(image_path: str, confidence_threshold: float = CONFIDENCE_THRESHOLD):
    """
    이미지 업로드 → OpenCV 전처리 → YOLO 탐지 → LLM 분석 전체 파이프라인
    """
    # 1단계: 객체 탐지 (OpenCV 전처리 + 신뢰도 필터링 적용)
    detection_result = detect_full_htp(image_path, confidence_threshold)

    # 2단계: 심리 분석 (필터링된 결과만 사용)
    psychology_result = analyze_psychology(
        detection_summary=detection_result["summary"],
        filtered_summary=detection_result["filtered_summary"]
    )

    return {
        "detection": detection_result,
        "psychology": psychology_result
    }


# 6. 단일 객체 탐지 함수
def detect_single_object(image_path: str, subject: str, confidence_threshold: float = CONFIDENCE_THRESHOLD):
    """
    단일 주제(House, Tree, Person)에 대한 객체 탐지
    """
    subject = subject.lower()
    model = models.get(subject)

    if not model:
        print(f"❌ Invalid subject: {subject}")
        return None

    # [추가] OpenCV 전처리 적용
    preprocessed_image = preprocess_image(image_path)

    # [변경] 전처리된 이미지를 YOLO에 입력
    results = model(preprocessed_image)
    df = results.pandas().xyxy[0]

    def strict_filter(row):
        if row['class'] == 0:
            return row['confidence'] >= max(0.75, confidence_threshold)
        else:
            return row['confidence'] >= confidence_threshold

    df_filtered = df[df.apply(strict_filter, axis=1)]

    has_whole_object = not df_filtered[df_filtered['class'] == 0].empty
    if not has_whole_object:
        print(f"⚠️ [{subject}] '전체(Whole)' 객체가 감지되지 않아 부속 부위 {len(df_filtered)}개를 무시합니다.")
        df_filtered = df_filtered.iloc[0:0]

    filtered_counts = df_filtered['name'].value_counts().to_dict()

    details = df_filtered.to_dict(orient="records")
    for item in details:
        item['subject_type'] = subject

    details.sort(key=lambda x: x['confidence'], reverse=True)

    return {
        "summary": {subject: filtered_counts},
        "filtered_summary": {subject: filtered_counts},
        "details": details,
        "result_image_url": None
    }


# 7. 단일 객체 심리 분석 함수
def analyze_single_psychology(filtered_summary: dict, subject: str):
    """
    단일 주제에 대한 심리 분석
    """
    subject_korean = {"house": "집", "tree": "나무", "person": "사람"}.get(subject.lower(), subject)

    system_prompt = f"""
당신은 {subject_korean}({subject}) 그림 심리 분석 전문가입니다.
분석 결과를 반드시 아래 마크다운 형식으로만 출력하세요.

# 전반적인 심리 상태
(2-3문장으로 요약)

## 😊 긍정적 특성
### 특성명 [강도: 85]
설명 (1-2문장)

## 😔 주의가 필요한 부분
### 특성명 [강도: 60]
설명 (1-2문장)

## 📌 세부 관찰
- 관찰 내용 1
- 관찰 내용 2

## 💡 심리적 조언
- 조언 1

**규칙:**
- 강도(intensity)는 0-100 사이의 정수
- {subject_korean}의 특징에 집중하여 분석하세요.
- 탐지된 객체가 없으면 "그림이 단순하거나 명확하지 않아..." 형태로 일반적인 해석 제공
"""

    subject_info = filtered_summary.get(subject.lower(), {})
    is_detected = bool(subject_info)

    user_prompt = f"""
[탐지된 {subject_korean} 객체 정보]
- {subject_korean}: {"탐지됨 " + str(subject_info) if is_detected else "탐지 안됨"}

위 정보를 바탕으로 심리 분석을 수행하세요.
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.6,
            max_tokens=1500
        )

        markdown_result = response.choices[0].message.content
        parsed_data = parse_markdown_analysis(markdown_result)
        return parsed_data

    except Exception as e:
        print(f"❌ Groq API Error: {e}")
        return {
            "markdown": "분석 실패",
            "overall_summary": "오류 발생",
            "positive_traits": [],
            "negative_traits": [],
            "neutral_observations": [],
            "recommendations": []
        }


# 8. 단일 분석 파이프라인
def single_analysis_pipeline(image_path: str, subject: str, confidence_threshold: float = CONFIDENCE_THRESHOLD):
    """
    단일 주제 분석 파이프라인
    이미지 업로드 → OpenCV 전처리 → YOLO 탐지 → LLM 분석
    """
    detection_result = detect_single_object(image_path, subject, confidence_threshold)
    if not detection_result:
        raise ValueError("Invalid subject or model error")

    psychology_result = analyze_single_psychology(detection_result["filtered_summary"], subject)

    return {
        "detection": detection_result,
        "psychology": psychology_result
    }


# 6. [New] 단일 객체 탐지 함수
def detect_single_object(image_path: str, subject: str, confidence_threshold: float = CONFIDENCE_THRESHOLD):
    """
    단일 주제(House, Tree, Person)에 대한 객체 탐지
    """
    subject = subject.lower()
    model = models.get(subject)
    
    if not model:
        print(f"❌ Invalid subject: {subject}")
        return None

    results = model(image_path)
    df = results.pandas().xyxy[0]
    
    # 신뢰도 필터링
    def strict_filter(row):
        if row['class'] == 0:  # 전체 객체
            return row['confidence'] >= max(0.75, confidence_threshold)
        else:
            return row['confidence'] >= confidence_threshold

    df_filtered = df[df.apply(strict_filter, axis=1)]
    
    # 전체 객체 확인
    has_whole_object = not df_filtered[df_filtered['class'] == 0].empty
    if not has_whole_object:
        print(f"⚠️ [{subject}] '전체(Whole)' 객체가 감지되지 않아 부속 부위 {len(df_filtered)}개를 무시합니다.")
        df_filtered = df_filtered.iloc[0:0]

    filtered_counts = df_filtered['name'].value_counts().to_dict()
    
    # 상세 정보
    details = df_filtered.to_dict(orient="records")
    for item in details:
        item['subject_type'] = subject
        
    # NMS (단일 모델 내에서도 중복이 있을 수 있으므로 적용 - IoU 기반)
    details.sort(key=lambda x: x['confidence'], reverse=True)

    return {
        "summary": {subject: filtered_counts}, # 호환성을 위해 구조 유지
        "filtered_summary": {subject: filtered_counts},
        "details": details,
        "result_image_url": None
    }


# 7. [New] 단일 객체 심리 분석 함수
def analyze_single_psychology(filtered_summary: dict, subject: str):
    """
    단일 주제에 대한 심리 분석
    """
    subject_korean = {"house": "집", "tree": "나무", "person": "사람"}.get(subject.lower(), subject)
    
    system_prompt = f"""
당신은 {subject_korean}({subject}) 그림 심리 분석 전문가입니다.
분석 결과를 반드시 아래 마크다운 형식으로만 출력하세요.

# 전반적인 심리 상태
(2-3문장으로 요약)

## 😊 긍정적 특성
### 특성명 [강도: 85]
설명 (1-2문장)

## 😔 주의가 필요한 부분
### 특성명 [강도: 60]
설명 (1-2문장)

## 📌 세부 관찰
- 관찰 내용 1
- 관찰 내용 2

## 💡 심리적 조언
- 조언 1

**규칙:**
- 강도(intensity)는 0-100 사이의 정수
- {subject_korean}의 특징에 집중하여 분석하세요.
- 탐지된 객체가 없으면 "그림이 단순하거나 명확하지 않아..." 형태로 일반적인 해석 제공
"""

    subject_info = filtered_summary.get(subject.lower(), {})
    is_detected = bool(subject_info)
    
    user_prompt = f"""
[탐지된 {subject_korean} 객체 정보]
- {subject_korean}: {"탐지됨 " + str(subject_info) if is_detected else "탐지 안됨"}

위 정보를 바탕으로 심리 분석을 수행하세요.
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.6,
            max_tokens=1500
        )
        
        markdown_result = response.choices[0].message.content
        parsed_data = parse_markdown_analysis(markdown_result)
        return parsed_data

    except Exception as e:
        print(f"❌ Groq API Error: {e}")
        return {
            "markdown": "분석 실패",
            "overall_summary": "오류 발생",
            "positive_traits": [],
            "negative_traits": [],
            "neutral_observations": [],
            "recommendations": []
        }


# 8. [New] 단일 분석 파이프라인
def single_analysis_pipeline(image_path: str, subject: str, confidence_threshold: float = CONFIDENCE_THRESHOLD):
    detection_result = detect_single_object(image_path, subject, confidence_threshold)
    if not detection_result:
        raise ValueError("Invalid subject or model error")
        
    psychology_result = analyze_single_psychology(detection_result["filtered_summary"], subject)
    
    return {
        "detection": detection_result,
        "psychology": psychology_result
    }