# 🧑‍🏫 KoBERT 뉴스 요약 평가 코드 초등학생도 이해할 수 있게 한 줄씩 설명!
# (실제 코드와 설명, 예시를 함께 적어둡니다)

# 1. 필요한 도구(라이브러리) 불러오기
import pandas as pd  # 표처럼 생긴 데이터를 쉽게 다루는 도구예요. (예: 엑셀처럼)
import numpy as np   # 숫자 계산을 쉽게 해주는 도구예요. (예: 평균, 합계 등)
import os            # 컴퓨터 파일을 다루는 도구예요. (예: 폴더, 파일 찾기)
import re            # 글자에서 규칙을 찾아 바꿔주는 도구예요. (예: 느낌표 없애기)
from sklearn.feature_extraction.text import TfidfVectorizer  # 글에서 중요한 단어를 뽑아주는 도구예요.
from sklearn.metrics.pairwise import cosine_similarity       # 두 글이 얼마나 비슷한지 알려주는 도구예요.
from collections import Counter  # 단어가 몇 번 나왔는지 세주는 도구예요.
from datetime import datetime    # 오늘 날짜와 시간을 알려주는 도구예요.
from tqdm import tqdm            # 작업이 얼마나 진행됐는지 보여주는 도구예요.
import warnings                  # 경고 메시지를 안 보이게 해주는 도구예요.

warnings.filterwarnings("ignore")  # 경고 메시지는 안 보이게 해요. (예: "이건 위험해요!" 같은 메시지 숨기기)

print("🚀 KoBERT 뉴스 요약 평가를 시작합니다!")  # 프로그램이 시작됐다고 알려줘요.
print("=" * 60)  # =를 60번 반복해서 줄을 그어요.

# 2. 파일에서 데이터를 읽어오는 함수 만들기
# 예시: 'data.csv'라는 파일이 있으면 표로 읽어와요

def safe_read_csv(file_path):  # safe_read_csv라는 이름의 함수를 만들어요.
    """
    파일을 안전하게 읽어오는 함수예요.
    만약 글자가 깨지면 다른 방법으로도 읽어봐요.
    """
    try:  # 먼저 utf-8-sig 방식으로 읽어봐요.
        return pd.read_csv(file_path, encoding="utf-8-sig")
    except:  # 만약 글자가 깨지면
        for encoding in ["utf-8", "cp949", "euc-kr"]:  # 다른 방식으로도 읽어봐요.
            try:
                return pd.read_csv(file_path, encoding=encoding)
            except:
                continue  # 또 안 되면 다음 방식으로 넘어가요.
        raise ValueError(f"CSV 파일 읽기 실패: {file_path}")  # 다 안 되면 에러를 내요.

# 예시: 실제로 파일이 없으면 아래처럼 실행하면 에러가 나요
# df = safe_read_csv('data.csv')

# 3. 글자를 예쁘게 다듬는 함수
# 예시: "안녕!!   나는  코딩을  좋아해!!" -> "안녕 나는 코딩을 좋아해"
def preprocess_text(text):  # preprocess_text라는 이름의 함수를 만들어요.
    if pd.isna(text) or text == "":  # 만약 글이 없거나 비어 있으면
        return ""  # 빈 문자열을 돌려줘요.
    text = str(text).strip()  # 글을 문자열로 바꾸고, 앞뒤 빈칸을 없애요.
    text = re.sub(r"\s+", " ", text)  # 여러 칸 띄어쓰기를 하나로 바꿔요.
    text = re.sub(r"[^\w\s가-힣]", "", text)  # 한글, 영어, 숫자 빼고 다 없애요.
    return text  # 다듬어진 글을 돌려줘요.

# 4. 데이터 파일에서 원본 기사와 요약문을 읽어와요
print("📁 데이터 파일에서 기사와 요약문을 읽어와요!")  # 지금부터 파일을 읽는다고 알려줘요.
try:
    original_df = safe_read_csv("data/crawling_origin.csv")  # 원본 기사 파일을 읽어요.
    summary_df = safe_read_csv("data/crawling_origin_with_summary.csv")  # 요약문 파일을 읽어요.
    print(f"✅ 데이터 읽기 성공! 원본 {len(original_df)}개, 요약 {len(summary_df)}개")  # 몇 개 읽었는지 알려줘요.
except Exception as e:
    print(f"❌ 데이터 읽기 실패: {e}")  # 파일을 못 읽으면 에러 메시지를 보여줘요.
    raise

# 5. 기사와 요약문을 예쁘게 다듬어서 리스트로 만들어요
originals = []  # 원본 기사들을 담을 빈 상자를 만들어요.
summaries = []  # 요약문들을 담을 빈 상자를 만들어요.
min_len = min(len(original_df), len(summary_df))  # 둘 중에 더 짧은 개수만큼만 비교해요

for i in tqdm(range(min_len), desc="전처리"):  # 0부터 min_len-1까지 반복해요. (진행상황도 보여줘요)
    try:
        orig_text = preprocess_text(original_df.iloc[i][0])  # 원본 기사 한 줄을 예쁘게 다듬어요.
        summ_text = preprocess_text(summary_df.iloc[i][0])   # 요약문 한 줄도 예쁘게 다듬어요.
        if len(orig_text) >= 10 and len(summ_text) >= 5:  # 너무 짧은 건 빼고
            originals.append(orig_text)  # 원본 기사 상자에 넣어요.
            summaries.append(summ_text)  # 요약문 상자에 넣어요.
    except:
        continue  # 에러가 나면 그냥 넘어가요.
print(f"✅ 전처리 완료! {len(originals)}개 데이터 준비")  # 몇 개 준비됐는지 알려줘요.

# 6. TF-IDF로 기사와 요약문이 얼마나 비슷한지 계산해요
print("🔄 TF-IDF로 비슷한 정도 계산 중...")
vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))  # 중요한 단어를 뽑아주는 도구를 만들어요.
all_texts = originals + summaries  # 원본과 요약문을 합쳐서 한 줄로 만들어요.
# 예시: ["나는 밥을 먹었다", "밥을 먹었다 나는"]
tfidf_matrix = vectorizer.fit_transform(all_texts)  # 글을 숫자로 바꿔줘요.
orig_matrix = tfidf_matrix[: len(originals)]  # 원본 기사 부분만 잘라요.
summ_matrix = tfidf_matrix[len(originals) :]  # 요약문 부분만 잘라요.

tfidf_scores = []  # 점수를 담을 빈 상자를 만들어요.
for i in tqdm(range(len(originals)), desc="TF-IDF"):  # 원본 기사 개수만큼 반복해요.
    sim = cosine_similarity(orig_matrix[i], summ_matrix[i])[0][0]  # i번째 기사와 요약문이 얼마나 비슷한지 계산해요.
    tfidf_scores.append(sim)  # 점수를 상자에 넣어요.
tfidf_scores = np.array(tfidf_scores)  # 리스트를 숫자 배열로 바꿔요.
print(f"TF-IDF 평균 점수: {np.mean(tfidf_scores):.4f}")  # 평균 점수를 보여줘요.

# 7. Jaccard 유사도: 기사와 요약문에 같은 단어가 얼마나 있는지 계산해요
print("🔄 Jaccard 유사도 계산 중...")
jaccard_scores = []  # 점수를 담을 빈 상자를 만들어요.
for orig, summ in tqdm(zip(originals, summaries), total=len(originals), desc="Jaccard"):  # 원본과 요약문을 한 쌍씩 꺼내서
    orig_words = set(orig.split())  # 원본 기사를 단어별로 나눠서 중복 없이 모아요.
    summ_words = set(summ.split())  # 요약문도 단어별로 나눠서 중복 없이 모아요.
    if len(orig_words) == 0 and len(summ_words) == 0:  # 둘 다 단어가 없으면
        jaccard_scores.append(1.0)  # 완전히 같다고 해요.
    elif len(orig_words) == 0 or len(summ_words) == 0:  # 한쪽만 없으면
        jaccard_scores.append(0.0)  # 완전히 다르다고 해요.
    else:
        intersection = len(orig_words & summ_words)  # 겹치는 단어 개수
        union = len(orig_words | summ_words)         # 전체 단어 개수
        jaccard_scores.append(intersection / union)  # 겹치는 비율을 점수로 넣어요.
jaccard_scores = np.array(jaccard_scores)  # 리스트를 숫자 배열로 바꿔요.
print(f"Jaccard 평균 점수: {np.mean(jaccard_scores):.4f}")  # 평균 점수를 보여줘요.

# 8. 키워드 유사도: 기사와 요약문에서 자주 나온 단어가 얼마나 겹치는지 계산해요
print("🔄 키워드 유사도 계산 중...")
keyword_scores = []  # 점수를 담을 빈 상자를 만들어요.
for orig, summ in tqdm(zip(originals, summaries), total=len(originals), desc="키워드"):  # 원본과 요약문을 한 쌍씩 꺼내서
    orig_words = Counter(orig.split())  # 원본 기사에서 단어가 몇 번 나왔는지 세요.
    summ_words = Counter(summ.split())  # 요약문도 단어가 몇 번 나왔는지 세요.
    orig_top = set([word for word, _ in orig_words.most_common(10)])  # 원본에서 많이 나온 10개 단어
    summ_top = set([word for word, _ in summ_words.most_common(10)])  # 요약문에서 많이 나온 10개 단어
    if len(orig_top) == 0 and len(summ_top) == 0:  # 둘 다 단어가 없으면
        keyword_scores.append(1.0)  # 완전히 같다고 해요.
    elif len(orig_top) == 0 or len(summ_top) == 0:  # 한쪽만 없으면
        keyword_scores.append(0.0)  # 완전히 다르다고 해요.
    else:
        intersection = len(orig_top & summ_top)  # 겹치는 단어 개수
        union = len(orig_top | summ_top)         # 전체 단어 개수
        keyword_scores.append(intersection / union)  # 겹치는 비율을 점수로 넣어요.
keyword_scores = np.array(keyword_scores)  # 리스트를 숫자 배열로 바꿔요.
print(f"키워드 평균 점수: {np.mean(keyword_scores):.4f}")  # 평균 점수를 보여줘요.

# 9. 결과를 표로 만들어 저장해요
import datetime  # 날짜와 시간을 다루는 도구를 한 번 더 불러와요.
results_df = pd.DataFrame({  # 표를 만들어요.
    'original_text': originals,  # 원본 기사
    'summary_text': summaries,   # 요약문
    'tfidf_score': tfidf_scores,  # TF-IDF 점수
    'jaccard_score': jaccard_scores,  # Jaccard 점수
    'keyword_score': keyword_scores   # 키워드 점수
})
filename = f"kobert_eval_easy_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"  # 파일 이름을 날짜와 함께 만들어요.
results_df.to_csv(filename, index=False, encoding='utf-8-sig')  # 표를 파일로 저장해요.
print(f"💾 결과 저장 완료! 파일명: {filename}")  # 저장이 끝났다고 알려줘요.

# 10. 예시로 결과 3개만 보여주기
print("\n예시 결과 3개!")  # 이제 예시를 보여줄 거예요.
for i in range(3):  # 0, 1, 2 세 번 반복해요.
    print(f"\n원본: {originals[i][:50]}...")  # 원본 기사 앞 50글자만 보여줘요.
    print(f"요약: {summaries[i][:50]}...")    # 요약문 앞 50글자만 보여줘요.
    print(f"TF-IDF: {tfidf_scores[i]:.2f}, Jaccard: {jaccard_scores[i]:.2f}, 키워드: {keyword_scores[i]:.2f}")  # 점수도 같이 보여줘요.
