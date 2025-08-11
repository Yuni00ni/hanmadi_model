#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
뉴스 요약 데이터를 데이터베이스에 저장하는 스크립트
"""

import sqlite3
import pandas as pd
import os
from datetime import datetime

def create_news_database(csv_file_path, db_file_path="news_summaries.db"):
    """
    CSV 파일을 읽어서 SQLite 데이터베이스에 저장
    
    Args:
        csv_file_path (str): CSV 파일 경로
        db_file_path (str): 생성할 데이터베이스 파일 경로
    """
    
    # CSV 파일 읽기
    try:
        df = pd.read_csv(csv_file_path)
        print(f"CSV 파일 로드 완료: {len(df)} 행")
        print(f"컬럼: {list(df.columns)}")
    except Exception as e:
        print(f"CSV 파일 읽기 오류: {e}")
        return
    
    # 컬럼명 매핑 (한글 -> 영어)
    column_mapping = {
        '원본_제목': 'original_title',
        '요약': 'summary'
    }
    
    # 컬럼명 변경
    df = df.rename(columns=column_mapping)
    print(f"컬럼명 변경 완료: {list(df.columns)}")
    
    # 데이터베이스 연결
    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()
    
    # 테이블 생성 (기존 테이블이 있으면 삭제)
    cursor.execute("DROP TABLE IF EXISTS news_summaries")
    
    create_table_sql = """
    CREATE TABLE news_summaries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        original_title TEXT NOT NULL,
        summary TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """
    
    cursor.execute(create_table_sql)
    print("테이블 'news_summaries' 생성 완료")
    
    # 데이터 삽입
    try:
        df.to_sql('news_summaries', conn, if_exists='append', index=False, method='multi')
        print(f"데이터 삽입 완료: {len(df)} 행")
    except Exception as e:
        print(f"데이터 삽입 오류: {e}")
        conn.rollback()
        return
    
    # 인덱스 생성 (검색 성능 향상)
    cursor.execute("CREATE INDEX idx_original_title ON news_summaries(original_title)")
    cursor.execute("CREATE INDEX idx_created_at ON news_summaries(created_at)")
    print("인덱스 생성 완료")
    
    # 데이터 확인
    cursor.execute("SELECT COUNT(*) FROM news_summaries")
    count = cursor.fetchone()[0]
    print(f"총 저장된 레코드 수: {count}")
    
    # 샘플 데이터 출력
    cursor.execute("SELECT id, original_title, summary FROM news_summaries LIMIT 3")
    sample_data = cursor.fetchall()
    print("\n샘플 데이터:")
    for row in sample_data:
        print(f"ID: {row[0]}")
        print(f"제목: {row[1][:50]}...")
        print(f"요약: {row[2][:50]}...")
        print("-" * 50)
    
    # 연결 종료
    conn.commit()
    conn.close()
    print(f"\n데이터베이스 파일 생성 완료: {db_file_path}")

def query_database(db_file_path="news_summaries.db", search_term=None):
    """
    데이터베이스 조회 함수
    
    Args:
        db_file_path (str): 데이터베이스 파일 경로
        search_term (str): 검색할 키워드 (선택사항)
    """
    
    if not os.path.exists(db_file_path):
        print(f"데이터베이스 파일이 존재하지 않습니다: {db_file_path}")
        return
    
    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()
    
    if search_term:
        # 제목이나 요약에서 키워드 검색
        query = """
        SELECT id, original_title, summary, created_at 
        FROM news_summaries 
        WHERE original_title LIKE ? OR summary LIKE ?
        ORDER BY created_at DESC
        """
        cursor.execute(query, (f'%{search_term}%', f'%{search_term}%'))
    else:
        # 전체 데이터 조회 (최신 10개)
        query = """
        SELECT id, original_title, summary, created_at 
        FROM news_summaries 
        ORDER BY created_at DESC 
        LIMIT 10
        """
        cursor.execute(query)
    
    results = cursor.fetchall()
    
    if results:
        print(f"\n검색 결과: {len(results)}개")
        print("=" * 80)
        for row in results:
            print(f"ID: {row[0]}")
            print(f"제목: {row[1]}")
            print(f"요약: {row[2]}")
            print(f"생성일: {row[3]}")
            print("-" * 80)
    else:
        print("검색 결과가 없습니다.")
    
    conn.close()

if __name__ == "__main__":
    # CSV 파일 경로
    csv_file = "news_summaries.csv"
    
    # 데이터베이스 생성
    if os.path.exists(csv_file):
        create_news_database(csv_file)
        
        # 간단한 조회 테스트
        print("\n" + "="*50)
        print("데이터베이스 조회 테스트")
        print("="*50)
        query_database()
        
    else:
        print(f"CSV 파일을 찾을 수 없습니다: {csv_file}")
        print("다음 중 하나의 파일을 사용하세요:")
        for file in os.listdir("."):
            if file.endswith(".csv"):
                print(f"  - {file}")
