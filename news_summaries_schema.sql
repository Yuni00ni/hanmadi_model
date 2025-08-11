-- 뉴스 요약 데이터베이스 스키마
-- 파일명: news_summaries_schema.sql

-- 뉴스 요약 테이블 생성
CREATE TABLE IF NOT EXISTS news_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,          -- 고유 ID (자동 증가)
    original_title TEXT NOT NULL,                  -- 원본 뉴스 제목 (원본_제목)
    summary TEXT NOT NULL,                         -- 요약된 내용 (요약)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, -- 생성 시간
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP  -- 수정 시간
);

-- 인덱스 생성 (검색 성능 향상)
CREATE INDEX IF NOT EXISTS idx_original_title ON news_summaries(original_title);
CREATE INDEX IF NOT EXISTS idx_created_at ON news_summaries(created_at);
CREATE INDEX IF NOT EXISTS idx_summary ON news_summaries(summary);

-- 전문 검색을 위한 FTS (Full-Text Search) 테이블 생성 (선택사항)
CREATE VIRTUAL TABLE IF NOT EXISTS news_summaries_fts USING fts5(
    original_title,
    summary,
    content='news_summaries',
    content_rowid='id'
);

-- FTS 테이블에 기존 데이터 삽입을 위한 트리거
CREATE TRIGGER IF NOT EXISTS news_summaries_ai AFTER INSERT ON news_summaries BEGIN
    INSERT INTO news_summaries_fts(rowid, original_title, summary) 
    VALUES (new.id, new.original_title, new.summary);
END;

CREATE TRIGGER IF NOT EXISTS news_summaries_ad AFTER DELETE ON news_summaries BEGIN
    INSERT INTO news_summaries_fts(news_summaries_fts, rowid, original_title, summary) 
    VALUES('delete', old.id, old.original_title, old.summary);
END;

CREATE TRIGGER IF NOT EXISTS news_summaries_au AFTER UPDATE ON news_summaries BEGIN
    INSERT INTO news_summaries_fts(news_summaries_fts, rowid, original_title, summary) 
    VALUES('delete', old.id, old.original_title, old.summary);
    INSERT INTO news_summaries_fts(rowid, original_title, summary) 
    VALUES (new.id, new.original_title, new.summary);
END;

-- 유용한 쿼리 예시

-- 1. 전체 데이터 조회 (최신 순)
-- SELECT * FROM news_summaries ORDER BY created_at DESC;

-- 2. 특정 키워드로 제목 검색
-- SELECT * FROM news_summaries WHERE original_title LIKE '%대통령%';

-- 3. 요약 내용에서 키워드 검색
-- SELECT * FROM news_summaries WHERE summary LIKE '%경제%';

-- 4. 제목과 요약 모두에서 키워드 검색
-- SELECT * FROM news_summaries 
-- WHERE original_title LIKE '%정치%' OR summary LIKE '%정치%';

-- 5. FTS를 이용한 전문 검색 (한글 지원이 제한적일 수 있음)
-- SELECT news_summaries.* FROM news_summaries_fts 
-- JOIN news_summaries ON news_summaries_fts.rowid = news_summaries.id 
-- WHERE news_summaries_fts MATCH '대통령';

-- 6. 데이터 통계 조회
-- SELECT 
--     COUNT(*) as total_records,
--     AVG(LENGTH(original_title)) as avg_title_length,
--     AVG(LENGTH(summary)) as avg_summary_length
-- FROM news_summaries;

-- 7. 날짜별 데이터 수 조회
-- SELECT 
--     DATE(created_at) as date,
--     COUNT(*) as count
-- FROM news_summaries 
-- GROUP BY DATE(created_at) 
-- ORDER BY date DESC;
