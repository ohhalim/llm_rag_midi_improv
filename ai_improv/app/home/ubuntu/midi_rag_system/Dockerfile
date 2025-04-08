FROM python:3.11-slim

WORKDIR /app

# 기본 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# 필요한 Python 패키지 복사
COPY requirements.txt ./
COPY setup.py ./
COPY __init__.py ./

# 모든 소스 코드 복사
COPY api/ ./api/
COPY core/ ./core/
COPY data/ ./data/
COPY docs/ ./docs/
COPY models/ ./models/
COPY tests/ ./tests/
COPY utils/ ./utils/
COPY main.py ./
COPY config.py ./
COPY .env ./

# 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt


# 필요한 디렉토리 생성
RUN mkdir -p data/input data/output data/training data/vectorstore

# 시스템 환경 설정
ENV PYTHONPATH="${PYTHONPATH}:/app"
ENV API_HOST=0.0.0.0
ENV API_PORT=8080
ENV MCP_SERVER_HOST=0.0.0.0
ENV MCP_SERVER_PORT=8000

# 기본 실행 명령
ENTRYPOINT ["python", "main.py"]
CMD ["server"]

# 포트 노출
EXPOSE 8080 8000 