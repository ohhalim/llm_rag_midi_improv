# improved_mcp_server.py
import logging
from mcp.server.fastmcp import FastMCP

# 로깅 설정
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MCP 서버 생성 - 명시적 포트 지정 및 호스트 설정
mcp = FastMCP("간단한 데모", port=8000, host="0.0.0.0")


# 계산 도구 추가
@mcp.tool()
def calculate(operation: str, a: float, b: float) -> float:
    """두 숫자에 대한 기본 연산 수행
    
    Args:
        operation: 수행할 연산 (add, subtract, multiply, divide)
        a: 첫 번째 숫자
        b: 두 번째 숫자
        
    Returns:
        계산 결과
    """
    logger.info(f"계산 도구 호출: {operation}({a}, {b})")
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        if b == 0:
            raise ValueError("0으로 나눌 수 없습니다")
        return a / b
    else:
        raise ValueError(f"지원되지 않는 연산: {operation}")


# 동적 인사말 리소스 추가
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """개인화된 인사말 생성
    
    Args:
        name: 인사할 사람의 이름
        
    Returns:
        인사말 메시지
    """
    logger.info(f"인사말 리소스 요청: {name}")
    return f"안녕하세요, {name}님! 오늘 좋은 하루 되세요!"


# 날씨 정보를 위한 가상 데이터 리소스
@mcp.resource("weather://{city}")
def get_weather(city: str) -> str:
    """도시별 가상 날씨 정보 제공
    
    Args:
        city: 날씨 정보를 요청할 도시 이름
        
    Returns:
        해당 도시의 가상 날씨 정보
    """
    logger.info(f"날씨 리소스 요청: {city}")
    weather_data = {
        "서울": "맑음, 22°C",
        "부산": "흐림, 24°C",
        "제주": "비, 20°C"
    }
    
    return f"{city}의 현재 날씨: {weather_data.get(city, '정보 없음')}"


# 간단한 프롬프트 추가
@mcp.prompt()
def weather_analysis(city: str, forecast: str) -> str:
    """날씨 분석을 위한 프롬프트 생성
    
    Args:
        city: 도시 이름
        forecast: 날씨 예보
        
    Returns:
        날씨 분석을 요청하는 프롬프트
    """
    logger.info(f"날씨 분석 프롬프트 요청: {city}")
    return f"""다음 {city}의 날씨 정보를 분석해주세요:
    
{forecast}

분석에 포함할 내용:
1. 현재 날씨 상태
2. 온도 범위
3. 여행하기 좋은 날씨인지 여부
4. 필요한 준비물 추천
"""


# 메인 실행 부분
if __name__ == "__main__":
    # MCP 서버 실행
    logger.info("MCP 서버를 시작합니다. 포트: 8000, 호스트: 0.0.0.0")
    
    try:
        mcp.run()
    except Exception as e:
        logger.error(f"서버 실행 중 오류 발생: {e}", exc_info=True)