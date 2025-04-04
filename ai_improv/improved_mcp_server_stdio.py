# improved_mcp_server_stdio.py
import logging
from mcp.server.fastmcp import FastMCP

# 로깅 설정
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MCP 서버 생성
mcp = FastMCP("간단한 데모")


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


# 메인 실행 부분
if __name__ == "__main__":
    # 중요: stdio 전송 방식으로 실행
    logger.info("MCP 서버를 stdio 모드로 시작합니다.")
    try:
        mcp.run(transport='stdio')
    except Exception as e:
        logger.error(f"서버 실행 중 오류 발생: {e}", exc_info=True)