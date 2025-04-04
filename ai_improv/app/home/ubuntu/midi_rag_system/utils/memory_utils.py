"""
메모리 사용량 최적화를 위한 유틸리티 함수
"""
import gc
import torch

def optimize_memory():
    """메모리 최적화 함수"""
    # 가비지 컬렉션 강제 실행
    gc.collect()
    
    # CUDA 캐시 정리 (GPU 사용 시)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def get_memory_info():
    """메모리 사용량 정보 반환"""
    import psutil
    
    # 시스템 메모리 정보
    memory = psutil.virtual_memory()
    system_info = {
        "total": f"{memory.total / (1024**3):.2f} GB",
        "available": f"{memory.available / (1024**3):.2f} GB",
        "used": f"{memory.used / (1024**3):.2f} GB",
        "percent": f"{memory.percent}%"
    }
    
    # GPU 메모리 정보 (가능한 경우)
    gpu_info = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_info[f"gpu_{i}"] = {
                "total": f"{torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB",
                "reserved": f"{torch.cuda.memory_reserved(i) / (1024**3):.2f} GB",
                "allocated": f"{torch.cuda.memory_allocated(i) / (1024**3):.2f} GB"
            }
    
    return {"system": system_info, "gpu": gpu_info}
