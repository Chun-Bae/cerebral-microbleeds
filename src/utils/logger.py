import sys
import os
from loguru import logger


class Logger:
    """
    Loguru 기반의 통합 로거 클래스
    터미널 출력과 파일 저장을 동시에 깔끔하게 관리합니다.
    """

    def __init__(self, filename=None):
        # 기존에 등록된 기본 핸들러들 제거 (중복 출력 방지)
        logger.remove()

        # 터미널용 핸들러
        logger.add(
            sys.stdout,
            colorize=True,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        )

        # 파일용 핸들러
        if filename:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            logger.add(
                filename,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
                mode="a",
                encoding="utf-8",
            )

        self._logger = logger

    def info(self, message):
        self._logger.info(message)

    def success(self, message):
        self._logger.success(message)

    def warning(self, message):
        self._logger.warning(message)

    def error(self, message):
        self._logger.error(message)


# 파이프라인에서 기본적으로 사용할 터미널 전용 로거 인스턴스
log = Logger()
