import sys
import os
from loguru import logger


import re


class DualStream:
    """
    기존 print()나 tqdm() 등의 콘솔 출력을 가로채서
    터미널에 출력함과 동시에 파일로 기록(Append)하는 클래스입니다.
    """

    def __init__(self, terminal_stream, filename):
        self.terminal_stream = terminal_stream
        self.filename = filename

    def write(self, data):
        # 1. 터미널 출력 (항상)
        self.terminal_stream.write(data)
        self.terminal_stream.flush()

        # 2. 파일 출력: \r만 있는 tqdm 중간 프레임은 건너띶 (로그 오염 방지)
        # tqdm은 \r로 덩어쓰기 방식을 사용하므로, \n이 없는 라인은 파일에 기록하지 않음
        if self.filename and "\n" in data and "\r" not in data:
            with open(self.filename, "a", encoding="utf-8") as f:
                # ANSI 색상/커서 제어 코드 제거
                clean_data = re.sub(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", "", data)
                f.write(clean_data)
                f.flush()

    def flush(self):
        self.terminal_stream.flush()


class Logger:
    """
    Loguru 기반의 통합 로거 클래스
    터미널 출력과 파일 저장을 동시에 깔끔하게 관리합니다.
    """

    def __init__(self, filename=None):
        # 기존에 등록된 기본 핸들러들 제거 (중복 출력 방지)
        logger.remove()

        # 터미널용 핸들러 (순수 __stdout__ 을 타게 함으로써 무한루프 방지)
        logger.add(
            sys.__stdout__,
            colorize=True,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        )

        # 파일용 핸들러 (Loguru 명시적 출력용)
        if filename:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            logger.add(
                filename,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
                mode="a",
                encoding="utf-8",
            )

            # Print, tqdm 등 일반 표준 출력을 가로채기
            sys.stdout = DualStream(sys.__stdout__, filename)
            sys.stderr = DualStream(sys.__stderr__, filename)

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
