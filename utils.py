import sys
import os
import datetime

class Logger(object):
    def __init__(self, filename):
        # 기존 stdout 사용
        self.terminal = sys.stdout
        # 로그 파일 (추가 모드)
        self.log = open(filename,"a", encoding='utf-8')
        
    def write(self, message):
        # 빈 줄이 아닌 경우에만 타임스탬프 추가
        if message.strip():
            timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")
            # 현재 터미널(stdout)에 쓰기
            self.terminal.write(f"{timestamp}{message}")
            # 로그 파일에 쓰기
            self.log.write(f"{timestamp}{message}")
        else:
            # 개행이 들어가는 거 처럼, 빈 줄은 그대로 출력  
            self.terminal.write(f"{message}")
            self.log.write(f"{message}")

    def flush(self):
        """
        실시간 출력 보장: 출력 버퍼 강제 비움 함수
        """
        self.terminal.flush()
        self.log.flush()

