# system/tts_worker_client.py
"""
SBV2 Worker Client

이 파일의 역할:
- sbv2_worker.py 를 "서브프로세스"로 단 1회 실행
- 모델 로딩이 끝났다는 READY 신호를 기다림
- 이후 speak(text) 호출마다
  → JSON을 stdin으로 보내고
  → wav 결과를 stdout으로 받음

즉:
이 클래스는 "무거운 TTS 엔진을 계속 띄워둔 채"
"가벼운 RPC 클라이언트"처럼 동작한다.
"""

from __future__ import annotations

# 표준 라이브러리
import subprocess   # 외부 프로세스 실행 및 제어
import json         # 프로세스 간 통신용 직렬화
import sys          # 현재 Python 인터프리터 경로
from pathlib import Path
from typing import Optional


class SBV2WorkerClient:
    def __init__(self) -> None:
        """
        1) sbv2_worker.py 경로를 찾고
        2) subprocess.Popen으로 워커를 실행
        3) READY 신호가 나올 때까지 stdout을 읽으며 대기

        이 시점에서:
        - Style-BERT-VITS2 모델
        - BERT
        - Vocoder
        - Style vectors
        가 전부 워커 쪽에서 "한 번만" 로드된다.
        """

        # sbv2_worker.py 위치 계산
        self.worker_path = (
            Path(__file__).resolve().parent.parent
            / "models"
            / "sbv2_core"
            / "sbv2_worker.py"
        )

        if not self.worker_path.exists():
            raise FileNotFoundError(
                f"sbv2_worker.py not found: {self.worker_path}"
            )

        # 서브프로세스 실행
        self.proc = subprocess.Popen(
            # 현재 실행 중인 Python 인터프리터로 워커 실행
            [sys.executable, str(self.worker_path)],

            # 표준 입력/출력 파이프 연결
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,

            # stderr도 stdout으로 합침
            # → 워커 로그를 전부 한 스트림에서 처리
            stderr=subprocess.STDOUT,

            # text=True:
            #   bytes 대신 str로 stdin/stdout 처리
            text=True,

            # bufsize=1:
            #   line-buffered (줄 단위 flush)
            #   → 실시간 통신에 중요
            bufsize=1,
        )

        # READY 신호 대기
        assert self.proc.stdout is not None

        ready = False

        while True:
            # 워커 stdout 한 줄 읽기
            line = self.proc.stdout.readline()

            # EOF → 프로세스가 죽었음
            if not line:
                break

            line = line.strip()

            # 워커 로그를 그대로 출력
            # (디버깅용, 없어도 무방)
            print(f"[SBV2_WORKER] {line}")

            # 워커에서 명시적으로 출력한 준비 완료 신호
            if line == "__SBV2_READY__":
                ready = True
                break

        if not ready:
            raise RuntimeError("SBV2 worker failed to start")

    def speak(
        self,
        text: str,
        style_index: int = 0,
        style_weight: float = 1.0,
    ) -> str:
        """
        TTS 요청을 워커에 보내고
        생성된 wav 파일 경로를 반환한다.

        이 함수는:
        - 모델 로딩 x
        - 가중치 로딩 x
        - BERT 로딩 x

        오직:
        - JSON 직렬화
        - 파이프 통신
        - infer() 호출
        만 수행된다.
        """

        if self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError("SBV2 worker is not running")

        # 요청 payload 구성
        payload = {
            "text": text,
            "style": style_index,
            "style_weight": style_weight,
        }

        # 요청 전송 (stdin → worker)
        self.proc.stdin.write(json.dumps(payload) + "\n")

        # flush가 매우 중요:
        # flush 안 하면 worker가 입력을 못 받는다
        self.proc.stdin.flush()

        # 응답 수신 (stdout ← worker)
        while True:
            line = self.proc.stdout.readline()

            if not line:
                raise RuntimeError("SBV2 worker terminated unexpectedly")

            line = line.strip()

            # JSON이 아닌 것은 로그 → 무시
            if not line.startswith("{"):
                print(f"[SBV2_WORKER] {line}")
                continue

            # JSON 응답 파싱
            resp = json.loads(line)

            # 워커 내부 오류 전달
            if "error" in resp:
                raise RuntimeError(f"TTS error: {resp['error']}")

            # 정상 응답: wav 경로 반환
            return resp["wav_path"]

    def close(self) -> None:
        """
        워커 프로세스 종료
        """
        if self.proc:
            self.proc.terminate()
            self.proc.wait(timeout=5)
