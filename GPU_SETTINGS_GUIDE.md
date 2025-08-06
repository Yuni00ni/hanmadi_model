# VS Code GPU 우선순위 전역 설정 가이드

## 🎯 목적

Windows GPU → Mac GPU → CPU 순서로 자동 선택하도록 VS Code를 설정합니다.

## 📁 생성된 설정 파일들

1. **`.vscode/settings.json`** - 작업공간 설정
2. **`.vscode/extensions.json`** - 추천 확장
3. **`.vscode/launch.json`** - 디버그 설정
4. **`hanmadi_model.code-workspace`** - 작업공간 파일

## 🔧 전역 설정 방법 (수동)

### Windows 사용자

1. **Ctrl + Shift + P** → `Preferences: Open User Settings (JSON)` 실행
2. 아래 설정을 추가:

```json
{
    "python.terminal.activateEnvironment": true,
    "jupyter.runStartupCommands": [
        "import os, torch",
        "if torch.cuda.is_available():",
        "    os.environ['CUDA_VISIBLE_DEVICES'] = '0'",
        "    print('🚀 CUDA GPU 활성화')",
        "elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():",
        "    print('🚀 MPS GPU 활성화')", 
        "else:",
        "    print('🖥️ CPU 사용')"
    ],
    "terminal.integrated.env.windows": {
        "CUDA_VISIBLE_DEVICES": "0",
        "CUDA_LAUNCH_BLOCKING": "1",
        "OMP_NUM_THREADS": "8"
    }
}
```

### Mac 사용자

```json
{
    "terminal.integrated.env.osx": {
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        "OMP_NUM_THREADS": "8"
    }
}
```

## 🚀 자동 적용 방법

1. VS Code 재시작
2. 새 노트북 생성시 자동으로 GPU 우선순위 적용
3. 터미널에서도 환경변수 자동 설정

## ✅ 확인 방법

노트북에서 다음 코드 실행:

```python
import torch
print(f"사용 중인 디바이스: {torch.cuda.is_available()}")
```

## 📋 주요 기능

- **자동 GPU 감지**: CUDA → MPS → CPU 순서
- **환경변수 설정**: 플랫폼별 최적화
- **메모리 관리**: GPU 메모리 효율적 사용
- **확장 추천**: 필요한 확장 자동 제안

## 🔄 업데이트 방법

설정 파일을 수정한 후 VS Code 재시작하면 자동 적용됩니다.
