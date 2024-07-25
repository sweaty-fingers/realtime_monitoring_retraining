# 전체 과정 실행
```bash
python run_expriment.py

# 이전 로그 삭제
python run_expriment.py --delete_logs

# 실시간 비디오 실행
python run_expriment.py --video
```


# 환경
- Windows OS
- python version 3.12.4

make virtualenv 맥이나 리눅스라면 pyenv 추천

```bash
python -m venv venv
or 
py -3 -m venv venv
```

```bash
# 가상환경 실행
venv\Scripts\activate

# 필요 라이브러리 설치
pip install -r requirements.txt
```

# 개별 파일 실행 예시

Device에서 프로그램(segmentation)이 실행되고 있으며 일정 시간간격으로 서버에 데이터를 전달하는 상황 시뮬레이션 코드
```bash
# 실제 카메라 영상이 윈도우로 뜸
python runnning_code/device_simulation.py 

# 백그라운드에서만 실행
python running_code/just_make_realtime_image.py 
```

디바이스로부터 서버에 들어온 데이터 처리
```bash
python running_code/make_log.py
```

대시보드 실행
```bash
streamlit run dashboard/app.py
```

requirements.txt 생성
```bash
pip install pipreqs

pipreqs {directory}
# or for window
pipreqs --encoding=utf8 {directory}
```

# 재학습 데이터셋
현재 노트북 환경에서 영상부터 모든 것을 실행하기에 무리가 있음으로 custom 데이터셋과 voc 데이터셋의 수를 5개씩으로 제한
이를 없애려면 subset_size 파라미터 변경
