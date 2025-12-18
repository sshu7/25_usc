# 프로그램 안내
- 본 프로그램은 KISTI에서 UNIST를 위해 작성된 가위,바위,보 AI 이미지 분석 및 게임 프로그램입니다.
- YOLO8 이미지 detection으로 손을 찾은 뒤, 손가락 형태를 보고 가위바위보 이미지 판별을 진행합니다.
- 게임은 jetson에 직접 연결된 모니터/키보드를 통해 실행됩니다.
- 게임이 시작되면, 화면에 손 박스가 표출되며, 현재 가위바위보의 판별 결과가 나옵니다.
- 화면 상단에 현재의 손 결과가 나오며, 스페이스바를 치게 되면 컴퓨터에서 나온 랜덤 결과를 통해 승패의 결과가 나오게 됩니다.
- 게임은 키보드 q를 누르면 끝나게 되어 있습니다. 

---
# 프로그램 실행 방법

1. jetson에 전체 파일을 업로드

2. USB 카메라 연결 확인
$ls /dev/video*
출력 결과로 video0 출력 되어야 함.

3. 폴더 내에서 아래 명령어 실행
$chmod +x install_pkg.sh
$./install_pkg.sh
$source .venv/bin/activate
$chmod +x run_game.sh
$./run_game.sh

4. 3번과정은 한번만 진행하고 다음 게임진행시에는 ./run_game.sh 로 계속 진행하면 됩니다.


#####
중학생 리더십교육
디렉터리

3. 폴더 내에서 아래 명령어 실행
$chmod +x install_pkg.sh
$./install_pkg.sh
$source .venv/bin/activate
$chmod +x run_game.sh

$chmod +x run_basic.sh run_stats.sh run_reaction.sh
$source .venv/bin/activate


KISTI_RPS_YOLO/
│ run_basic.sh
│ run_stats.sh
│ run_reaction.sh
│ run_game.py
│ run_game_stats.py
│ run_game_reaction.py
│
└── utils/
     ├── pipeline.py
     ├── camera.py
     ├── rps_logic.py
     ├── visualization.py
     └── ...

실행흐름
cd KISTI_RPS_YOLO
source .venv/bin/activate

선택실행
./run_basic.sh       #기본 모드
./run_stats.sh        #전적(승/무/패) 표시 모드
./run_reaction.sh   #반응속도 게임 모드 r키 - 새로운 목표 제스처 생성 - 손모양 맞추면 - 반응속도 출력


