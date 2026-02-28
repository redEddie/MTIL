# 1. causal_conv1d 패키지 경로 설정
TARGET_DIR="/Data/pilab/anaconda3/envs/mtil/lib/python3.9/site-packages/causal_conv1d"

# 2. '타입 | None' -> 'Optional[타입]' 치환
find $TARGET_DIR -type f -name "*.py" -exec sed -i \
    -e 's/\([a-zA-Z0-9._]*\) | None/Optional[\1]/g' \
    -e 's/Union\[\([^]]*\), None\]/Optional[\1]/g' {} +

# 3. 파일 상단에 'from typing import Optional' 추가 (중복 방지 포함)
find $TARGET_DIR -type f -name "*.py" -exec grep -l "Optional\[" {} + | xargs -I {} bash -c '
    if ! grep -q "from typing import.*Optional" {}; then
        sed -i "1i from typing import Optional" {}
    fi
'

echo "✅ causal_conv1d 패키지 수정이 완료되었습니다."