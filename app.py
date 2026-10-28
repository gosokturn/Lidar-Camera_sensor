import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="LiDAR-Camera 투영 시뮬레이션", layout="wide")
st.title("라이다 포인트 클라우드 → 카메라 이미지 투영 시뮬레이션")
st.markdown("""
**탐구 주제**: 라이다의 3차원 점과 카메라의 2차원 이미지를 수학적으로 연결하고,  
센서 간 오차가 투영 정확도에 미치는 영향을 분석합니다.
""")

# ====================== 사이드바 파라미터 ======================
st.sidebar.header("시뮬레이션 설정")

# 카메라 내부 파라미터 (고정)
f = st.sidebar.slider("초점거리 f (픽셀)", 400, 1200, 800, 50)
cx, cy = 320, 240  # 이미지 중심 (640x480 가정)

# 오차 파라미터
theta_deg = st.sidebar.slider("상대 각도 오차 θ (°)", -5.0, 5.0, 0.0, 0.1)
d_cm = st.sidebar.slider("상대 위치 오차 d (cm)", -20.0, 20.0, 0.0, 0.5)

# 대상 물체 거리
Z = st.sidebar.slider("물체 거리 Z (m)", 3.0, 30.0, 10.0, 0.5)

# 포인트 클라우드 생성 옵션
num_points = st.sidebar.slider("포인트 개수", 5, 50, 15)

# ====================== 수학 함수 ======================
def rotation_matrix_y(theta_rad):
    """Y축 회전 행렬 (각도 오차 시뮬레이션)"""
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])

def project_points(points_3d, f, cx, cy):
    """원근 투영: 3D → 2D 픽셀"""
    # points_3d: (N, 3) 배열, 카메라 좌표계
    X, Y, Z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
    # Z가 0에 가까우면 방지
    Z = np.maximum(Z, 0.1)
    u = f * (X / Z) + cx
    v = f * (Y / Z) + cy
    return np.column_stack([u, v])

# ====================== 데이터 생성 ======================
# 간단한 포인트 클라우드 생성 (가로로 퍼진 점들, 예: 차량 앞면)
np.random.seed(42)
base_points = np.zeros((num_points, 3))
base_points[:, 0] = np.linspace(-2.0, 2.0, num_points)  # X: 좌우
base_points[:, 1] = np.random.uniform(-0.5, 1.5, num_points)  # Y: 높이
base_points[:, 2] = Z  # Z: 거리 (고정)

# 이상적인 경우 (오차 0)
ideal_points = base_points.copy()
ideal_proj = project_points(ideal_points, f, cx, cy)

# 오차가 있는 경우
theta_rad = np.deg2rad(theta_deg)
R = rotation_matrix_y(theta_rad)
t = np.array([d_cm / 100.0, 0.0, 0.0])  # cm → m, X방향 이동

error_points = (R @ base_points.T).T + t
error_proj = project_points(error_points, f, cx, cy)

# 투영 오차 계산 (픽셀 거리 평균)
pixel_errors = np.linalg.norm(ideal_proj - error_proj, axis=1)
mean_error = np.mean(pixel_errors)
max_error = np.max(pixel_errors)

# ====================== 결과 표시 ======================
col1, col2 = st.columns(2)

with col1:
    st.subheader("투영 결과 시각화")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 이상적 투영 (파란색)
    ax.scatter(ideal_proj[:, 0], ideal_proj[:, 1], c='blue', s=40, label='이상적 투영 (오차 0)', alpha=0.7)
    # 오차 포함 투영 (빨간색)
    ax.scatter(error_proj[:, 0], error_proj[:, 1], c='red', s=40, label=f'오차 적용 (θ={theta_deg}°, d={d_cm}cm)', alpha=0.7)
    
    # 연결선으로 오차 표시
    for i in range(len(ideal_proj)):
        ax.plot([ideal_proj[i,0], error_proj[i,0]], 
                [ideal_proj[i,1], error_proj[i,1]], 
                'gray', alpha=0.3, linewidth=0.8)
    
    ax.set_xlim(0, 640)
    ax.set_ylim(480, 0)  # 이미지 좌표계 (y축 아래가 +)
    ax.set_xlabel("u (픽셀)")
    ax.set_ylabel("v (픽셀)")
    ax.set_title("카메라 이미지 평면 위의 투영점")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    st.pyplot(fig)

with col2:
    st.subheader("오차 분석")
    st.metric("평균 투영 오차", f"{mean_error:.2f} 픽셀")
    st.metric("최대 투영 오차", f"{max_error:.2f} 픽셀")
    
    st.markdown(f"""
    **현재 설정**
    - 각도 오차 θ: **{theta_deg}°**
    - 위치 오차 d: **{d_cm} cm**
    - 물체 거리 Z: **{Z} m**
    - 초점거리 f: **{f} px**
    """)
    
    # 오차 경향 설명
    if abs(theta_deg) < 0.5 and abs(d_cm) < 2:
        st.success("오차가 매우 작습니다. 라이다의 정확한 거리 정보와 카메라의 상세 정보가 잘 결합됩니다.")
    elif abs(theta_deg) > 2:
        st.warning("각도 오차가 큽니다. 투영이 크게 틀어져 센서 융합 효과가 떨어집니다.")
    else:
        st.info("약간의 오차가 있습니다. 센서 정렬의 중요성을 확인할 수 있습니다.")

# ====================== 수식 설명 ======================
with st.expander("사용된 수학 공식 보기"):
    st.latex(r"u = f \cdot \frac{X_c}{Z_c} + c_x, \quad v = f \cdot \frac{Y_c}{Z_c} + c_y")
    st.markdown("""
    - \( (X_c, Y_c, Z_c) \): 카메라 좌표계에서의 3D 점
    - 각도 오차 θ는 Y축 회전 행렬로 적용
    - 위치 오차 d는 평행이동 벡터로 적용
    """)

# ====================== 데이터 테이블 ======================
st.subheader("포인트별 상세 데이터 (일부)")
table_data = {
    "X (m)": base_points[:8, 0].round(2),
    "이상적 u": ideal_proj[:8, 0].round(1),
    "오차 적용 u": error_proj[:8, 0].round(1),
    "픽셀 오차": pixel_errors[:8].round(2)
}
st.dataframe(table_data)

st.caption("※ 이 시뮬레이션은 단순화된 모델입니다. 실제로는 회전 행렬의 모든 축과 카메라 캘리브레이션이 더 복잡하게 적용됩니다.")
