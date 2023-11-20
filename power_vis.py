import matplotlib.pyplot as plt
import pandas as pd

# 데이터프레임 df에 'name', 'time', 'pve' 컬럼이 있다고 가정하고 코드를 작성합니다.
df = pd.read_csv('asos_power.csv')
df['pve'] = df['pve'].fillna(-1)
df['Date'] = pd.to_datetime(df['Date'])
# 'name' 별로 데이터를 그룹화
grouped = df.groupby('Location')

# 그룹의 개수에 따라 subplot의 행과 열 결정
n_groups = len(grouped)
n_rows = n_groups // 2 + n_groups % 2
n_cols = 2 if n_groups > 1 else 1

# 각 그룹('name')에 대해 별도의 그래프 그리기
fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6 * n_rows))
axes = axes.flatten()  # 축 리스트를 1차원 배열로 변환

for i, (name, group) in enumerate(grouped):
    ax = axes[i]
    ax.plot(group['Date'], group['pve'], label=name)
    ax.set_title(f"'pve' Distribution for {name}")
    ax.set_xlabel('Time')
    ax.set_ylabel('PVE')
    ax.legend()

# 빈 subplot을 숨김 (짝수 개의 그래프가 아닐 경우)
for j in range(i+1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()

