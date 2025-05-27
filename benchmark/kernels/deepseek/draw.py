import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 从文件读取数据
# 假设文件格式为：id, m, n, k, num_groups, time_us
df = pd.read_csv('kernel_data.txt', 
                 names=['id', 'm', 'n', 'k', 'num_groups', 'time_us'],
                 delim_whitespace=True)  # 使用空白字符作为分隔符

# 固定m和num_groups，计算两个kernel时间和
plt.figure(figsize=(10, 6))
for ng in df['num_groups'].unique():
    for n in df['n'].unique():
        for k in df['k'].unique():
            time_sums = []
            m_values = []
            for m in sorted(df['m'].unique()):
                kernel1 = df[(df['m'] == m) & (df['n'] == n) & (df['k'] == k) & (df['num_groups'] == ng)]
                kernel2 = df[(df['m'] == m) & (df['n'] == k) & (df['k'] == n//2) & (df['num_groups'] == ng)]
                if not kernel1.empty and not kernel2.empty:
                    time_sum = (kernel1['time_us'].iloc[0] + kernel2['time_us'].iloc[0])/1000
                    time_sums.append(time_sum)
                    m_values.append(m)
            
            if time_sums:
                plt.plot(m_values, time_sums, 'o-', label=f'num_groups={int(ng)}, n={int(n)}, k={int(k)}')

plt.xlabel('m')
plt.ylabel('Total Time (ms)')
plt.title('Sum of kernel times for different m and num_groups')
plt.legend()
plt.grid(True)
plt.show()