import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
# 母集団を生成
mu=1; sigma=0.4; N=1000000
population = np.random.lognormal(mu, sigma, N)
print('Population:\tmean={0}'.format(np.average(population)))

# 標本を抽出
N_sample = 100
sample = population[0:N_sample]

# ブートストラップサンプルを抽出
bs_mean = [] 
bs_trial = 1000
for i in range(0, bs_trial):
    # 重複を許してN_sample個の標本からN_sample個抽出
    bs_sample = np.random.choice(sample, N_sample)
    # ブートストラップサンプルの平均を求める
    bs_mean.append(np.average(bs_sample))
print(
        'Bootstrap:\tmean={0}, mean_deviation={1}'.
        format(np.average(bs_mean), np.std(bs_mean, ddof=1)))

fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)
# 母集団のヒストグラム
ax1.hist(population, bins=100, range=(0,5))
# ブートストラップサンプルの平均のヒストグラム
ax2.hist(bs_mean, bins=50)
plt.show()

