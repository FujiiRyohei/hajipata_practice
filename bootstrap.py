import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

# 母集団を生成
mu=1; sigma=0.4; N=1000000
population = np.random.lognormal(mu, sigma, N)
population_avg = np.average(population)
print('Population:\tavg={0}'.format(population_avg))

# 標本を抽出
N_sample = 100
sample = population[0:N_sample]

# ブートストラップサンプルを抽出
bs_avgs = []
# ブートストラップサンプルの個数
bs_trial = 10000
for i in range(0, bs_trial):
    # 重複を許してN_sample個の標本からN_sample個抽出
    bs_sample = np.random.choice(sample, N_sample)
    # 各ブートストラップサンプル内の平均を求める
    bs_avgs.append(np.average(bs_sample))
print(
        'Bootstrap:\tavg={0}, avg_deviation={1}'.
        format(np.average(bs_avgs), np.std(bs_avgs)))

# 母集団のヒストグラム
fig, (ax_pop,ax_bs) = plt.subplots(nrows=2)
plt.subplots_adjust(hspace=0.4)
ax_pop.set_title("Histgram of population")
y,x,_ = ax_pop.hist(population, bins=100, range=(0,10))
ax_pop.axvline(population_avg, color="r")
ax_pop.text(
        population_avg*1.02, y.max()*0.95,
        "Average= {0}".format(round(population_avg,2)))
# 各ブートストラップサンプル内におけるの平均のヒストグラム
ax_bs.set_title("Histgram of average of Bootstrap sample")
y,x,_ = ax_bs.hist(bs_avgs, bins=50)
ax_bs.axvline(np.average(bs_avgs), color="r")
ax_bs.text(
        np.average(bs_avgs)*1.02, y.max()*0.95,
        "mu = {0}, sigma= {1}"
        .format(round(np.average(bs_avgs),2), round(np.std(bs_avgs),2)))
plt.show()

