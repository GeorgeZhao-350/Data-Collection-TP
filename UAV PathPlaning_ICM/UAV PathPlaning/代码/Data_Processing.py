import matplotlib
import matplotlib.pyplot as plt
import pandas
import seaborn

TYPE='Perf_Data'

plt.rcParams['font.size'] = 10  # 设置全局字体大小
plt.rcParams['font.family'] = 'serif'  # 设置全局字体族
plt.rcParams['font.serif'] = ['Arial']  # 设置serif字体

if TYPE=='Reward':
    plt.ylabel('Reward', fontsize=14, fontname='Arial')
    plt.xlabel('Episodes', fontsize=14, fontname='Arial')
    plt.xlim(0,500)
    # plt.ylim(-600,300)
    datatable=pandas.read_csv("C:\DRQN_Data\DRQN_DQN_Reward_Average.csv")
    seaborn.lineplot(data=datatable, x="Episodes", y="Reward", hue="Algorithm")
    plt.legend(loc='lower right')
    plt.show()
elif TYPE=="Perf_Data":
    plt.xlabel("Dynamic Obstacles", fontsize=14, fontname='Arial')
    plt.ylabel("Success Rate", fontsize=14, fontname='Arial')
    plt.ylim(100, 400)
    plt.ylim(0.8, 1)
    datatable = pandas.read_csv("C:\DRQN_Data\SR_DRQN_07052024.csv")
    ax=seaborn.lineplot(data=datatable, x="Dynamic Obstacles", y="Success Rate", hue="Algorithm", marker='o')
    plt.legend(loc='lower right')
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    plt.show()

#Success Rate,Data Collected Rate,Collision Rate,Average Time
