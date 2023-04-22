import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#랜덤 시드 설정: 동일한 실행 결과 얻기
np.random.seed(42)

#그래프 출력 설정
plt.rc('font', size=12)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=12)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

#데이터 가져오기
datapath = "https://github.com/ageron/data/raw/main/lifesat/"
gdp_per_capita = pd.read_csv(datapath + "gdp_per_capita.csv",
                             thousands=',',
                             encoding='latin1',
                             na_values="n/a")

#gdp_per_capita = gdp_per_capita[gdp_per_capita["Year"] == 2020]
#국가별 1인당 GDP 데이터에서 2020년 데이터만 추려낸다.
mask_gdp_2020 = gdp_per_capita["Year"] == 2020
gdp_per_capita = gdp_per_capita[mask_gdp_2020]

#국가별 1인당 GDP 데이터 내용 중 "Code" 와 "Year" 열은 필요하지 않기 떄문에 삭제
gdp_per_capita = gdp_per_capita.drop(["Code", "Year"], axis=1)

#1인당 GDP 데이터에서는 국가명 South Korea가 사용되며, 삶의 만족도 데이터에서는 국가명으로 Korea가 사용된다.
#동일한 국가명으로 맞춰주기 위해 1인당 GP 데이터의 South Korea를 Korea로 변경해준다.
gdp_per_capita.loc[gdp_per_capita["Entity"] == "South Korea", "Entity"]


