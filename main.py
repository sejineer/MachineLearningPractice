import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

# 랜덤 시드 설정: 동일한 실행 결과 얻기
np.random.seed(42)

# 그래프 출력 설정
plt.rc('font', size=12)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=12)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

# 데이터 가져오기
datapath = "https://github.com/ageron/data/raw/main/lifesat/"
gdp_per_capita = pd.read_csv(datapath + "gdp_per_capita.csv",
                             thousands=',',
                             encoding='latin1',
                             na_values="n/a")
# gdp_per_capita = gdp_per_capita[gdp_per_capita["Year"] == 2020]
# 국가별 1인당 GDP 데이터에서 2020년 데이터만 추려낸다.
mask_gdp_2020 = gdp_per_capita["Year"] == 2020
gdp_per_capita = gdp_per_capita[mask_gdp_2020]

# 국가별 1인당 GDP 데이터 내용 중 "Code" 와 "Year" 열은 필요하지 않기 떄문에 삭제
gdp_per_capita = gdp_per_capita.drop(["Code", "Year"], axis=1)

# 1인당 GDP 데이터에서는 국가명 South Korea가 사용되며, 삶의 만족도 데이터에서는 국가명으로 Korea가 사용된다.
# 동일한 국가명으로 맞춰주기 위해 1인당 GP 데이터의 South Korea를 Korea로 변경해준다.
gdp_per_capita.loc[gdp_per_capita["Entity"] == "South Korea", "Entity"] = "Korea"

# 1인당 GDP 데이터 테이블 각 열(column)의 인덱스를 다음과 같이 변경해준다.
gdp_per_capita.columns = ["Country", "GDP per capita (USD)"]

# 국가명을 행 인덱스로 지정한다. 총 224개 국가의 1인당 GDP 정보를 담고있다.
# inplace=True는 set_index 실행 결과로 gdp_per_capita 데이터프레임 자체가 업데이트 되도록 한다.
gdp_per_capita.set_index("Country", inplace=True)

# OECD 국가별 삶의 만족도 데이터는 '더 나은 삶의 지수'데이터 파일에 포함되어 있다. 먼저 해당 csv 파일을 판다스의 데이터프레임
# 객체로 불러온 후에 삶의 만족도와 관련된 내용을 추출하는 과정을 수행하도록 한다.
oecd_bli = pd.read_csv(datapath + "oecd_bli.csv", thousands=',')

#전체 인구를 대상으로 하는 TOT 기준에 포함된 데이터만 추출한다.
oecd_bli = oecd_bli[oecd_bli["INEQUALITY"] == "TOT"]

"""
데이터프레임 객체의 pivot()메서드를 이용하여 'Indicator'의 항목에 대한 각 국가별 수치만을 추출하기 위해 국가명
('Country' 열의 항목)을 행의 인덱스 이름으로, 'Indicator'의 항목을 열의 인덱스 이름으로 사용하면서 해당 행과 열의
항목에는 'Value' 열에 포함된 값을 사용하는 데이터프레임을 아래와 같이 생성할 수 있다.
"""
oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")

"""
1인당 GDP 데이터는 OECD 회원국 이외의 국가 데이터도 포함되어 있다. OECD 회원국으로 제한해서 1인당 GDP와 삶의
만족도 사이의 관계를 파악하기 위해 앞서 구한 두 개의 데이터 프레임을 하나로 병합한다.
"""
oecd_country_stats = pd.merge(left=gdp_per_capita['GDP per capita (USD)'],
                              right=oecd_bli['Life satisfaction'],
                              left_index=True, right_index=True)

# 1인당 GDP와 삶의 만족도 사이의 선형 관계를 눈으로 확인하기 위해 국가를 1인당 GDP 기준 오름차순으로 졍렬시킨다.
oecd_country_stats.sort_values(by="GDP per capita (USD)", inplace=True)

"""
다음 9개 국가의 데이터를 데이터 셋에서 제외시키고 훈련 시킬 때와 그렇지 않을 때를 비교하고자 한다.
제외 대상 국가는 1인당 GDP가 $23,500 이하이거나 $62,500이상인 국가들이다. 데이터프레임에서 아래 인덱스를 이용하여
9개 국가를 제외시키고 훈련을 시킬 것이다.
"""
omitted_indices = [0, 1, 2, 3, 4, 33, 34, 35, 36]
kept_indices = list(set(range(37)) - set(omitted_indices))

# 제외된 9개 국가의 1인당 GDP와 삶의 만족도 데이터
missing_data = oecd_country_stats.iloc[omitted_indices]

# 9개 국가를 제외한 국가들의 1인당 GDP와 삶의 만족도 데이터
sample_data = oecd_country_stats.iloc[kept_indices]

"""
아래 코드는 앞서 언급된 9개 국가의 데이터를 제외한 국가들의 1인당 GDP와 삶의 만족도 사이의 관계를 산점도로 나타낸다.
선형관계를 잘 보여주는 다음 다섯 개 국가는 빨간색 점으로 표시한다.
헝가리, 대한민국, 프랑스, 호주, 미국
"""
# 9개 국가를 제외한 국가들의 데이터 산점도
sample_data.plot(kind='scatter', x="GDP per capita (USD)", y='Life satisfaction', figsize=(5, 3))
plt.axis([10000, 70000, 0, 10])

# 언급된 5개 국가명과 좌표
position_text = {
    "Hungary" : (15000, 3),
    "Korea" : (24000, 1.7),
    "France" : (33000, 2.2),
    "Australia" : (43000, 2.7),
    "United States" : (52000, 3.8),
}

# 5개 국가는 좌표를 이용하여 빨간색 점으로 표기
for country, pos_text in position_text.items():
    pos_data_x, pos_data_y = sample_data.loc[country] # 5개 국가의 지표

    # 5개 국가명 표기
    country = "U.S." if country == "United States" else country # 미국은 U.S. 로 표기
    # 화살표 그리기
    plt.annotate(country, xy=(pos_data_x, pos_data_y),
                 xytext=pos_text,
                 arrowprops=dict(facecolor='black', width=0.5, shrink=0.1, headwidth=5))

    # 5개 국가 산점도 그리기: 빨간색 점
    plt.plot(pos_data_x, pos_data_y, "ro")

# x축 제목 새롭게 지정
plt.xlabel("GDP per capita (USD)")

"""
직선처럼 y축의 값이 x축의 값에 선형적으로 의존하는 관계를 선형관게라 하며, 그런 선형관계를 함수로 구현하는 모델을
선현회귀 모델(linear regression model)이라 한다. 선형회귀 모델은 직선을 나타내는 1차 함수의 그래프로 표현되며, 직선은
절편(y축과 만나는 점)과 기울기 두 개의 값에 의해 정해진다.

예를 들어, 1인당 GDP와 삶의 만족도 사이에 다음 선형 관계가 성립하도록 하는 절편 𝜃0와 기울기 𝜃1을 구해야 한다.
Life Satisfaction = 𝜃0+𝜃1⋅GDP per Capita
𝜃0  와  𝜃1  처럼 모델 구현에 사용되는 값들을 모델의 파라미터parameter라 하며, 
모델의 파라미터를 찾아내는 것이 머신러닝 모델훈련의 핵심이다.
"""

"""
사이킷런(scikit-learn)라이브러리는 머신러닝에서 사용되는 다양한 모델을 제공한다. 선형회귀의 경우 LinearRegression
클래스의 객체를 생성하여 훈련시키면 최적의 절편과 기울기를 계산해준다.
"""

"""
과정 1: 모델 지정 / 사이킷런 패키지의 linear_model 모듈에 포함된 LinearRegression 클래스의 객체를 선언헌다. 선언된 모델은
아직 어떤 훈련도 하지 않은 상태이다.
"""
lin1 = linear_model.LinearRegression()

"""
과정 2: 훈련 셋 지정 / 입력 데이터 셋은 x에, 타깃 데이터 셋은 y에 해당한다.
입력 데이터: 1인당 GDP
타깃 데이터: 삶의 만족도
입력 데이터와 타깃 데이터를 2차원 어레이로 지정한다. 사이킷런의 선형회귀 모델이 어레이 형식의 입력 데이터 셋과 타깃 데이터 셋을 요구한다.
넘파이의 c_ 함수를 활용해서 차원을 늘려준다.
"""
Xsample = np.c_[sample_data["GDP per capita (USD)"]]
Ysample = np.c_[sample_data["Life satisfaction"]]

"""
과정 3: 모델 훈련 / 선형 모델의 fit() 메서드를 지정된 입력 데이터 셋과 타깃 데이터 셋을 인자로 사용해서 호출하면 최적의
𝜃0 와  𝜃1 파라미터를 찾는 훈련이 실행되며 훈련이 종료되면 최적의 파라미터가 업데이트된 객체 자신이 반환된다.
"""
lin1.fit(Xsample, Ysample)

"""
훈련된 모델이 알아낸 최적 선형 모델의 절편과 기울기는 아래 두 속성에 저장된다.
intercept_[0]: 직선의 절편
coef_[0]: 직선의 기울기
"""
t0, t1 = lin1.intercept_[0], lin1.coef_[0][0]
print(f"절편:\t {t0}")
print(f"기울기:\t {t1}")

"""
구해진 기울기와 절편을 이용하여 산점도와 함께 직선을 그리면 다음과 같다.
"""
# 산점도
sample_data.plot(kind='scatter',x="GDP per capita (USD)", y="Life satisfaction", figsize=(5, 3))
plt.xlabel("GDP per capita (USD)")
plt.axis([10000, 70000, 0, 10])

# 직선 그리기
X=np.linspace(0, 70000, 1000)
plt.plot(X, t0 + t1*X, "b")

#직선의 절편과 기울기 정보 명시
plt.text(15000, 3.1, r"$\theta_0 = 4.85$", fontsize=14, color="b")
plt.text(15000, 2.2, r"$\theta_1 = 4.91 \times 10^{-5}$", fontsize=14, color="b")

plt.show()

"""
한 국가의 1인당 GDP가 알려졌을 때 훈련된 모델에 포함된 𝜃0  와 𝜃1을 이용하여 해당 국가의 삶의 만족도를 예측한다.
예를 들어, OECD에 속하지 않는 키프러스(Cyprus)의 1인당 GDP를 이용하여 키프러스 국민의 삶의 만족도를 예측한다.
"""
cyprus_gdp_per_capita = gdp_per_capita.loc["Cyprus"]["GDP per capita (USD)"]

#훈련된 모델의 predict()메서드가 식을 이용하여 삶의 만족도를 게산한다.
cyprus_predicted_life_satisfaction = lin1.predict([[cyprus_gdp_per_capita]])[0, 0]
print(cyprus_predicted_life_satisfaction)