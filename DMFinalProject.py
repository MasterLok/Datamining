##导入需要使用的库
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from lightgbm.sklearn import LGBMRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV

##导入数据
Train_data = pd.read_csv('Train/used_car_train_20200313.csv', sep=' ')
Test_data = pd.read_csv('Test/used_car_testB_20200421.csv', sep=' ')
#合并方便后面操作
df = pd.concat([Train_data, Test_data], ignore_index=True)

##数据分析
"""
#'price'为长尾分布，需要做数据转换
#plt.figure()
plt.figure(figsize=(10, 3))
plt.subplot(1, 2,1)
sns.distplot(Train_data['price'])
plt.subplot(1,2,2)
Train_data['price'].plot.box()

#'price'转化后的分布
plt.figure()
sns.distplot(np.log1p(Train_data['price']))
"""
"""
#观察数值特征分布
num_fea = ['power', 'kilometer','price','v_0', 'v_1', 'v_2', 'v_3','v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12','v_13', 'v_14']
f = pd.melt(Train_data, value_vars=num_fea)
g = sns.FacetGrid(f, col="variable", col_wrap=3, sharex=False, sharey=False)
g = g.map(sns.distplot, "value")
plt.tight_layout()


#热图
corr1 = abs(df[df['price'].notnull()][num_fea].corr())
plt.figure(figsize=(10, 10))
sns.heatmap(corr1, linewidths=0.1, annot=True, cmap="YlOrRd")
"""
#plt.show()

##数据清洗
#统计'name'重复值
df['name_count'] = df.groupby(['name'])['SaleID'].transform('count')
del df['name']
del df['offerType']
del df['seller']

#对'price'做对数变换
df['price'] = np.log1p(df['price'])

#用众数填充缺失值
df['fuelType'] = df['fuelType'].fillna(0)
df['gearbox'] = df['gearbox'].fillna(0)
df['bodyType'] = df['bodyType'].fillna(0)
df['model'] = df['model'].fillna(0)

#处理异常值
df['power'] = df['power'].map(lambda x: 600 if x>600 else x) #限定power<=600
df['notRepairedDamage'] = df['notRepairedDamage'].astype('str').apply(lambda x: x if x != '-' else None).astype('float32') #类型转换

#对可分类的连续特征进行分桶
bin = [i*10 for i in range(31)]
df['power_bin'] = pd.cut(df['power'], bin, labels=False)
bin = [i*10 for i in range(24)]
df['model_bin'] = pd.cut(df['model'], bin, labels=False)

##特征工程
#时间提取出年月日和使用时间
from datetime import datetime
def date_process(x):
    year = int(str(x)[:4])
    month = int(str(x)[4:6])
    day = int(str(x)[6:8])
    if month < 1:
        month = 1
    date = datetime(year, month, day)
    return date
df['regDate'] = df['regDate'].apply(date_process)
df['creatDate'] = df['creatDate'].apply(date_process)
df['regDate_year'] = df['regDate'].dt.year
df['regDate_month'] = df['regDate'].dt.month
df['regDate_day'] = df['regDate'].dt.day
df['creatDate_year'] = df['creatDate'].dt.year
df['creatDate_month'] = df['creatDate'].dt.month
df['creatDate_day'] = df['creatDate'].dt.day
df['car_age_day'] = (df['creatDate'] - df['regDate']).dt.days#二手车使用天数
df['car_age_year'] = round(df['car_age_day'] / 365, 1)#二手车使用年数

#类别特征对价格的统计最大，最小，平均值等等
car_cols = ['brand','model','kilometer','fuelType','bodyType']
for col in car_cols:
    t = Train_data.groupby(col,as_index=False)['price'].agg(
        {col+'_count':'count',col+'_price_max':'max',col+'_price_median':'median',
         col+'_price_min':'min',col+'_price_sum':'sum',col+'_price_std':'std',col+'_price_mean':'mean'})
    df = pd.merge(df,t,on=col,how='left')

#行驶路程与功率统计
kp = ['kilometer','power']
t1 = Train_data.groupby(kp[0],as_index=False)[kp[1]].agg(
        {kp[0]+'_'+kp[1]+'_count':'count',kp[0]+'_'+kp[1]+'_max':'max',kp[0]+'_'+kp[1]+'_median':'median',
         kp[0]+'_'+kp[1]+'_min':'min',kp[0]+'_'+kp[1]+'_sum':'sum',kp[0]+'_'+kp[1]+'_std':'std',kp[0]+'_'+kp[1]+'_mean':'mean'})
df = pd.merge(df,t1,on=kp[0],how='left')

#由前面数据探索的结果可知部分v_0,v_3,v_8,v_12，kilometer与price的相关性很高，所以做一些简单组合
num_cols = [0,3,8,12]
for i in num_cols:
    for j in num_cols:
        df['new'+str(i)+'*'+str(j)]=df['v_'+str(i)]*df['v_'+str(j)]
        
for i in num_cols:
    for j in num_cols:
        df['new'+str(i)+'+'+str(j)]=df['v_'+str(i)]+df['v_'+str(j)]


for i in num_cols:
    for j in num_cols:
        df['new'+str(i)+'-'+str(j)]=df['v_'+str(i)]-df['v_'+str(j)]


for i in range(15):
    df['new'+str(i)+'*year']=df['v_'+str(i)] * df['car_age_year']

##建立模型预测
#划分训练数据和测试数据
df1 = df.copy()
test = df1[df1['price'].isnull()]
X_train = df1[df1['price'].notnull()].drop(['price','regDate','creatDate','SaleID','regionCode'],axis=1)
Y_train = df1[df1['price'].notnull()]['price']
X_test = df1[df1['price'].isnull()].drop(['price','regDate','creatDate','SaleID','regionCode'],axis=1)
print("test information:")
test.info()
print("X_train information:")
X_train.info()
cols = list(X_train)
oof = np.zeros(X_train.shape[0])
sub = test[['SaleID']].copy()
sub['price'] = 0
feat_df = pd.DataFrame({'feat': cols, 'imp': 0})
skf = KFold(n_splits=10, shuffle=True, random_state=2022)

#定义xgb和lgb模型函数
clf = LGBMRegressor(
    n_estimators=10000,
    learning_rate=0.02,
    boosting_type= 'gbdt',
    objective = 'regression_l1',
    max_depth = -1,
    num_leaves=31,
    min_child_samples = 20,
    feature_fraction = 0.8,
    bagging_freq = 1,
    bagging_fraction = 0.8,
    lambda_l2 = 2,
    random_state=2022,
    metric='mae',
    device = 'gpu'  #不使用gpu这个也注释掉
    )

lgb_mae = 0
sub_lgb = 0
for i, (trn_idx, val_idx) in enumerate(skf.split(X_train, Y_train)):
    print('--------------------- {} fold ---------------------'.format(i+1))
    trn_x, trn_y = X_train.iloc[trn_idx].reset_index(drop=True), Y_train[trn_idx]
    val_x, val_y = X_train.iloc[val_idx].reset_index(drop=True), Y_train[val_idx]
    clf.fit(
        trn_x, trn_y,
        eval_set=[(val_x, val_y)], 
        eval_metric='mae',
        early_stopping_rounds=300,
        #verbose_eval=300
        verbose=False
    )
    
    sub_lgb += np.expm1(clf.predict(X_test)) / skf.n_splits
    val_lgb = clf.predict(val_x)
    #print('val mae:', mean_absolute_error(np.expm1(val_y), np.expm1(val_lgb)))
    lgb_mae += mean_absolute_error(np.expm1(val_y), np.expm1(val_lgb))/skf.n_splits
print('MAE of val with lgb:', lgb_mae)

xlf= XGBRegressor(
    tree_method='gpu_hist',
    gpu_id='0',
    n_estimators=1000,
    gamma=0, subsample=0.8,
    colsample_bytree=0.9,
    max_depth=7
    )#, objective ='reg:squarederror'
param_grid = {'learning_rate': [0.01, 0.05, 0.1, 0.2]}

gbm = GridSearchCV(xlf, param_grid)
xgb_mae = 0
sub_xgb = 0

for i, (trn_idx, val_idx) in enumerate(skf.split(X_train, Y_train)):
    print('--------------------- {} fold ---------------------'.format(i+1))
    trn_x, trn_y = X_train.iloc[trn_idx].reset_index(drop=True), Y_train[trn_idx]
    val_x, val_y = X_train.iloc[val_idx].reset_index(drop=True), Y_train[val_idx]
    gbm.fit(
        trn_x, trn_y,
        eval_set=[(val_x, val_y)], 
        eval_metric='mae',
        early_stopping_rounds=300,
        verbose=False
    )
    sub_xgb += np.expm1(gbm.predict(X_test)) / skf.n_splits
    val_xgb= gbm.predict(val_x)
    #print('val mae:', mean_absolute_error(np.expm1(val_y), np.expm1(val_xgb)))
    xgb_mae += mean_absolute_error(np.expm1(val_y), np.expm1(val_xgb))/skf.n_splits
print('MAE of val with xgb:', xgb_mae)

##进行模型融合
#采取了加权融合的方式
val_Weighted = (1-lgb_mae/(lgb_mae+xgb_mae))*np.expm1(val_lgb)+(1-xgb_mae/(lgb_mae+xgb_mae))*np.expm1(val_xgb)
val_Weighted[val_Weighted<0]=10 #预测的最小值有负数，而真实情况下，price为负是不存在的
print('MAE of val with Weighted ensemble:',mean_absolute_error(np.expm1(val_y),val_Weighted))

sub_Weighted = (1-lgb_mae/(lgb_mae+xgb_mae))*sub_lgb+(1-xgb_mae/(lgb_mae+xgb_mae))*sub_xgb
sub['price'] = sub_Weighted

##生成提交文件
sub.to_csv('./sumbit3.csv',index=False)