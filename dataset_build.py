##初始化依赖库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

##数据集构建
# 读取原始数据集
raw_courses = pd.read_csv(
    "./src/dataset/Coursera_courses.csv", header=0, index_col=None, on_bad_lines="skip"
)
raw_reviews = pd.read_csv(
    "./src/dataset/Coursera_reviews.csv", header=0, index_col=None, on_bad_lines="skip"
)
# 有效评论长度确定
length = raw_reviews["reviews"].fillna("").apply(len)
# # 绘图
# plt.figure(figsize=(10, 3), dpi=300)
# # 调整子图间距
# plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.2, hspace=0.4)
# plt.subplot(1, 2, 1)
# plt.hist(np.log10(length + 1e-4), bins=150, density=True)
# plt.title("log10(remark length)")
# plt.axvline(x=1.25, color="red")
# 选取评论大于17字符（log10(length)>=1.25）作为有效评分标准，选取原因见图
raw_reviews["reviews"] = raw_reviews["reviews"].fillna("")
raw_reviews["reviewers"] = raw_reviews["reviewers"].apply(
    lambda x: x.replace("By", "").strip()
)#移除名称中多余的"By"等字符
reviews_vaild = raw_reviews[length > np.power(10, 1.25)]
# 活跃评论用户确定(平均每门课评价数)
review_count = {}
for user, table in reviews_vaild.groupby("reviewers"):
    review_count[user] = table.shape[0] / table["course_id"].unique().shape[0]
review_count = pd.Series(review_count)
# plt.subplot(1, 2, 2)
# plt.title("Review counts per user's course")
# plt.hist(np.log2(review_count), bins=50, density=True)
# plt.axvline(1.55, color="red")
# 选取平均每门学习课程发表2.93条评论的用户作为活跃用户，选取原因见图
user_filtered = review_count[review_count >= np.power(2, 1.55)].index
reviews_clean = reviews_vaild[reviews_vaild["reviewers"].isin(user_filtered)]
print(reviews_clean.shape)
# # 数据集过大，可能无法计算相似性矩阵，随机选取20%数据
# reviews_clean = reviews_clean.loc[np.random.choice(reviews_clean.index,size=math.ceil(0.20*reviews_clean.shape[0]))]
# print(reviews_clean.shape)

# 用户名和课程名称编号
reviews_clean["u_id"] = reviews_clean["reviewers"].astype("category").cat.codes
reviews_clean["c_id"] = reviews_clean["course_id"].astype("category").cat.codes
reviews_clean['date_reviews'] = pd.to_datetime(reviews_clean['date_reviews'])
reviews_clean=reviews_clean.loc[reviews_clean['date_reviews'].sort_values().index]

# 存储编号映射关系
map_users = reviews_clean[["u_id", "reviewers"]].drop_duplicates()
map_users = map_users.set_index(map_users.columns[0]).sort_index()
dict_courses = dict(reviews_clean[["course_id","c_id"]].drop_duplicates().values)

raw_courses=raw_courses.loc[
    raw_courses["course_idname"].isin(dict_courses.keys())]

# 在课程表格中添加课程id信息
raw_courses["c_id"] = raw_courses["course_idname"].apply(lambda x: int(dict_courses[x])).astype('int32')

raw_courses = raw_courses.set_index(raw_courses.columns[4]).sort_index()
raw_courses=raw_courses.loc[raw_courses.index.dropna()]


##存储处理后的表格
reviews_clean.to_csv("./src/dataset/clean_Coursera_reviews.csv",index=None)
raw_courses.to_json("./src/dataset/meta_courses.json",orient="index")
map_users["reviewers"].to_json("./src/dataset/meta_users.json",orient="index")

#数据集创建，按时间排序,后续切分数据集时会选中时间靠后的记录用于测试集
dataset=reviews_clean[["u_id","c_id","rating"]]
#全数据集保存
dataset.to_csv("./src/dataset/dataset.csv",index=None)