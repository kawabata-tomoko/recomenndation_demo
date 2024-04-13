import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


# 近邻聚类推荐
class ClusterRecommendationModel(object): 
    def __init__(self,trainset,testset,transpose=False,cluser_args={}) -> None:
        self.trainset = trainset.transpose if transpose else trainset
        self.testset = testset.transpose if transpose else testset
        self.fit(**cluser_args)
    def Canopy(self,t1,t2,max_iter=100):
        #注意:错误设定阈值可能导致收敛非常缓慢
        points = self.trainset.ratings_matrix.copy()
        neighbors = NearestNeighbors(radius=t1).fit(points)
        canopies = []
        iter_count = 0
        while not points.empty and iter_count < max_iter:
            center_point = points.iloc[0]
            canopies.append(center_point.values.tolist())
            distances, indices = neighbors.radius_neighbors([center_point])
            close_points = indices[0][np.where(distances[0] <= t2)]
            points =points.loc[points.index.difference(close_points)]
            iter_count+=1
        return canopies
    def fit(self,t1,t2, random_state=42, **cluser_args):
        # 使用Canopy算法确定初始中心点,通识介绍可参考:https://zhuanlan.zhihu.com/p/112801784
        canopies=self.Canopy(t1,t2)
        # 使用Canopy算法得到的中心点作为KMeans算法的初始中心
        self.model = KMeans(n_clusters=len(canopies), init=np.array(canopies), n_init=1, random_state=random_state, **cluser_args)
        #对用户聚类,记录用户所属簇信息
        self.user_clusters = pd.Series(
            self.model.fit_predict(self.trainset.ratings_matrix),
            index=self.trainset.ratings_matrix.index)
        
    def recommend(self,user_id, n_recommendations=10,**kwargs):
        N=n_recommendations
        # 获取用户所属簇
        cluster_id = self.user_clusters[user_id]
        cluster_rating = self.trainset.ratings_matrix[np.where(self.user_clusters==cluster_id)[0]]
        # 获取在同一簇中受欢迎的课程
        return cluster_rating.mean(axis=1).sort_values(ascending=False)[:N]

    