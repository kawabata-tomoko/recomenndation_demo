# coding = utf-8

# 协同过滤推荐算法实现
from warnings import warn

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class CollaborativeFilteringModel(object):
    def __init__(
        self,
        trainset,
        testset,
        n_reference=20,
        transpose=False,
    ) -> None:
        self.n_reference = n_reference
        self.similarity_matrix = pd.DataFrame()
        # 默认dataset行为userid，列为courseid
        # 不转置为基于用户的协同过滤推荐，转置后为基于物品/课程的协同过滤推荐
        self.transpose = transpose
        self.trainset = trainset.transpose if transpose else trainset
        self.testset = testset.transpose if transpose else testset

    def get_items_sim_mat(self, index):
        # 课程数量较少时可以直接计算相似性矩阵
        if self.item_sim_mat.empty:
            mat = self.trainset.ratings_matrix
            self.item_sim_mat = pd.DataFrame(
                cosine_similarity(mat), index=mat.index, columns=mat.index
            )
            print("Build co-rated user_ids matrix success!")
        return self.item_sim_mat.loc[index, self.item_sim_mat.index != index]

    def get_similarity(self, query, index):
        # 数据量过多时不预先计算相似性矩阵，在查询推荐结果的同时计算当前查询向量与其他向量相似性以减少内存压力
        # mat = (
        #     self.trainset.ratings_matrix.T
        #     if self.transpose
        #     else self.trainset.ratings_matrix
        # )

        if index in  self.trainset.ratings_matrix.index:
            others =  self.trainset.ratings_matrix.loc[ self.trainset.ratings_matrix.index != index]
        else:
            others =  self.trainset.ratings_matrix
        return pd.DataFrame(cosine_similarity([query], others), columns=others.index)

    def base_users(self, user_id, user_rank, keep_learnt=False, ignore_index=[]):
        K = self.n_reference
        rank = pd.Series(np.zeros(shape=len(user_rank.index)), index=user_rank.index)

        # 获取与目标用户最相似的K个用户的ID
        sim_score = self.get_similarity(user_rank, user_id)
        # similar_users = sim_score.T.sort_values(by=0, ascending=False)[:K]
        similar_users = sim_score.T.nlargest(K,columns=0)

        # 遍历最相似的用户，并为目标用户推荐课程
        for v, weight in similar_users[0].items():
            for course, rating in self.trainset.ratings_matrix.loc[v].items():
                # 排除目标用户已经学过的课程
                if keep_learnt or course not in ignore_index:
                    rank[course] += weight * rating  # 考虑相似用户对课程的评分
        return rank / K

    def base_items(self, user_rank, keep_learnt=False, ignore_index=[]):
        K = self.n_reference
        rank = pd.Series(np.zeros(shape=len(user_rank.index)), index=user_rank.index)
        learnt_courses = user_rank[user_rank != 0]
        # 根据已学习的课程查找相关课程
        for course, rating in learnt_courses.items():
            sim_score = self.get_similarity(
                self.trainset.ratings_matrix.loc[course], course
            )
            # similar_courses = sim_score.T.sort_values(by=0, ascending=False)[:K]
            similar_courses = sim_score.T.nlargest(K,columns=0)
            for ralated_course, w in similar_courses.iterrows():
                if keep_learnt or course not in ignore_index:
                    rank[ralated_course] += w * float(rating)
        return rank / K

    def query_constructor(self, user_id, axis=0, force=None):
        query = None
        if user_id in self.trainset.ratings_matrix.index:
            query = (
                self.trainset.ratings_matrix.loc[user_id]
                if axis == 0
                else self.trainset.ratings_matrix[user_id]
            )
        elif force:
            query = np.ones(self.trainset.ratings_matrix.shape[1]) * 5
        else:
            warn(
                f"{user_id} is invaild for no history data", category=None, stacklevel=1
            )
        return query

    def recommend(
        self, user_id, n_recommendations=10, axis=0,force=False, keep_learnt=True,**kwargs
    ):
        """推荐列表函数

        Args:
            user_id (int): 待推荐用户id
            axis (int, optional): 描述协同过滤轴向,设定为0则基于用户进行协同过滤(需要初始化时transpose=False),设定为1基于课程进行协同过滤(需要初始化时transpose=True). Defaults to 0.
            n_recommendations (int, optional): 推荐课程数量. Defaults to 10.
            force (bool, optional): 是否强制输出结果(设定为True时即便用户无历史信息也会假定对所有课程感兴趣来进行计算,不推荐). Defaults to False.
            keep_learnt (bool, optional): 是否保留已学习的课程(用于评估模式). Defaults to True.

        Returns:
            pd.Series: 返回的前N个推荐课程列表
        """
        N = n_recommendations
        query = self.query_constructor(user_id, axis=axis, force=force)
        ignore_index = (
            query[query != 0] if keep_learnt else []
        )  # 检索已经完成学习的课程
        rank = pd.DataFrame(np.zeros(shape=len(query.index)), index=query.index)#初始化待推荐课表
        if axis == 0:#对行协同过滤，即基于用户进行协同过滤推荐
            rank = self.base_users(
                user_id, query, keep_learnt=keep_learnt, ignore_index=ignore_index
            )
        else:#对列协同过滤，即基于课程进行协同过滤推荐
            rank = self.base_items(
                query, keep_learnt=keep_learnt, ignore_index=ignore_index
            )

        return rank.nlargest(N)#返回对应推荐列表
