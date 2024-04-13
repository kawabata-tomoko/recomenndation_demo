from .CollaborativeFilteringModel import CollaborativeFilteringModel as CFM
from .ClusterRecommendationModel import ClusterRecommendationModel as CRM
from ..dataset import Dataset

import pandas as pd
import os
import math


class RecommendationCollection(object):
    def __init__(self, dataPath, datasetArgs, meta_user=None, meta_course=None) -> None:
        self.dataset = Dataset.from_csv(dataPath)
        datadir = os.path.dirname(os.path.abspath(dataPath))
        # 读取课程和用户信息用于后续显示
        self.meta_user = (
            pd.read_json(meta_user, orient="index")
            if meta_user is not None
            else pd.read_json(os.path.join(datadir, "meta_users.json"), orient="index")
        )
        self.meta_course = (
            pd.read_json(meta_course, orient="index")
            if meta_course is not None
            else pd.read_json(
                os.path.join(datadir, "meta_courses.json"), orient="index"
            )
        )
        # 分割数据集
        self.trainset, self.testset = self.dataset.data_split(**datasetArgs)

    def init_model(self, method, modelArgs):
        # 初始化模型
        self.model = (
            CFM(self.trainset, self.testset, **modelArgs)
            if method == "CFM"
            else CRM(self.trainset, self.testset, **modelArgs)
        )

    def recommend(self, user_id, n_recommendations=10, recommend_args={}):
        user_id = int(user_id)
        return self.model.recommend(
            user_id, n_recommendations=n_recommendations, **recommend_args
        ).index

    # 产生推荐并通过准确率、召回率、F1-score和覆盖率进行评估
    def evaluate(self, n_recommendations=10, recommend_args={}, max_iter=2000):
        print("Evaluation start ...")
        # 准确率\召回率\F1-score
        hit = 0  # 正确数,TP
        rec_count = 0  # 推荐总数TP+FP
        test_count = 0  # 真实总数TP+FN
        # 覆盖率
        all_rec_ids = set()
        iter_count = 0  # 考虑到测试集过大,提供提前终止参数
        for user, test_table in self.testset.data.groupby("u_id"):
            test_id = set(test_table["c_id"])
            rec_id = set(self.recommend(user, n_recommendations, recommend_args))
            all_rec_ids.union(rec_id)
            hit += len(rec_id.intersection(test_id))
            rec_count += len(rec_id)
            test_count += len(test_id)
            iter_count += 1
            if iter_count == max_iter:
                break
        precision = hit / (1.0 * rec_count)
        recall = hit / (1.0 * test_count)
        F1 = 2 * (precision * recall) / (precision + recall)
        coverage = len(all_rec_ids) / (1.0 * len(self.trainset.data["c_id"].unique()))
        print(
            "precisioin=%.4f\trecall=%.4f\tF1-score=%.4f\tcoverage=%.4f"
            % (precision, recall, F1, coverage)
        )

    # 获取用户已学课程信息列表。
    def get_user_courses_history(self, user_id, top_num=10):
        # 将user_id转换为整数
        user_id = int(user_id)
        # 获取某个用户的评分数据
        user_ratings = self.trainset.ratings_matrix.loc[user_id]

        # 获取用户评分前10的课程信息和评分
        top_rated_courses = user_ratings.nlargest(top_num)
        top_rated_courses_info = []

        for course_id, rating in top_rated_courses.items():
            course_info = self.get_course_info_by_id(course_id)
            if course_info:
                course_info["user_rating"] = rating
                top_rated_courses_info.append(course_info)

        return top_rated_courses_info

    # 根据用户id获取用户信息
    def get_user_info(self, user_id):
        # 将user_id转换为整数
        user_id = int(user_id)
        # 将用户信息的字段值转换为基本的 Python 数据类型
        return {
            "user_id": user_id,
            "reviewers": self.meta_user.loc[user_id, 0],
        }

    # 通过id获取课程信息
    def get_course_info_by_id(self, course_id):
        course_id = int(course_id)
        course_info = self.meta_course.loc[course_id, :]

        # 返回课程信息，省略 'course_idname'
        return {
            "name": course_info["name"],
            "institution": course_info["institution"],
            "course_url": course_info["course_url"],
            "course_id": course_id,
        }

    # 通过一组id获取一组课程信息
    def get_course_info_by_id_list(self, course_id_list):
        # 筛选包含在课程ID列表中的课程信息
        selected_index = self.meta_course.index.intersection(course_id_list)
        selected_courses = self.meta_course.loc[selected_index]

        # 如果没有匹配的课程信息，返回空列表
        if selected_courses.empty:
            return []

        # 获取每个课程的平均评分和评分数量
        course_ratings_info = []
        for course_id in course_id_list:
            # 获取课程的评分信息
            course_reviews = self.trainset.ratings_matrix[course_id]
            course_reviews = course_reviews[course_reviews!=0]
            # 计算平均评分和评分数量
            avg_rating = course_reviews.mean()
            num_ratings = course_reviews.shape[0]

            # 获取课程的其他信息
            course_info = selected_courses.loc[course_id]

            # 构建包含信息的字典
            course_dict = course_info.to_dict()
            course_dict["average_rating"] = avg_rating
            course_dict["num_ratings"] = num_ratings

            # 将字典添加到结果列表
            course_ratings_info.append(course_dict)

        return course_ratings_info

    # 通过模糊查询课程名字获取课程列表
    def search_courses_by_name(self, query):
        # 将查询字符串转换为小写，以便不区分大小写进行搜索
        query_lower = query.lower()

        # 使用 Pandas 字符串方法进行模糊查找
        matching_courses = self.meta_course[
            self.meta_course["name"].str.lower().str.contains(query_lower)
        ]

        # 将符合条件的课程信息提取为字典列表
        search_results = []
        for index, course in matching_courses.iterrows():
            search_results.append(course.to_dict())

        return search_results

    # 分页查询课程信息
    def get_courses_for_page(self, page_number, page_size):
        # 计算起始索引和结束索引
        start_idx = (page_number - 1) * page_size
        end_idx = start_idx + page_size

        # 获取指定页的课程信息
        page_courses = self.meta_course.iloc[start_idx:end_idx]

        # 计算总页数
        total_pages = math.ceil(len(self.meta_course) / page_size)

        return {"courses": page_courses.to_dict("records"), "total_pages": total_pages}

    # 分页查询用户信息
    def get_user_for_page(self, page_number, page_size):
        # 计算起始索引和结束索引
        start_idx = (page_number - 1) * page_size
        end_idx = start_idx + page_size

        # 获取指定页的课程信息
        page_courses = self.meta_user.iloc[start_idx:end_idx]

        # 计算总页数
        total_pages = math.ceil(len(self.meta_user) / page_size)

        return {
            "users": [
                {"user_id": k, "reviewers": list(v.values())[0]}
                for k, v in page_courses.to_dict("index").items()
            ],
            "total_pages": total_pages,
        }
