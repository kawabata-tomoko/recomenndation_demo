from src.dataset import Dataset
# from src.model.CollaborativeFilteringModel import CollaborativeFilteringModel as CFM
# from src.model.ClusterRecommendationModel import ClusterRecommendationModel as CRM
# from src.model import RecommendationCollection as RC
from src.model import RecommendationCollection

# rating_file = './src/dataset/dataset.csv'
# dataset=Dataset.from_csv(rating_file)
# train,test=dataset.data_split(x=2/3,random=False)
#基于用户的协同推荐
# 初始化推荐系统
RCM = RecommendationCollection(
    "./src/dataset/dataset.csv", datasetArgs=dict(x=2 / 3, random=False)
)

#测试实例:基于课程协同过滤
RCM.init_model(
        "CFM",
        modelArgs=dict(
            n_reference=20,  # 协同节点数
            transpose=True,  # 如果需要基于用户协同过滤请改为False
        ),
    ) 
RCM.evaluate(
    n_recommendations=5,
    recommend_args=dict(axis=1, force=False, keep_learnt=True),#如果需要基于用户协同过滤请改为axis=0
    max_iter=50
    )
