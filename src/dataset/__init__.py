import pandas as pd
import numpy as np

import math


class Dataset:
    @classmethod
    def from_csv(cls, filename, dtype=None,**kwargs):
        """从文件读取数据集

        Args:
            filename (str): 数据集路径
            dtype (dict): 读取列名称和数据类型，默认{"u_id": np.int32, "c_id": np.int32, "rating": np.float32}

        Returns:
            Dataset: 读取的数据集
        """
        dtype = dtype or {"u_id": np.int32, "c_id": np.int32, "rating": np.float32}
        cratings = pd.read_csv(filename, dtype=dtype)
        return cls(data=cratings,**kwargs)
    
    def __init__(self, data=None, pivotArgs=None,**kwargs):
        self.data=data if data is not None else pd.DataFrame()
        pivotArgs=pivotArgs or dict(index='u_id', columns='c_id', values='rating', aggfunc='mean')
        # 构建评分矩阵，并处理NaN（缺失值）为0
        self.ratings_matrix=data.pivot_table(**pivotArgs).fillna(0)
        
        self.__dict__.update(kwargs)

    def data_split(self,x=0.75,column="u_id",random=False,seed=42):
        """数据集切分

        Args:
            x (float, optional): 训练集切分比例（由于舍入会比该指标略小）.默认取 0.75.
            column (str, optional): 关注的列名，可选"u_id"或"c_id". Defaults to "u_id".
            random (bool, optional): 是否随机切分数据. Defaults to False.
            seed (int, optional): 随机种子(random=False无效). Defaults to 42.

        Returns:
            Dataset: 训练集和测试集
        """
        train_index = []
        # 为了保证每个用户在测试集和训练集都有数据，因此默认按userId聚合
        for uid ,table in self.data.groupby(column):
            # 将每个用户的x比例的数据作为训练集，剩余的作为测试集
            sample_size=round(len(table)*x)
            if random:
                np.random.seed(seed=seed)
                train_index.extend(np.random.choice(table.index, size=sample_size, replace=False))
            else:
                train_index.extend(table.index.values[:sample_size])

        trainset = self.data.loc[train_index]
        testset = self.data.drop(train_index)
        
        print('Split trainingSet and testSet success!')
        print('TrainSet:', trainset.shape)
        print('TestSet:' , testset.shape)
        return Dataset(data=trainset),Dataset(data=testset)
    @property
    def transpose(self):
        self.ratings_matrix=self.ratings_matrix.copy().T
        return self
