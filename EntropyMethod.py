import pandas as pd
class EntropyMethod:
    def __init__(self, index, positive, negative, row_name):
        if len(index) != len(row_name): # 比较数据指标行数以及期刊的行数
            raise Exception('数据指标行数与行名称数不符')
        if sorted(index.columns) != sorted(positive+negative): # 比较正向负向指标的项目数有总指标项目数
            raise Exception('正项指标加负向指标不等于数据指标的条目数')

        self.index = index.copy().astype('float64') # 复制原数据集并将数据类型改为float64
        self.positive = positive
        self.negative = negative
        self.row_name = row_name.copy() # 复制期刊名称列
    
    def uniform(self):
        uniform_mat = self.index.copy() # 复制原数据集作为初始归一化矩阵
        min_index = {column:min(uniform_mat[column]) for column in uniform_mat.columns} # 分别获得每一列的最大值和最小值并建立字典
        max_index = {column:max(uniform_mat[column]) for column in uniform_mat.columns}
        for i in range(len(uniform_mat)): # 对于每一行
            for column in uniform_mat.columns: # 对于此行每一列
                if column in self.negative: # 如果这一列属于负向指标
                    uniform_mat[column][i] = (uniform_mat[column][i] - min_index[column]) / (max_index[column] - min_index[column]) # 分别归一化
                else:
                    uniform_mat[column][i] = (max_index[column] - uniform_mat[column][i]) / (max_index[column] - min_index[column])

        self.uniform_mat = uniform_mat
        return self.uniform_mat # 返回归一化矩阵
    def calc_probability(self):
        try:
            p_mat = self.uniform_mat.copy()
        except AttributeError:
            raise Exception('需调用uniform方法')
        for column in p_mat.columns:
            sigma_x_1_n_j = sum(p_mat[column]) # 求归一化矩阵每一列的和
            # 为了取对数计算时不出现无穷,将比重为0的值修改为1e-6
            p_mat[column] = p_mat[column].apply(lambda x_i_j: x_i_j / sigma_x_1_n_j if x_i_j / sigma_x_1_n_j != 0 else 1e-6) # 计算每个实例占这个指标的比重

        self.p_mat = p_mat
        return p_mat
    def calc_entropy(self):
        try:
            self.p_mat.head(0)
        except AttributeError:
            raise Exception('需调用calc_probability方法')

        import numpy as np
        e_j  = -(1 / np.log(len(self.p_mat))) * np.array([sum([pij*np.log(pij) for pij in self.p_mat[column]]) for column in self.p_mat.columns]) #计算指标的熵值
        ejs = pd.Series(e_j, index=self.p_mat.columns, name='指标的熵值')

        self.entropy_series = ejs
        return self.entropy_series
    def calc_entropy_redundancy(self):
        try:
            self.d_series = 1 - self.entropy_series # 1-熵值为效用值
            self.d_series.name = '信息效用值'
        except AttributeError:
            raise Exception('需调用calc_entropy方法')

        return self.d_series
    def calc_Weight(self):
        self.uniform()
        self.calc_probability()
        self.calc_entropy()
        self.calc_entropy_redundancy() # 走一遍上面的流程
        self.Weight = self.d_series / sum(self.d_series) # 将信息效用值分配成权重
        self.Weight.name = '权值'
        return self.Weight