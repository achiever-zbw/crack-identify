class EarlyStopping:
    """早停类，防止过拟合，返回最佳模型
    Attributes:
        patience (int): 容忍的epoch数,即验证集的准确率在多少个epoch内没有提升时才停止训练。
        delta (float): 最小变化值，只有验证集的准确率提高超过此值才视为有效的进展。
        counter (int): 计数器,记录验证集准确率未提高的epoch数。
        best_score (float): 记录当前最佳的验证集准确率。
        early_stop (bool): 标志，指示是否提前停止训练。
        best_model (dict): 保存最好的模型参数。
    """

    def __init__(self, patience, delta):
        """初始化类对象

        Args:
            patience (int, optional): 容忍的epoch数,即验证集的准确率没有提高的最大epoch数. Defaults to 5.
            delta (int, optional): 最小变化值，验证集的准确率变化小于这个值时，认为没有变化了. Defaults to 0.
        """
        self.patience = patience  # 容忍的epoch数
        self.delta = delta  # 最小的变化，超过这个才视为有进展
        self.counter = 0  # 计数器，跟踪验证集中准确率没有提高的epoch
        self.best_score = None  # 最好的验证集准确率
        self.early_stop = False  # 是否停止训练
        self.best_model = None

    def __call__(self, val_score, model):
        """根据验证集的准确率决定是否停止训练，并保存最好的模型

        Args:
            val_score (float): 当前epoch的验证集准确率。
            model (nn.Module): 当前训练的模型对象
        """
        # 如果没有找到最好的模型，直接保存当前的模型
        if self.best_model is None:
            self.best_score = val_score
            self.best_model = model.state_dict()
            return
        # 验证集准确率提高，则更新最好的模型
        if val_score > self.best_score+self.delta:
            self.best_score = val_score
            self.best_model = model.state_dict()
            self.counter = 0
        else:
            self.counter += 1
        # 不更新了，没耐心了，停！
        if self.counter >= self.patience:
            self.early_stop = True
