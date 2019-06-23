class Environment:

    PRICE_IDX = 4  # 종가의 위치

    def __init__(self, chart_data=None):
        self.chart_data = chart_data
        self.observation = None
        self.observation_prev = None
        self.idx = -1

    def reset(self):
        self.observation = None
        self.idx = -1

    def observe(self):
        if len(self.chart_data) > self.idx + 1:
            self.idx += 1
            self.observation = self.chart_data.iloc[self.idx]
            self.observation_prev = self.chart_data.iloc[self.idx - 1] if self.idx >=1 else self.observation
            return self.observation, self.observation_prev
        return None

    def get_price(self):
        if self.observation is not None:
            return self.observation[self.PRICE_IDX]
        return None
    
    def get_prev_price(self):
        return self.observation_prev[self.PRICE_IDX]

    def set_chart_data(self, chart_data):
        self.chart_data = chart_data
