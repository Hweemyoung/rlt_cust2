import os
import locale
import logging
import numpy as np
import pandas as pd
import settings
from environment import Environment
from agent_cust0 import Agent
from policy_network_cust0 import PolicyNetwork
from visualizer_cust0 import Visualizer


logger = logging.getLogger(__name__)
locale.setlocale(locale.LC_ALL, 'ko_KR.UTF-8')


class PolicyLearner:

    def __init__(self, stock_code, chart_data, training_data=None,
                 min_trading_unit=1, max_trading_unit=2,
                 delayed_reward_threshold=.05, lr=0.01, sub_dir='',
                 str_num_agent='', dropout=0.5, batch_ver=0, TU_ver=0, margin=0.1,
                 date_run=None, date_end=None):
        self.stock_code = stock_code  # 종목코드
        self.sub_dir = sub_dir
        self.batch_ver = batch_ver
        self.str_num_agent = str_num_agent
        self.date_run = date_run
        self.date_end = date_end
        self.chart_data = chart_data
        self.environment = Environment(chart_data)  # 환경 객체
        # 에이전트 객체
        self.agent = Agent(self.environment,
                           min_trading_unit=min_trading_unit,
                           max_trading_unit=max_trading_unit,
                           delayed_reward_threshold=delayed_reward_threshold,
                           margin=margin, TU_ver=TU_ver)
        self.training_data = training_data  # 학습 데이터
        self.sample = None
        self.training_data_idx = -1
        # 정책 신경망; 입력 크기 = 학습 데이터의 크기 + 에이전트 상태 크기
        self.num_features = self.training_data.shape[1] + self.agent.STATE_DIM
        self.policy_network = PolicyNetwork(
            input_dim=self.num_features, output_dim=self.agent.output_dim, lr=lr, dropout=dropout)
        self.visualizer = Visualizer()  # 가시화 모듈

    def reset(self):
        self.sample = None
        self.training_data_idx = -1

    def fit(
        self, num_epoches=1000, max_memory=60, balance=10000000,
        discount_factor=0, start_epsilon=.5, learning=True, batch_ver=0,
        windows=None, project_num=None, history=0, model_path=None):
        dt_run = settings.get_time_datetime()
        logger.info("V: {batch_ver}, "
                    "LR: {lr}, DF: {discount_factor}, "
                    "DRT: {delayed_reward_threshold}, "
                    "MEMORY: {max_memory}, "
                    "AGENT#: {str_num_agent}, "
                    "win: {windows}, "
                    "sub_dir: {sub_dir}".format(
            batch_ver=batch_ver,
            lr=self.policy_network.lr,
            discount_factor=discount_factor,
            delayed_reward_threshold=self.agent.delayed_reward_threshold,
            max_memory=max_memory,
            str_num_agent=self.str_num_agent,
            windows=windows,
            sub_dir=self.sub_dir
        ))

        # 가시화 준비
        # 차트 데이터는 변하지 않으므로 미리 가시화
        self.visualizer.prepare(self.environment.chart_data)

        # 가시화 결과 저장할 폴더 준비
        epoch_summary_dir = os.path.join(
            settings.BASE_DIR, self.sub_dir)

        # 에이전트 초기 자본금 설정
        self.agent.set_balance(balance)

        # 학습에 대한 정보 초기화
        max_portfolio_value = 0
        epoch_win_cnt = 0
        
        # 학습 반복
        for epoch in range(num_epoches):
            # 학습 재개의 경우, 과거 학습 이력은 제외
            if epoch < history:
                None
            else:
                # 에포크 관련 정보 초기화
                loss = 0.
                itr_cnt = 0
                win_cnt = 0
                exploration_cnt = 0
                batch_size = 0
                pos_learning_cnt = 0
                neg_learning_cnt = 0
        
                # 메모리 초기화
                memory_sample = []
                memory_action = []
                memory_reward = []
                memory_prob = []
                memory_TU = []
                memory_states = []
                memory_pv = []
                memory_num_stocks = []
                memory_exp_idx = []
                memory_learning_idx = []
                
                # 환경, 에이전트, 정책 신경망 초기화
                self.environment.reset()
                self.agent.reset()
                self.policy_network.reset()
                self.reset()
        
                # 가시화 초기화
                self.visualizer.clear([0, len(self.chart_data)])
        
                # 학습을 진행할 수록 탐험 비율 감소
                if learning:
                    epsilon = start_epsilon * (1. - float(epoch) / (num_epoches - 1))
                else:
                    epsilon = 0
        
                while True:
                    # 샘플 생성
                    next_sample, states = self._build_sample()
                    if next_sample is None:
                        break
        
                    # 정책 신경망 또는 탐험에 의한 행동 결정
                    # 한 idx에 있어서 샘플 -> 행동 -> 보상 순으로 진행
                    probs, exploration = self.agent.decide_action(
                        self.policy_network, self.sample, epsilon)
                    
                    # 결정한 행동을 수행하고 즉시 보상과 지연 보상 획득
                    curr_pv = memory_pv[len(memory_pv) - 1] if len(memory_pv) > 0 else self.agent.portfolio_value
                    prev_pv = memory_pv[len(memory_pv) - 2] if len(memory_pv) > 1 else self.agent.portfolio_value
                    
                    try:
                        immediate_reward, delayed_reward, action, min_trading_unit, max_trading_unit, trading_unit,\
                        curr_pv, prev_pv, pv_temp, p_inv_temp = self.agent.act(probs, curr_pv, prev_pv)
                    except ValueError:
                        pd.DataFrame(memory_TU, columns = ['min_trading_unit', 'max_trading_unit', 'trading_unit',\
                                                           'curr_pv', 'prev_pv', 'pv_temp', 'p_inv_temp', 'agent.balance']).to_csv(os.path.join(settings.BASE_DIR, self.sub_dir, 'memory_TU.csv'), index=False)
                        print(self.sub_dir)
                    
                    # 행동 및 행동에 대한 결과를 기억
                    memory_sample.append(next_sample)
                    memory_action.append(action)
                    memory_reward.append(immediate_reward)
                    memory_pv.append(self.agent.portfolio_value)
                    memory_TU.append((min_trading_unit, max_trading_unit, trading_unit, curr_pv, prev_pv, pv_temp, p_inv_temp, self.agent.balance))
                    memory_states.append(states)
                    memory_num_stocks.append(self.agent.num_stocks)
                    memory = [(
                        memory_sample[i],
                        memory_action[i],
                        memory_reward[i])
                        for i in list(range(len(memory_action)))[-max_memory:]
                    ]
                    if exploration:
                        memory_exp_idx.append(itr_cnt)
                        memory_prob.append([np.nan])
                    else:
                        memory_prob.append(probs)
        
                    # 반복에 대한 정보 갱신
                    batch_size += 1
                    itr_cnt += 1
                    exploration_cnt += 1 if exploration else 0
                    win_cnt += 1 if delayed_reward > 0 else 0
        
                    # 학습 모드이고 지연 보상이 존재할 경우 정책 신경망 갱신
                    if delayed_reward == 0 and batch_size >= max_memory:
                        delayed_reward = immediate_reward
                        self.agent.base_portfolio_value = self.agent.portfolio_value
                    if learning and delayed_reward != 0:
                        # 배치 학습 데이터 크기
                        batch_size = min(batch_size, max_memory)
                        # 배치 학습 데이터 생성
                        x, y = self._get_batch(
                            memory, batch_size, discount_factor, delayed_reward, batch_ver)
                        if len(x) > 0:
                            if delayed_reward > 0:
                                pos_learning_cnt += 1
                            else:
                                neg_learning_cnt += 1
                            # 정책 신경망 갱신
                            loss += self.policy_network.train_on_batch(x, y)
                            memory_learning_idx.append([itr_cnt, delayed_reward])
                        batch_size = 0
        
                # 학습 시 매 에포크 종료마다 가중치 저장
                if learning:
                    prev_model_path = model_path
                    model_path = os.path.join(epoch_summary_dir, 'model_e%d_%s.h5' % (epoch + 1, self.str_num_agent))
                    if not epoch == 0:
                        os.remove(prev_model_path)
                    self.policy_network.save_model(model_path)
                
                # 에포크 관련 정보 가시화
                num_epoches_digit = len(str(num_epoches))
                epoch_str = str(epoch + 1).rjust(num_epoches_digit, '0')
                
                # infr, trading_unit도 출력하도록 해보자.
                self.visualizer.plot(
                    epoch_str=epoch_str, num_epoches=num_epoches, epsilon=epsilon,
                    action_list=Agent.ACTIONS, actions=memory_action,
                    num_stocks=memory_num_stocks, outvals=memory_prob,
                    exps=memory_exp_idx, learning=memory_learning_idx,
                    initial_balance=self.agent.initial_balance, pvs=memory_pv, sub_dir=self.sub_dir
                )
                self.visualizer.save(os.path.join(
                    epoch_summary_dir, 'summary_%s_%s.png' % (
                        self.str_num_agent, epoch_str)))
        
                # 에포크 관련 정보 로그 기록
                if pos_learning_cnt + neg_learning_cnt > 0:
                    loss /= pos_learning_cnt + neg_learning_cnt
                logger.info("[Epoch %s/%s]\tEpsilon:%.4f\t#Expl.:%d/%d\t"
                            "#Buy:%d\t#Sell:%d\t#Hold:%d\t"
                            "#Stocks:%d\tPV:%s\t"
                            "POS:%s\tNEG:%s\tLoss:%10.6f" % (
                                epoch_str, num_epoches, epsilon, exploration_cnt, itr_cnt,
                                self.agent.num_buy, self.agent.num_sell, self.agent.num_hold,
                                self.agent.num_stocks,
                                locale.currency(self.agent.portfolio_value, grouping=True),
                                pos_learning_cnt, neg_learning_cnt, loss))
        
                # 학습 관련 정보 갱신
                max_portfolio_value = max(
                max_portfolio_value, self.agent.portfolio_value)
                if self.agent.portfolio_value > self.agent.initial_balance:
                    epoch_win_cnt += 1

        #평균 보유 비율 계산(기하)
        mean_arith, mean_geo = self.calc_mean(np.array(memory_states))
        
        #첫/마지막 종가 확인
        price_run = self.chart_data.iloc[0][self.environment.PRICE_IDX]
        price_end = self.chart_data.iloc[-1][self.environment.PRICE_IDX]

        #마지막 에포크 PV
        last_pv = self.agent.portfolio_value

        profit_model = round(last_pv / balance , 2)
        profit_chart = round(price_end / price_run , 2)
        
        # 학습 관련 정보 로그 기록
        dt_end = settings.get_time_datetime()
        logger.info("PV: %s, \t Max PV: %s, \t # Win: %d, \t Time: %s \n PV profit: %s \t Chart profit: %s \t position_ratio: %s / %s" 
                    % (locale.currency(last_pv, grouping=True),
                       locale.currency(max_portfolio_value, grouping=True), epoch_win_cnt,
                       dt_end - dt_run, profit_model, profit_chart, mean_arith, mean_geo))
        
        #기록표에 기록: 00은 학습, 01은 비학습
        result_fname = 'result04' if learning else 'result05'
        result_path = os.path.join(settings.BASE_DIR, 'project_%s' % project_num,'%s' % self.stock_code,'{}.csv'.format(result_fname))
        result_data = [project_num, self.str_num_agent, self.stock_code, self.date_run, self.date_end, balance,
                       mean_arith, mean_geo, batch_ver, self.policy_network.lr, discount_factor,
                       self.agent.delayed_reward_threshold, max_memory, self.agent.margin, windows,
                       self.policy_network.dropout, num_epoches, profit_model,
                       profit_chart]

        if not os.path.exists(result_path):
            pd.DataFrame(result_data).to_csv(result_path, index=False)
    
        else:
            result_table = pd.read_csv(result_path)
            case_num = str(len(result_table.columns) + 1)
            result_table[case_num] = result_data
            result_table.to_csv(result_path, index=False)

    def _get_batch(self, memory, batch_size, discount_factor, delayed_reward, batch_ver):
        if int(batch_ver) == 0:
            x = np.zeros((batch_size, 1, self.num_features))
            y = np.full((batch_size, self.agent.output_dim), 0.5)

            for i, (sample, action, reward) in enumerate(
                    reversed(memory[-batch_size:])):
                x[i] = np.array(sample).reshape((-1, 1, self.num_features))
                y[i, 0] = (delayed_reward + 1) / 2
                if discount_factor > 0:
                    y[i, action] *= discount_factor ** i
            return x, y
# =============================================================================
#         elif int(batch_ver) == 3:
#             # x = [((2,3,..6)),((1,6,....,4)),...,((7,4,...,5))]
#             x = np.zeros((batch_size, 1, self.num_features))
#             y = np.full((batch_size, self.agent.output_dim), 0.5)
# 
#             for i, (sample, action, reward) in enumerate(
#                     reversed(memory[-batch_size:])):
#                 # 과거 데이터일수록 감가율 거듭제곱 -> enumerate(reversed.list)로 편하게
#                 x[i] = np.array(sample).reshape((-1, 1, self.num_features))
#                 # ex)현재로부터 3샘플 전 데이터 i에 매수 + 지연보상 있음 -> y[i] = [1.0*감가율^3 , 0.5 , 0.5]
#                 # action = 매도 -> [.5,1] or [.5,0]
#                 y[i, 0] = (delayed_reward + 1) / 2
#                 if discount_factor > 0:
#                     y[i] = (y[i] - 0.5) * discount_factor ** (i) + 0.5
#             return x, y
# 
# =============================================================================
        
        #ver1: 2의 레이블을 뒤집기.
        elif batch_ver == 1:
            x = np.zeros((batch_size, 1, self.num_features))
            y = np.zeros((batch_size, self.agent.NUM_ACTIONS))

            #y[0] 설정: 당일 긍정 보상 시 마지막 매수, 부정 보상 시 마지막 매도하는 정책.
            if delayed_reward > 0:
                y[0 , self.agent.ACTION_SELL] = 1.0
            else:
                y[0 , self.agent.ACTION_BUY] = 1.0
            #x[batch_size - 1] 설정
            x[batch_size - 1] = np.array(memory[-batch_size][0]).reshape((-1, 1, self.num_features))
            #batch_size=1일 경우는 제외해야 한다.(최대 인덱스 0)
            if batch_size != 1:
                for i, (sample, action, reward) in enumerate(reversed(memory[-batch_size + 1 :])):
                    x[i] = np.array(sample).reshape((-1, 1, self.num_features))
                    #sample은 1차원 리스트, 트레이딩 데이터의 close/lastclose 인덱스는 3.
                    #다음날의 close/lastclose로 당일의 ACTION을 피드백.
                    if x[i][0][3] > self.agent.TRADING_CHARGE + self.agent.TRADING_TAX:
                        y[i+1, self.agent.ACTION_SELL] = .5
                    else:
                        y[i+1, self.agent.ACTION_BUY] = .5
                    if discount_factor > 0:
                        y[i+1] *= discount_factor ** (i+1)
            return x, y

        elif batch_ver == 1.1:
            x = np.zeros((batch_size, 1, self.num_features))
            y = np.zeros((batch_size, self.agent.NUM_ACTIONS))

            #y[0] 설정: 당일 긍정 보상 시 마지막 매수, 부정 보상 시 마지막 매도하는 정책.
            if delayed_reward > 0:
                y[0 , self.agent.ACTION_SELL] = 1.0
            else:
                y[0 , self.agent.ACTION_BUY] = 1.0
            #x[batch_size - 1] 설정
            x[batch_size - 1] = np.array(memory[-batch_size][0]).reshape((-1, 1, self.num_features))
            #batch_size=1일 경우는 제외해야 한다.(최대 인덱스 0)
            if batch_size != 1:
                for i, (sample, action, reward) in enumerate(reversed(memory[-batch_size + 1 :])):
                    x[i] = np.array(sample).reshape((-1, 1, self.num_features))
                    #sample은 1차원 리스트, 트레이딩 데이터의 close/lastclose 인덱스는 3.
                    #다음날의 close/lastclose로 당일의 ACTION을 피드백.
                    #ver1.1: 긍정 보상의 경우 레이블의 매수확률 1.0을 0.5로 만들어버리지 않도록 조건 추가
                    if x[i][0][3] > self.agent.TRADING_CHARGE + self.agent.TRADING_TAX and not delayed_reward > 0:
                        y[i+1, self.agent.ACTION_SELL] = .5
                    elif x[i][0][3] <= self.agent.TRADING_CHARGE + self.agent.TRADING_TAX and delayed_reward > 0:
                        y[i+1, self.agent.ACTION_BUY] = .5
                    if discount_factor > 0:
                        y[i+1] = (y[i+1] - 0.5) * discount_factor ** (i+1) + 0.5
            return x, y
        
        elif batch_ver == 2:
            x = np.zeros((batch_size, 1, self.num_features))
            y = np.zeros((batch_size, 0))

            #y[0] 설정: 당일 긍정 보상 시 마지막 매수, 부정 보상 시 마지막 매도하는 정책.
            if delayed_reward > 0:
                y[0 , 0] = 1.0
            else:
                y[0 , 0] = .0
            #x[batch_size - 1] 설정
            x[batch_size - 1] = np.array(memory[-batch_size][0]).reshape((-1, 1, self.num_features))
            #batch_size=1일 경우는 제외해야 한다.(최대 인덱스 0)
            if batch_size != 1:
                for i, (sample, action, reward) in enumerate(reversed(memory[-batch_size + 1 :])):
                    x[i] = np.array(sample).reshape((-1, 1, self.num_features))
                    #sample은 1차원 리스트, 트레이딩 데이터의 close/lastclose 인덱스는 3.
                    #다음날의 close/lastclose로 당일의 ACTION을 피드백.
                    if x[i][0][3] > self.agent.TRADING_CHARGE + self.agent.TRADING_TAX:
                        y[i+1, self.agent.ACTION_BUY] = .5
                    else:
                        y[i+1, self.agent.ACTION_SELL] = .5
                    if discount_factor > 0:
                        y[i+1] = (y[i+1] - 0.5) * discount_factor ** (i+1) + 0.5
            return x, y

        elif batch_ver == 2.1:
            x = np.zeros((batch_size, 1, self.num_features))
            y = np.zeros((batch_size, 0))

            #y[0] 설정: 당일 긍정 보상 시 마지막 매수, 부정 보상 시 마지막 매도하는 정책.
            if delayed_reward > 0:
                y[0 , 0] = 1.0
            else:
                y[0 , 0] = .0
            #x[batch_size - 1] 설정
            x[batch_size - 1] = np.array(memory[-batch_size][0]).reshape((-1, 1, self.num_features))
            #batch_size=1일 경우는 제외해야 한다.(최대 인덱스 0)
            if batch_size != 1:
                for i, (sample, action, reward) in enumerate(reversed(memory[-batch_size + 1 :])):
                    x[i] = np.array(sample).reshape((-1, 1, self.num_features))
                    #sample은 1차원 리스트, 트레이딩 데이터의 close/lastclose 인덱스는 3.
                    #다음날의 close/lastclose로 당일의 ACTION을 피드백.
                    #ver2.1: 긍정 보상의 경우 레이블의 매수확률 1.0을 0.5로 만들어버리지 않도록 조건 추가
                    if x[i][0][3] > self.agent.TRADING_CHARGE + self.agent.TRADING_TAX and not delayed_reward > 0:
                        y[i+1, 0] = 1.0
                    elif x[i][0][3] <= self.agent.TRADING_CHARGE + self.agent.TRADING_TAX and delayed_reward > 0:
                        y[i+1, 0] = .0
                    if discount_factor > 0:
                        y[i+1] = (y[i+1] - 0.5) * discount_factor ** (i+1) + 0.5
            return x, y

        #ver2.2: 그냥 시차 없이 단순 [1,0],[0,1]로만 구성 후 0.5 적용
        elif batch_ver == 2.2:
            x = np.zeros((batch_size, 1, self.num_features))
            y = np.zeros((batch_size, 1))

            #y[0] 설정: 당일 긍정 보상 시 마지막 매수, 부정 보상 시 마지막 매도하는 정책.
            if delayed_reward > 0:
                for i in range(batch_size):
                    y[i , 0] = 1.0
# =============================================================================
#           # 이미 충족되어있기 때문에 필요없다
#             else:
#                 for i in range(batch_size):
#                     y[i , 0] = .0
# =============================================================================
            for i, (sample, action, reward) in enumerate(reversed(memory[-batch_size :])):
                x[i] = np.array(sample).reshape((-1, 1, self.num_features))
                #sample은 1차원 리스트, 트레이딩 데이터의 close/lastclose 인덱스는 3.
                #다음날의 close/lastclose로 당일의 ACTION을 피드백.
                #ver2.1: 긍정 보상의 경우 레이블의 매수확률 1.0을 0.5로 만들어버리지 않도록 조건 추가
                if x[i][0][3] > self.agent.TRADING_CHARGE + self.agent.TRADING_TAX and not delayed_reward > 0:
                    y[i, 0] = .25
                elif x[i][0][3] <= self.agent.TRADING_CHARGE + self.agent.TRADING_TAX and delayed_reward > 0:
                    y[i, 0] = .75
                if discount_factor > 0:
                    y[i] = (y[i] - 0.5) * discount_factor ** (i) + 0.5
            return x, y
        
        elif batch_ver == 2.3:
            x = np.zeros((batch_size, 1, self.num_features))
            y = np.zeros((batch_size, self.agent.NUM_ACTIONS))

            #y[0] 설정: 당일 긍정 보상 시 마지막 매수, 부정 보상 시 마지막 매도하는 정책.
            if delayed_reward > 0:
                for i in range(batch_size):
                    y[i , self.agent.ACTION_BUY] = 1.0
            else:
                for i in range(batch_size):
                    y[i , self.agent.ACTION_SELL] = 1.0
            #x[batch_size - 1] 설정
            x[batch_size - 1] = np.array(memory[-batch_size][0]).reshape((-1, 1, self.num_features))
            #batch_size=1일 경우는 제외해야 한다.(최대 인덱스 0)
            if batch_size != 1:
                for i, (sample, action, reward) in enumerate(reversed(memory[-batch_size + 1 :])):
                    x[i] = np.array(sample).reshape((-1, 1, self.num_features))
                    #sample은 1차원 리스트, 트레이딩 데이터의 close/lastclose 인덱스는 3.
                    #다음날의 close/lastclose로 당일의 ACTION을 피드백.
                    #ver2.1: 긍정 보상의 경우 레이블의 매수확률 1.0을 0.5로 만들어버리지 않도록 조건 추가
                    if x[i][0][3] > self.agent.TRADING_CHARGE + self.agent.TRADING_TAX and not delayed_reward > 0:
                        y[i+1, self.agent.ACTION_BUY] = .5
                    elif x[i][0][3] <= self.agent.TRADING_CHARGE + self.agent.TRADING_TAX and delayed_reward > 0:
                        y[i+1, self.agent.ACTION_SELL] = .5
                    if discount_factor > 0:
                        y[i+1] = (y[i+1] - 0.5) * discount_factor ** (i+1) + 0.5
            return x, y
        
        elif int(batch_ver) == 3:
            # x = [((2,3,..6)),((1,6,....,4)),...,((7,4,...,5))]
            x = np.zeros((batch_size, 1, self.num_features))
            y = np.full((batch_size, self.agent.output_dim), 0.5)

            for i, (sample, action, reward) in enumerate(
                    reversed(memory[-batch_size:])):
                # 과거 데이터일수록 감가율 거듭제곱 -> enumerate(reversed.list)로 편하게
                x[i] = np.array(sample).reshape((-1, 1, self.num_features))
                # ex)현재로부터 3샘플 전 데이터 i에 매수 + 지연보상 있음 -> y[i] = [1.0*감가율^3 , 0.5 , 0.5]
                # action = 매도 -> [.5,1] or [.5,0]
                y[i, 0] = (delayed_reward + 1) / 2
                if discount_factor > 0:
                    y[i] = (y[i] - 0.5) * discount_factor ** (i) + 0.5
            return x, y
        
    def _build_sample(self):
        #environment.observe() : 차트데이터에서 선 idx +1 후 참조
        self.environment.observe()
        if len(self.training_data) > self.training_data_idx + 1:
            #트레이닝 데이터에서도 선 +1 후 참조 -> 둘은 동시
            self.training_data_idx += 1
            self.sample = self.training_data.iloc[self.training_data_idx].tolist()
            states = self.agent.get_states()
            self.sample.append(states)
            return self.sample, states
        return None, None
    
    def calc_mean(self, table):
        mean_arith = table.mean()
        mean_geo = 1
        for i in table:
            if i == 0:
                i = 0.0001
            mean_geo *= i
        mean_geo = mean_geo ** ( 1 / len(table))
        return round(mean_arith, 2), round(mean_geo, 2)
    
    def _get_p_dev(self, p_dev_data, training_data_idx):
        p_dev = p_dev_data.iloc[training_data_idx].tolist()
        p_dev = p_dev[2]
        return p_dev

    def resume(self, sub_dir=None, num_epoches=1000, max_memory=60, balance=10000000,
        discount_factor=0, start_epsilon=.5, learning=True, batch_ver=0,
        windows=[], project_num=''):
        model_dir = os.path.join(settings.BASE_DIR, sub_dir)
        if not os.path.exists(model_dir):
            return
        for item in os.listdir(model_dir):
            if item.find('model') is not -1:
                break
        model_path = os.path.join(model_dir, item)
        self.policy_network.load_model(model_path=model_path)
        history = int(item.split('_')[1].split('e')[1])
        self.fit(num_epoches, max_memory=max_memory, balance=balance,
                 discount_factor=discount_factor, start_epsilon=start_epsilon, learning=True,
                 batch_ver=batch_ver, windows=windows, project_num=project_num, history=history,
                 model_path=model_path)

    def trade(self, model_path=None, num_epoches=1, max_memory=60, balance=10000000,
                 discount_factor=0, batch_ver=0, windows=[], project_num=''):
        if model_path is None:
            return
        self.policy_network.load_model(model_path=model_path)
        self.fit(num_epoches=num_epoches, max_memory=max_memory, balance=balance,
                 discount_factor=discount_factor, start_epsilon=0, learning=False, batch_ver=batch_ver,
                 windows=windows, project_num=project_num)
