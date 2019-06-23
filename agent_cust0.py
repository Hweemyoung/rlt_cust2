# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 14:13:49 2019

@author: hweem
"""

import numpy as np


class Agent:
    # 에이전트 상태가 구성하는 값 개수
    STATE_DIM = 1  # 주식 보유 비율

    # 매매 수수료 및 세금
    TRADING_CHARGE = 0  # 거래 수수료 미고려 (일반적으로 0.015%)
    TRADING_TAX = 0  # 거래세 미고려 (실제 0.3%)

    # 행동
    probs = .0
    output_dim = 1
    ACTION_BUY = 0  # 매수
    ACTION_SELL = 1  # 매도
    ACTION_HOLD = 2  # 홀딩
    ACTIONS = [ACTION_BUY, ACTION_SELL, ACTION_HOLD]
    NUM_ACTIONS = len(ACTIONS)

    def __init__(
        self, environment, min_trading_unit=1, max_trading_unit=2, 
        delayed_reward_threshold=.05, margin=.01, TU_ver=0):
        # Environment 객체
        self.environment = environment  # 현재 주식 가격을 가져오기 위해 환경 참조

        # 지연보상 임계치
        self.delayed_reward_threshold = delayed_reward_threshold  # 지연보상 임계치
        self.margin = margin
        self.TU_ver = TU_ver

        # Agent 클래스의 속성
        self.initial_balance = 0  # 초기 자본금
        self.balance = 0  # 현재 현금 잔고
        self.prev_balance = 0 # 이전 현금 잔고
        self.num_stocks = 0  # 보유 주식 수
        self.portfolio_value = 0  # balance + num_stocks * {현재 주식 가격}
        self.base_portfolio_value = 0  # 직전 학습 시점의 PV
        self.num_buy = 0  # 매수 횟수
        self.num_sell = 0  # 매도 횟수
        self.num_hold = 0  # 홀딩 횟수
        self.immediate_reward = 0  # 즉시 보상

        # Agent 클래스의 상태
        self.ratio_hold = 0  # 주식 보유 비율
        self.ratio_portfolio_value = 0  # 포트폴리오 가치 비율

    def reset(self):
        self.balance = self.initial_balance
        self.num_stocks = 0
        self.portfolio_value = self.initial_balance
        self.base_portfolio_value = self.initial_balance
        self.num_buy = 0
        self.num_sell = 0
        self.num_hold = 0
        self.immediate_reward = 0
        self.ratio_hold = 0
        self.ratio_portfolio_value = 0

    def set_balance(self, balance):
        self.initial_balance = balance
        self.prev_balance = balance

    def get_states(self):
        self.ratio_position = self.num_stocks / int(
            self.portfolio_value / self.environment.get_price())
        return self.ratio_position
        

    def decide_action(self, policy_network, sample, epsilon):
        # 탐험 결정
        if np.random.rand() < epsilon:
            exploration = True
            probs = np.random.rand()  # 무작위 수 0~1
        else:
            exploration = False
            probs = policy_network.predict(sample)  # 각 행동에 대한 확률
        return probs, exploration

    def validate_action(self, trading_unit):
        validity = True
        if trading_unit > 0 :
            # 적어도 1주를 살 수 있는지 확인 + 잔고는 0이 되지 않도록
            if self.balance <= self.environment.get_price() * (
                1 + self.TRADING_CHARGE):
                validity = False
        elif trading_unit < 0 :
            # 주식 잔고가 있는지 확인 
            if self.num_stocks <= 0:
                validity = False
        return validity

    def decide_trading_unit(self, TU_ver, probs, margin, curr_price, prev_price, curr_pv, prev_pv,
                            curr_balance, prev_balance, num_stocks):
        if TU_ver == 1:
            trading_unit = int(num_stocks*(probs - 0.5))
            return trading_unit, trading_unit, trading_unit, trading_unit, trading_unit
        
        elif TU_ver == 2:
            # p.1
            pv_temp = curr_pv - curr_balance ** 2 / prev_balance
            p_inv_temp =  prev_price / curr_price ** 2
            try:
                min_trading_unit = (-margin * curr_pv ** 2 / prev_pv + pv_temp) * p_inv_temp - num_stocks
                max_trading_unit = (margin * curr_pv ** 2 / prev_pv + pv_temp) * p_inv_temp - num_stocks
                trading_unit = int((1 - probs) * min_trading_unit + probs * max_trading_unit)
            except OverflowError:
                print(pv_temp, p_inv_temp, probs[0], curr_pv, curr_balance, prev_balance, curr_price, prev_price)
            #trading_unit 은 음의 정수가 될 수 있다.
            return int(min_trading_unit), int(max_trading_unit), trading_unit, pv_temp, p_inv_temp

    def act(self, probs, curr_pv, prev_pv):
        curr_price = self.environment.get_price()
        prev_price = self.environment.get_prev_price()
        if self.TU_ver == 1:
            min_trading_unit, max_trading_unit, trading_unit, pv_temp, p_inv_temp \
            = self.decide_trading_unit(probs, self.TU_ver, self.margin, curr_price, prev_price, curr_pv, prev_pv,
                                self.balance, self.prev_balance, self.num_stocks)

        
        elif self.TU_ver == 2:
            min_trading_unit, max_trading_unit, trading_unit, pv_temp, p_inv_temp \
            = self.decide_trading_unit(probs, self.TU_ver, self.margin, curr_price, prev_price, curr_pv, prev_pv,
                                self.balance, self.prev_balance, self.num_stocks)
        
        # 지금 self.balance는 다음 trading unit에서 prev_balance
        self.prev_balance = self.balance
        
        # 관망
        if not self.validate_action(trading_unit) or trading_unit == 0 :
            action = Agent.ACTION_HOLD
            self.num_hold += 1  # 홀딩 횟수 증가
            # 즉시 보상 초기화
            self.immediate_reward = 0

        # 매수
        elif trading_unit > 0 :
            # 매수할 단위를 판단
            balance = self.balance - curr_price * (1 + self.TRADING_CHARGE) * trading_unit
            # 보유 현금이 모자랄 경우 보유 현금으로 가능한 만큼 최대한 매수
            if balance < 0:
                trading_unit = min(
                    int(self.balance / (
                        curr_price * (1 + self.TRADING_CHARGE))), trading_unit)
                    
            # 수수료를 적용하여 총 매수 금액 산정
            invest_amount = curr_price * (1 + self.TRADING_CHARGE) * trading_unit
            self.balance -= invest_amount  # 보유 현금을 갱신
            self.num_stocks += trading_unit  # 보유 주식 수를 갱신
            self.num_buy += 1  # 매수 횟수 증가
            action = Agent.ACTION_BUY

        # 매도
        elif trading_unit < 0 :
            # 보유 주식이 모자랄 경우 가능한 만큼 최대한 매도
            trading_unit = max(trading_unit, -self.num_stocks)
            # 매도
            invest_amount = curr_price * (
                1 - (self.TRADING_TAX + self.TRADING_CHARGE)) * -trading_unit
            self.num_stocks += trading_unit  # 보유 주식 수를 갱신
            self.balance += invest_amount  # 보유 현금을 갱신
            self.num_sell += 1  # 매도 횟수 증가
            action = Agent.ACTION_SELL

        # 포트폴리오 가치 갱신
        self.portfolio_value = self.balance + curr_price * self.num_stocks
        profitloss = (
            (self.portfolio_value - self.base_portfolio_value) / self.base_portfolio_value)

        # 즉시 보상 판단
        self.immediate_reward = 1 if profitloss >= 0 else -1

        # 지연 보상 판단
        if profitloss > self.delayed_reward_threshold:
            delayed_reward = 1
            # 목표 수익률 달성하여 기준 포트폴리오 가치 갱신
            self.base_portfolio_value = self.portfolio_value
        elif profitloss < -self.delayed_reward_threshold:
            delayed_reward = -1
            # 손실 기준치를 초과하여 기준 포트폴리오 가치 갱신
            self.base_portfolio_value = self.portfolio_value
        else:
            delayed_reward = 0
        return self.immediate_reward, delayed_reward, action, min_trading_unit, max_trading_unit, trading_unit, curr_pv, prev_pv, pv_temp, p_inv_temp
