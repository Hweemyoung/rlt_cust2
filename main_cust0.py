import logging
import os
import settings
from data_manager_cust0 import DataManager
from policy_learner_cust0 import PolicyLearner
# =============================================================================
# ver.0.10
# Updated Last on: Mar 03 2019
# 활성화함수로 sortmax 사용
# 속성 sub_dir, 에이전트 넘버 도입, timestring 제거
# close/last close 를 이용해서 정답레이블 작성방식을 수정 ->policy_learner.py ->?positive_win_cnt, negative_win_cnt를 위해서는 Reward 1이나 -1이 필요. 그렇지않다면 그냥 T,F로 하면 됨.
# 보상 발생 시 당일 거래는 어떻게 하는가? -> 긍정시 매수, 부정시 매도 ver
# =============================================================================

# =============================================================================
# 프로젝트
#0
#1
#2
#3
#4 기관+외국인 거래량 지표 추가 (이격률, 전일 대비). 이격률의 이평선은 절대값평균.
#4.1 (w/ infr)dnn을 실험해보자. Agent01:dnn -> 망;;
#4.2 (w/ infr)dnn + batch_ver=2.2
#5 network의 layer를 늘려보자! 4 5 6 7

#10 가변TU사용, min max_trading_unit을 생성자 인자로만 받고 속성은 만들지 않음.
#11 DF
#12 node 변화: 128,384,512
#13 layer 증가: 4,5,6
#14 ver2.2
#15 ver2.2수정, 마진 .01 .005 시
#16 보유 주식 비율, short 특징량 추가
#17 infr의 abs를 제거 -> 에러 때문에 불가
#18 보유 주식 비율 가시화 및 평균 보유 비율 계산해보자!
#19 DRT(->주가와 보유비율 변동?)와 MEM(DRT에 비례?)의 최적화를 시도해보자!
#20 margin 높임 ->pv변동 큼->학습 촉진 시
#21 DF 실험
#24 short_lastshort_ratio 제거
#25 TIGER200금융(139270)
#['055550','105560','086790','000810','032830']
#30 TIGER방통(098560)
#['017670','032640','030200']
# =============================================================================
# batch_ver
# 3: training_data 에서 agent상태 (주식 보유 비율, pv 비율) 제


# =============================================================================
# cust1: infr 포함
# =============================================================================

# 학습/학습재개/비학습 설정
mode = 0
if __name__ == '__main__':
    project_num = '25'
    stock_code = '139270' #TIGER200금융
    components = ['055550','105560','086790','000810','032830']
    features = ['volume','infr','short']
    boundarys = [5, 10, 20, 60, 120]
    balance = 800000
    
    delayed_reward_threshold = .05
    lr = .002
    num_epoches = 1000
    discount_factor = 0
    start_epsilon = .5
    max_memory = 30
    batch_ver = 3
    TU_ver = 1
    dropout = .5
    margin = .1
    num_agent = 1
    
    date_run = '2017-06-01'
    date_end = '2017-09-30'

    #windows 설정
    ma_start = 5
    ma_stop = 150
    ma_step = 5
    min_case = 118
    max_case = 120
    case_step = 1
    num_windows = 1
    
    #에이전트 넘버 자동 설정
    if mode == 0:
        num_agent = 0
        while True:    
            str_num_agent = str(num_agent).rjust(2 , '0')
            sub_dir = 'project_%s/%s/V%s_LR%s_DF%s_DRT%s_MEM%s_DO%s_e%s_%s' % (project_num,
                           stock_code, batch_ver, lr, discount_factor, delayed_reward_threshold,
                           max_memory, dropout, num_epoches, str_num_agent)
            if os.path.exists(sub_dir):
                num_agent += 1
            else:
                os.makedirs(sub_dir)        
                break    
    else:
        str_num_agent = str(num_agent).rjust(2 , '0')
        sub_dir = 'project_%s/%s/V%s_LR%s_DF%s_DRT%s_MEM%s_DO%s_e%s_%s' % (project_num,
               stock_code, batch_ver, lr, discount_factor, delayed_reward_threshold,
               max_memory, dropout, num_epoches, str_num_agent)

    #DM객체
    data_manager = DataManager()

    #로그 기록
    log_dir = os.path.join(settings.BASE_DIR, sub_dir)
    timestr = settings.get_time_str()
    file_handler = logging.FileHandler(filename=os.path.join(
        log_dir, "%s_%s.log" % (stock_code, str_num_agent)), encoding='utf-8')
    stream_handler = logging.StreamHandler()
    file_handler.setLevel(logging.DEBUG)
    stream_handler.setLevel(logging.INFO)
    logging.basicConfig(format="%(message)s",
                        handlers=[file_handler, stream_handler], level=logging.DEBUG)
    
    #학습 데이터 준비(열0:date)
    training_data = data_manager.get_training_data(components, date_run, date_end)
    
    #윈도우와 구성 종목 인덱스와의 관계
    for i in enumerate(components):
        #component의 components 내 인덱스 준비
        comp_idx = i[0]
        component = i[1]
        # 주식 데이터 준비
        chart_data = data_manager.load_chart_data(component)
        prep_data = data_manager.preprocess(chart_data, component, comp_idx)
        training_data_temp = data_manager.build_training_data(prep_data, component, comp_idx)

        # 기간 필터링
        training_data_temp = training_data_temp[(training_data_temp['date'] >= date_run) &
                                  (training_data_temp['date'] <= date_end)]
        training_data_temp = training_data_temp.dropna()

# =============================================================================
#         # 차트 데이터 분리
#         features_chart_data = ['date', 'open_{}'.format(component), 'high_{}'.format(component),
#                                'low_{}'.format(component), 'close_{}'.format(component), 'volume_{}'.format(component),
#                                'infr_{}'.format(component), 'short_{}'.format(component)]
#         chart_data = training_data_temp[features_chart_data]
# =============================================================================

        # 학습 데이터 분리
        features_training_data_ratio = ['open_lastclose_ratio_{}'.format(component),
                                        'high_close_ratio_{}'.format(component),
                                        'low_close_ratio_{}'.format(component),
                                        'close_lastclose_ratio_{}'.format(component),
                                        'volume_lastvolume_ratio_{}'.format(component)]
        features_training_data_ma = []
        for window in data_manager.windows[comp_idx][0]:
            features_training_data_ma.append('close_ma%d_ratio_%s' % (window, component))
        for window in data_manager.windows[comp_idx][1]:
            features_training_data_ma.append('volume_ma%d_ratio_%s' % (window, component))
        for window in data_manager.windows[comp_idx][2]:
            features_training_data_ma.append('infr_ma%d_ratio_%s' % (window, component))
        for window in data_manager.windows[comp_idx][3]:
            features_training_data_ma.append('short_ma%d_ratio_%s' % (window, component))
        
        features_training_data = features_training_data_ratio + features_training_data_ma
        for feature in features_training_data:
            training_data[feature] = training_data_temp[feature].values
        training_data_temp = None
        
    training_data = training_data.drop(['date'], axis = 'columns')
    
    #차트 데이터 준비
    chart_data = data_manager.load_chart_data(stock_code)
    # 기간 필터링
    chart_data = chart_data[(chart_data['date'] >= date_run) &
                                  (chart_data['date'] <= date_end)]
    chart_data = chart_data.dropna()
    
    features = ['close_{}'.format(stock_code),'volume_{}'.format(stock_code),
                'infr_{}'.format(stock_code),'short_{}'.format(stock_code)]
    windows = data_manager.get_windows(components, features, boundarys, chart_data, ma_start,
                    ma_stop, ma_step, date_end, min_case, max_case, case_step)

    policy_learner = PolicyLearner(
            stock_code=stock_code, chart_data=chart_data, training_data=training_data,
            delayed_reward_threshold=delayed_reward_threshold, lr=lr, sub_dir=sub_dir,
            str_num_agent=str_num_agent, dropout=dropout, TU_ver=TU_ver, batch_ver=batch_ver, margin=margin,
            date_run=date_run, date_end=date_end)

    if mode == 0:
    # 강화학습 시작
        policy_learner.fit(balance=balance, num_epoches=num_epoches, max_memory=max_memory,
                           discount_factor=discount_factor, start_epsilon=start_epsilon,
                           batch_ver=batch_ver, windows=windows, project_num=project_num)

# =============================================================================
#     # 정책 신경망을 파일로 저장 -> 매 에포크마다 저장
#         model_dir = os.path.join(settings.BASE_DIR, sub_dir)
#         model_path = os.path.join(model_dir, 'model_%s.h5' % str_num_agent)
#         policy_learner.policy_network.save_model(model_path)
#         
# =============================================================================
    elif mode == 1:
        policy_learner.resume(sub_dir=sub_dir, num_epoches=num_epoches,
                              max_memory=max_memory, balance=balance,
                              discount_factor=discount_factor, start_epsilon=start_epsilon,
                              batch_ver=batch_ver, windows=windows, project_num=project_num)
    
    else:
    # 비 학습 투자 시뮬레이션 시작
        policy_learner.trade(model_path=os.path.join(settings.BASE_DIR, sub_dir,
                                                     'model_e%d_%s.h5' % (num_epoches, str_num_agent)),
                             num_epoches=1, max_memory=60, balance=balance,
                             discount_factor=discount_factor, batch_ver=batch_ver,
                             windows=windows, project_num=project_num)

