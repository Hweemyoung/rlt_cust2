#cust0: 기관+외국인 거래량 지표 추가

import pandas as pd
import numpy as np
import os
import settings

class DataManager:    
    def __init__(self, features, windows=[5,10,20,60,120], boundary=[5,10,20,60,120]):
        self.features = features
        self.windows = windows
        self.boundary = boundary

    def load_chart_data(self, component, features):
        chart_data = pd.read_csv(os.path.join(settings.BASE_DIR,
                             'data/chart_data/{}.csv'.format(component)), thousands=',', header=None)
        columns = []
        for feature in features:
            columns.append('%s_%s' %(feature, component))
        chart_data.columns = ['date','open_%s'%component,'high_%s'%component,
                              'low_%s'%component,'close_%s'%component] + columns
        return chart_data

    def preprocess(self, chart_data, component, features, windows_comp):
        prep_data = chart_data
        for idx, feature in enumerate(features):
            for window in windows_comp[idx]:
                prep_data['%s_ma%d_%s' %(feature,window,component)] = \
                prep_data['%s_%s' % (feature, component)].rolling(window).mean()
        return prep_data
    
    def build_training_data(self, prep_data, component, features, windows_comp):
        training_data_temp = prep_data
        
        training_data_temp['open_lastclose_ratio_{}'.format(component)] = np.zeros(len(training_data_temp))
        training_data_temp.loc[1:, 'open_lastclose_ratio_{}'.format(component)] = \
            (training_data_temp['open_{}'.format(component)][1:].values -
                                training_data_temp['close_{}'.format(component)][:-1].values) / \
            training_data_temp['close_{}'.format(component)][:-1].values
        training_data_temp['high_close_ratio_{}'.format(component)] = \
            (training_data_temp['high_{}'.format(component)].values -
                                training_data_temp['close_{}'.format(component)].values) / \
            training_data_temp['close_{}'.format(component)].values
        training_data_temp['low_close_ratio_{}'.format(component)] = \
            (training_data_temp['low_{}'.format(component)].values -
                                training_data_temp['close_{}'.format(component)].values) / \
            training_data_temp['close_{}'.format(component)].values
        training_data_temp['close_lastclose_ratio_{}'.format(component)] = np.zeros(len(training_data_temp))
        training_data_temp.loc[1:, 'close_lastclose_ratio_{}'.format(component)] = \
            (training_data_temp['close_{}'.format(component)][1:].values -
                                training_data_temp['close_{}'.format(component)][:-1].values) / \
            training_data_temp['close_{}'.format(component)][:-1].values
        
        for feature in features:
            training_data_temp['%s_last%s_ratio_%s' % (feature, feature, component)] = np.zeros(len(training_data_temp))
            training_data_temp.loc[1:, '%s_last%s_ratio_%s' % (feature, feature, component)] = \
                (training_data_temp['%s_%s'%(feature, component)][1:].values -
                training_data_temp['%s_%s'%(feature, component)][:-1].values) / \
                training_data_temp['%s_%s'%(feature, component)][:-1]\
                .replace(to_replace=0, method='ffill') \
                .replace(to_replace=0, method='bfill').values
        
        #infr_lastinfr_ratio는 제거할까?
        training_data_temp.drop('infr_lastinfr_ratio', axis=1, inplace=True)
        
        #각 특징량 ma 생성
        for idx, feature in enumerate(features):
            for window in windows_comp[idx]:
                training_data_temp['%s_ma%d_ratio_%s' % (feature, window, component)] = \
                    (training_data_temp['%s_%s'%(feature, component)] - \
                    training_data_temp['%s_ma%d_%s' % (feature, window, component)]) / \
                                        training_data_temp['%s_%s'%(feature, component)]
                                        
# =============================================================================
#여기서 트레이딩 데이터의 close/lastclose 인덱스는 3.
# =============================================================================
        return training_data_temp
    
    def _get_data(self, stock_code, components, features, date_run, date_end):
        #차트 데이터 획득
        chart_data_stock = self.load_chart_data(stock_code, features)
        #기간 필터링
        chart_data_stock = chart_data_stock[(chart_data_stock['date'] >= date_run) &
                                  (chart_data_stock['date'] <= date_end)]
        chart_data_stock.dropna(inplace = True)
        #차트 데이터 분리
        chart_data_stock = chart_data_stock['date','open_%s'%stock_code,'high_%s'%stock_code,
                              'low_%s'%stock_code,'close_%s'%stock_code].values
        
        #학습 데이터 획득
        training_data = chart_data_stock['date']
        for component in components:
            windows_comp = _get_windows_comp
            chart_data = self.load_chart_data(component, features)
            prep_data = self.preprocess(chart_data, component, features, windows_comp)
            training_data_temp = self.build_training_data(prep_data, component, features, windows_comp)
            
            #기간 필터링
            training_data_temp = training_data_temp[(training_data_temp['date'] >= date_run) &
                                  (training_data_temp['date'] <= date_end)]
            training_data_temp.dropna(inplace = True)
            
            #학습 데이터 분리
            columns_training = ['open_lastclose_ratio_{}'.format(component),'high_close_ratio_{}'.format(component),
                                'low_close_ratio_{}'.format(component),'close_lastclose_ratio_{}'.format(component)]
            for idx, feature in enumerate(features):
                columns_training.append('%s_last%s_ratio_%s' % (feature, feature, component))
                for window in windows_comp[idx]:
                    columns_training.append('%s_ma%d_ratio_%s')
            training_data[columns_training] = training_data_temp[columns_training].values
        
        #일자 제거
        training_data.drop('date', axis=1, inplace = True)
        return chart_data_stock, training_data
            
    def _get_windows_comp(self, ver, component, features):
        if ver == 1:
            
            
    
    
    def roller(self, component, chart_data, features, ma_start, ma_stop, ma_step, date_end, min_case, max_case, case_step):        
        import matplotlib.pyplot as plt
        import os
        
        rolled_data = chart_data
        
        rolled_data['close_lastclose_ratio'] = np.zeros(len(rolled_data))
        rolled_data['close_lastclose_ratio'][1:] = \
            (rolled_data['close_{}'.format(component)][1:].values -
                         rolled_data['close_{}'.format(component)][:-1].values) / \
            rolled_data['close_{}'.format(component)][:-1].values
        
        case_arange = np.arange(start=min_case, stop=max_case, step=case_step)
        corrcoef=[]
        for feature in features:                
            for window in np.arange(start=ma_start, stop=ma_stop, step=ma_step):
                temp_data=rolled_data
                temp_data['ma{}'.format(window)] = temp_data[feature].rolling(window).mean()
                temp_data['ma{}_ratio'.format(window)] = \
                (temp_data[feature] - temp_data['ma{}'.format(window)]) / \
                temp_data['ma{}'.format(window)]
                temp_data=temp_data[(temp_data['date'] <= date_end)]
                for case in reversed(case_arange):
                    temp_data=temp_data[-case-1:-1]
                    temp_data=temp_data.dropna()
                    corrcoef.append((feature ,window,
                             np.corrcoef(temp_data['close_lastclose_ratio'], temp_data['ma{}_ratio'.format(window)])[0][1],
                             case
                             )
                            )
        corrcoef = pd.DataFrame(corrcoef)
        #0열 특징량/1열 ma윈도우/2열 상관계수/3열 상관계수산출 케이스 수
        arg1 = 0
        values = []
        detail = pd.DataFrame(index = np.arange(start=ma_start, stop=ma_stop, step=ma_step))
        fig, axes = plt.subplots(len(case_arange), len(features), sharex=True, sharey=True)
        for feature in features:
            arg0 = 0
            axes[0,arg1].set_title(feature)
            temp_data = corrcoef[(corrcoef[0] == feature)]
            for case in case_arange:
                temp_data_case = temp_data[(temp_data[3] == case)]
                values.append(tuple(temp_data_case.loc[np.argmax(temp_data_case[2])]))
                values.append(tuple(temp_data_case.loc[np.argmin(temp_data_case[2])]))
                axes[arg0,arg1].scatter(temp_data_case[1], temp_data_case[2], marker='.')
                axes[arg0,0].set_ylabel(case)
                arg0 += 1
            if max_case - min_case <= 2:
                detail['corecoef_{}'.format(feature)] = temp_data_case.values[:,2:3]
            arg1 += 1
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(os.path.join('E:/','Textbook','fig0.png'), dpi=400, bbox_inches='tight')
        values = pd.DataFrame(values)
        #각 특징량 + 케이스 수 조합에 있어서 상관계수 최대/최소 모음. -> ma 윈도우 관찰
        return values, detail if max_case - min_case <= 2 else values
    
    def get_detail(self, component, chart_data, features, ma_start, ma_stop,
                   ma_step, date_end, min_case, max_case, case_step):
        import os
        
        rolled_data = chart_data
        
        rolled_data['close_lastclose_ratio'] = np.zeros(len(rolled_data))
        rolled_data['close_lastclose_ratio'][1:] = \
            (rolled_data['close_{}'.format(stock_code)][1:].values -
                         rolled_data['close_{}'.format(component)][:-1].values) / \
            rolled_data['close_{}'.format(component)][:-1].values
        
        case_arange = np.arange(start=min_case, stop=max_case, step=case_step)
        corrcoef=[]
        for feature in features:                
            for window in np.arange(start=ma_start, stop=ma_stop, step=ma_step):
                temp_data=rolled_data
                temp_data['ma{}'.format(window)] = temp_data[feature].rolling(window).mean()
                temp_data['ma{}_ratio'.format(window)] = \
                (temp_data[feature] - temp_data['ma{}'.format(window)]) / \
                temp_data['ma{}'.format(window)]
                temp_data=temp_data[(temp_data['date'] <= date_end)]
                for case in reversed(case_arange):
                    temp_data=temp_data[-case-1:-1]
                    temp_data=temp_data.dropna()
                    corrcoef.append((feature ,window,
                             np.corrcoef(temp_data['close_lastclose_ratio'], temp_data['ma{}_ratio'.format(window)])[0][1],
                             case
                             )
                            )
        corrcoef = pd.DataFrame(corrcoef)
        #0열 특징량/1열 ma윈도우/2열 상관계수/3열 상관계수산출 케이스 수
        detail = np.zeros((len(features),len(arange_ma)))
        
    def _get_table_corr(self, close_lastclose_ratio, training_data_temp, component, feature, ma_start, ma_stop, ma_step):
        #table_ma: stock_code의 close_lastclose_ratio 와 한 component의 
        #한 feature에 대해 각 window로 roll된 데이터에 대해 ratio를 취한 DF
        #DF.shape = (case개수, 1+len(arange_ma))
        arange_ma = np.arange(start=ma_start, stop=ma_stop, step=ma_step)
        table_corr = np.zeros(len(arange_ma))
        for i,window in enumerate(arange_ma):
            table_corr[i] = np.corrcoef(close_lastclose_ratio,
                      training_data_temp['%s_ma%d_ratio_%s' % (feature, window, component)])
        return table_corr
    
    def _get_table_ma(self, chart_data_stock, training_data_temp, feature, ma_start, ma_stop, ma_step, num_case):
        table_ma = pd.DataFrame(chart_data_stock['close_lastclose_ratio'])
        arange_ma = np.arange(start=ma_start, stop=ma_stop, step=ma_step)        
        for i,window in enumerate(arange_ma):
            table_ma['ma%s_ratio'%window_ma] = training_data_temp['short_ma%d_ratio_%s' % (window, component)]
        
        
                
# =============================================================================
#             plot = pd.DataFrame(corrcoef)[1].plot()
# =============================================================================

    def get_windows(self, components, features, boundarys, chart_data, ma_start,
                    ma_stop, ma_step, date_end, min_case, max_case, case_step):
        arange_ma = np.arange(start=ma_start, stop=ma_stop, step=ma_step)
        windows = np.zeros((len(components),len(features),len(boundarys)))
        components = np.array(components)
        boundarys = np.array(boundarys)
        for component in components:
            values, detail = self.roller(component, chart_data, features, ma_start,
                                         ma_stop, ma_step, date_end, min_case, max_case, case_step)
            for boundary in boundarys:
                #j:detail 칼럼의 인덱스
                for j in range(detail.shape[1]):
                    windows[np.argmax(components[(components == component)])]\
                    [j][np.argmax(boundarys(boundarys == boundary))] \
                    = table_ma[table_ma.index <= boundary].sort_values(by=detail.columns[j]).index[0]
                detail.drop(detail.index[(detail.index <= boundary)], inplace = True)
        return windows
                    
        
# =============================================================================
#     def get_windows(self, num_windows, features, values):
#         table = pd.DataFrame(np.zeros((num_windows, len(features))), columns = features)
#         for feature in features:
#             values_feature = pd.DataFrame(values[values[0] == feature].values[:,[1,2]])
#             values_feature = values_feature.sort_values(by = 1, ascending = False if feature != 'short' else True)
#             table[feature] = values_feature[0].values[:num_windows]
#         return table
# =============================================================================

    def build_p_dev(self, prep_data, window):
        p_dev_data = prep_data[['date', 'close_ma{}'.format(window)]]
        p_dev_data['p_dev'] = np.zeros(len(p_dev_data))
        p_dev_data.loc[1:, 'p_dev'] = p_dev_data['close_ma{}'.format(window)][1:].values / \
            p_dev_data['close_ma{}'.format(window)][:-1].values
        return p_dev_data
    


# chart_data = pd.read_csv(fpath, encoding='CP949', thousands=',', engine='python')
