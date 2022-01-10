import os

model_list = ['CNN', 'CNN_GRU', 'CNN_LSTM', 'GRU', 'GRU_AE', 'GRU_LSTM',
              'LSTM', 'LSTM_AE', 'LSTM_GRU', 'MLP', 'MLP_AE']

security_list = ['Al', 'Cu', 'Corn', 'EURCHF', 'EURUSD', 'GBPUSD', 'Gilt10y',
                 'Bund10y', 'Treasury10y', 'Amazon', 'Google', 'Nvidia']

for m in model_list:
    for s in security_list:
        old_path = './notebooks_json/'+m+'/'+m+'_'+s+'.ipynb'
        new_path = './notebooks_json/'+m+'/'+m+'_'+s+'.json'

        if not os.path.isfile(old_path):
            print(old_path+'is not a file')

        os.rename(old_path, new_path)
