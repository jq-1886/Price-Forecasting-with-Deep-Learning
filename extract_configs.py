import json

# storing models as a list to iterate through them
model_list = ['CNN', 'CNN_GRU', 'CNN_LSTM', 'GRU', 'GRU_AE', 'GRU_LSTM',
              'LSTM', 'LSTM_AE', 'LSTM_GRU', 'MLP', 'MLP_AE']

# storing securities as a list to iterate through them
security_list = ['Al', 'Cu', 'Corn', 'EURCHF', 'EURUSD', 'GBPUSD', 'Gilt10y',
                 'Bund10y', 'Treasury10y', 'Amazon', 'Google', 'Nvidia']

# storing the index of the cell that contains best_config dict
cell_index = 20

# opening the text file
text_file = open('./best_configs.txt', "a")

# writing to text
text_file.write("{\n")

# a loop to move through each model
for m in model_list:
    # logic to amend indexing based on model
    if m == 'GRU':
        line_index = 1
    else:
        line_index = 2
    # a loop to move through each security
    for s in security_list:
        # setting the path of the .json notebook data
        path = './notebooks_json/'+m+'/'+m+'_'+s+'.json'
        # opening the relevant .json file
        json_f = open(path,)
        # loading the data
        json_data = json.load(json_f)
        # accessing the relevant cell via indexing and the assigning
        json_dict = json_data["cells"][cell_index]['source']
        # creating a list to form the first line of the dictionary
        list_dict = ['\n'+'\t'+"'"+m+'_'+s+"'"+' : {\n']
        # looping through each line in json_dict
        for i in json_dict[line_index:]:
            # appending to list_dict
            list_dict.append(i)
        # looping through each line in list_dict
        for i in range(0, len(list_dict)):
            # logic to ensure final lines get a comma
            if i == len(list_dict) - 1:
                # special case if the very final line, do not include comma
                if m == model_list[len(model_list)-1] and s == security_list[
                        len(security_list)-1]:
                    text_file.write('\t'+list_dict[i])
                # else, if final line, include a comma
                else:
                    text_file.write('\t'+list_dict[i]+',')
            # else simply writing to file
            else:
                text_file.write('\t'+list_dict[i])
        # finally closing the json file
        json_f.close()


# writing the final line
text_file.write('\n}')

# closing the text file
text_file.close()
