import pandas as pd
import numpy as np


def statistics(arrayz):

    race_scores_fair = []
    gender_scores_fair = []
    age_scores_fair = []
    race_preds_fair = []
    gender_preds_fair = []
    age_preds_fair = []

    for i in range (len(arrayz)):
        outputs = np.array(arrayz[i])
        outputs = np.squeeze(outputs)

        race_outputs = outputs[:7]
        gender_outputs = outputs[7:9]
        age_outputs = outputs[9:18]

        race_score = np.exp(race_outputs) / np.sum(np.exp(race_outputs))
        gender_score = np.exp(gender_outputs) / np.sum(np.exp(gender_outputs))
        age_score = np.exp(age_outputs) / np.sum(np.exp(age_outputs))

        race_pred = np.argmax(race_score)
        gender_pred = np.argmax(gender_score)
        age_pred = np.argmax(age_score)

        race_scores_fair.append(race_score)
        gender_scores_fair.append(gender_score)
        age_scores_fair.append(age_score)

        race_preds_fair.append(race_pred)
        gender_preds_fair.append(gender_pred)
        age_preds_fair.append(age_pred)






    result = pd.DataFrame([race_preds_fair,
                           gender_preds_fair,
                           age_preds_fair,
                           race_scores_fair,
                           gender_scores_fair,
                           age_scores_fair, ]).T
    result.columns = ['race_preds_fair',
                      'gender_preds_fair',
                      'age_preds_fair',
                      'race_scores_fair',
                      'gender_scores_fair',
                      'age_scores_fair']
    result.loc[result['race_preds_fair'] == 0, 'race'] = 'White'
    result.loc[result['race_preds_fair'] == 1, 'race'] = 'Black'
    result.loc[result['race_preds_fair'] == 2, 'race'] = 'Latino_Hispanic'
    result.loc[result['race_preds_fair'] == 3, 'race'] = 'East Asian'
    result.loc[result['race_preds_fair'] == 4, 'race'] = 'Southeast Asian'
    result.loc[result['race_preds_fair'] == 5, 'race'] = 'Indian/Native American'
    result.loc[result['race_preds_fair'] == 6, 'race'] = 'Middle Eastern'

    # # race fair 4
    #
    # result.loc[result['race_preds_fair_4'] == 0, 'race4'] = 'White'
    # result.loc[result['race_preds_fair_4'] == 1, 'race4'] = 'Black'
    # result.loc[result['race_preds_fair_4'] == 2, 'race4'] = 'Asian'
    # result.loc[result['race_preds_fair_4'] == 3, 'race4'] = 'Indian'

    # gender
    result.loc[result['gender_preds_fair'] == 0, 'gender'] = 'Male'
    result.loc[result['gender_preds_fair'] == 1, 'gender'] = 'Female'

    # age
    result.loc[result['age_preds_fair'] == 0, 'age'] = '0-2'
    result.loc[result['age_preds_fair'] == 1, 'age'] = '3-9'
    result.loc[result['age_preds_fair'] == 2, 'age'] = '10-19'
    result.loc[result['age_preds_fair'] == 3, 'age'] = '20-29'
    result.loc[result['age_preds_fair'] == 4, 'age'] = '30-39'
    result.loc[result['age_preds_fair'] == 5, 'age'] = '40-49'
    result.loc[result['age_preds_fair'] == 6, 'age'] = '50-59'
    result.loc[result['age_preds_fair'] == 7, 'age'] = '60-69'
    result.loc[result['age_preds_fair'] == 8, 'age'] = '70+'

    # result[['face_name_align',
    #         'race',
    #         'gender', 'age',
    #         'race_scores_fair',
    #         'gender_scores_fair', 'age_scores_fair']].to_csv(save, index=False)

    output = result[['race', 'gender', 'age',]]
    ret = []
    for i in range (len(output)):
        temp = [output.iat[i, 0], output.iat[i, 1], output.iat[i, 2]]
        ret.append(temp)


    # print("saved results at ", save)
    return ret
