import numpy as np
import pandas as pd
import os
import csv

# function which decides which columns are relevant to read from the database
def filtered_col_names():
   cols =  [
    'race', 'gender', 'age', 'admission_type_id', 'discharge_disposition_id',
    'admission_source_id', 'time_in_hospital', 'medical_specialty', 'num_lab_procedures',
    'num_procedures', 'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient',
    'number_diagnoses', 'insulin', 'change', 'diabetesMed', 'readmitted'
   ]
   return cols

# function which maps the different race names to numbers
def mapping_race(s):
    race_value_mapping = {
        'Caucasian':0,
        'AfricanAmerican':1,
        'Other':2,
        'Asian':3,
        'Hispanic':4,
        '?':5
    }
    return race_value_mapping[s]
# function which maps the 2 gender buckets to numbers
def mapping_gender(s):
    gender_value_mapping = {
        'Male':0,
        'Female':1,
        'Unknown/Invalid':2
    }
    return gender_value_mapping[s]

# function which maps the different age buckets to the mid-point of the age group
def mapping_age(s):
    age_value_mapping = {
        '[0-10)':5, '[10-20)':15, '[20-30)':25, '[30-40)':35,
        '[40-50)':45, '[50-60)':55, '[60-70)':65, '[70-80)':75,
        '[80-90)':85, '[90-100)':95
    }
    return age_value_mapping[s]

# function which maps insulin changes to numbers
def mapping_insulin(s):
    insulin_value_mapping = {
        'No':0,
        'Down': 1,
        'Steady':2,
        'Up':3
    }
    return insulin_value_mapping[s]

# fucntion which maps diabetesMed into numbers
def mapping_diabetesMed(s):
    diabetedMed_value_mapping = {
        'Yes':1,
        'No':0
    }
    return diabetedMed_value_mapping[s]

# function which maps A1Cresults into numbers
def mapping_A1Cresult(s):
    A1Cresult_value_mapping = {
        '>7':1,
        '>8':2,
        'Norm':0
    }
    return A1Cresult_value_mapping[s]

# function which maps "change" into numbers
def mapping_change(s):
    change_value_mapping = {'No':0, 'Ch':1}
    return change_value_mapping[s]

# function which maps "max_glu_serum" to numbers
def mapping_max_glu_serum(s):
    max_glu_serum_value_mapping = {'Norm':0,'>200':1,'>300':2}
    return max_glu_serum_value_mapping[s]

# function which maps medical specialty to numbers; p is the dictionary with the mapping
def mapping_medical_specialty(s,p):
    return int(p[s])

# function which maps the different readmittance states to number.
def mapping_readmittance(s):
    readmitted_value_mapping = {
        'NO':-1,
        '>30':0,
        '<30':1
    }
    return readmitted_value_mapping[s]


# function which reads the csv file into a pandas datframe
def read_data(filename,nrows=None, cols = None):
    nrows_to_read = None
    cols_to_read = None
    if not nrows == None:
        nrows_to_read = nrows
    if not cols == None:
        cols_to_read = cols

    x = pd.read_csv(filename,nrows=nrows_to_read, usecols=cols_to_read, converters={
        'age':mapping_age,
        'race':str,
        'max_glu_serum':str,
        'A1Cresult':str,
        'medical_specialty':str,
        'payer_code':str,
        'admission_source_id':np.int32,
        'time_in_hospital':np.int32,
        'admission_type_id':np.int32,
        'discharge_disposition_id':np.int32,
        'insulin':mapping_insulin,
        'diabetesMed':mapping_diabetesMed,
        'change':mapping_change,
        'readmitted':mapping_readmittance
    }, na_values=['?'], low_memory=False)
    return x

# function which reads the full path so that the code can be run from any directory
def get_full_path(dir_name,file_name):
    path = os.path.dirname(__file__)
    path = os.path.join(path,dir_name)
    full_path = os.path.join(path,file_name)
    return full_path

def read_filter_clean_map_csv():
    path_to_df = get_full_path('dataset_diabetes','diabetic_data.csv')
    df = read_data(path_to_df,cols=filtered_col_names())

    mapping_file = get_full_path('dataset_diabetes','medical_specialty_mapping.csv')
    with open(mapping_file, mode='r') as infile:
        reader = csv.reader(infile)
        p = {rows[0]:rows[1] for rows in reader}

    # df = df[df.race != '?']
    # df = df[df.gender != 'Unknown/Invalid']
    # df = df[df.medical_specialty != '?' ]
    df['race'] = df['race'].map(lambda a: mapping_race(a))
    df['gender'] = df['gender'].map(lambda a: mapping_gender(a))
    df['medical_specialty'] = df['medical_specialty'].map(lambda a: mapping_medical_specialty(a,p))

    return df


if __name__ == '__main__':
    df = read_filter_clean_map_csv()
    n,m = df.shape
    print n,m

    cols = df.columns.values.tolist()
    print cols
    for c in cols:
        print c, df[c].unique()

