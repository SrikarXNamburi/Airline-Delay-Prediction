import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
def myModel():
    import pandas as pd
    penguins = pd.read_csv('Airlines.csv')

    df = penguins.copy()
    df = df.drop('id',axis=1)
    df = df.drop('Flight',axis=1)
    target = 'Delay'
    encode = ['Airline','AirportFrom','AirportTo']

    for col in encode:
        dummy = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df,dummy], axis=1)
        del df[col]

    # Separating X and y
    X = df.drop('Delay', axis=1)
    Y = df['Delay']

    # Build random forest model
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier()
    clf.fit(X, Y)

    return clf;

    
st.write("""
# Airlines Prediction App

Created by: Srikar Namburi

This app predicts the **delay** in Airlines using Random Forest Classification

Data obtained from https://www.kaggle.com/datasets/jimschacko/airlines-dataset-to-predict-a-delay with regards to Jims Chacko
""")

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        airline = st.sidebar.selectbox('Airline',('CO','US', 'AA', 'AS', 'DL', 'B6', 'HA', 'OO', '9E', 'OH', 'EV',
       'XE', 'YV', 'UA', 'MQ', 'FL', 'F9', 'WN'))
        airportFrom = st.sidebar.selectbox('AirportFrom',('SFO', 'PHX', 'LAX', 'ANC', 'LAS', 'SLC', 'DEN', 'ONT', 'FAI',
       'BQN', 'PSE', 'HNL', 'BIS', 'IYK', 'EWR', 'BOS', 'MKE', 'GFK',
       'OMA', 'GSO', 'LMT', 'SEA', 'MCO', 'TPA', 'DLH', 'MSP', 'FAR',
       'MFE', 'MSY', 'VPS', 'BWI', 'MAF', 'LWS', 'RST', 'ALB', 'DSM',
       'CHS', 'MSN', 'JAX', 'SAT', 'PNS', 'BHM', 'LIT', 'SAV', 'BNA',
       'ICT', 'ECP', 'DHN', 'MGM', 'CAE', 'PWM', 'ACV', 'EKO', 'PHL',
       'ATL', 'PDX', 'RIC', 'BTR', 'HRL', 'MYR', 'TUS', 'SBN', 'CAK',
       'TVC', 'CLE', 'ORD', 'DAY', 'MFR', 'BTV', 'TLH', 'TYS', 'DFW',
       'FLL', 'AUS', 'CHA', 'CMH', 'LRD', 'BRO', 'CRP', 'LAN', 'PVD',
       'FWA', 'JFK', 'LGA', 'OKC', 'PIT', 'PBI', 'ORF', 'DCA', 'AEX',
       'SYR', 'SHV', 'VLD', 'BDL', 'FAT', 'BZN', 'RDM', 'LFT', 'IPL',
       'EAU', 'ERI', 'BUF', 'IAH', 'MCI', 'AGS', 'ABI', 'GRR', 'LBB',
       'CLT', 'LEX', 'MBS', 'MOD', 'AMA', 'SGF', 'AZO', 'ABE', 'SWF',
       'BGM', 'AVP', 'FNT', 'GSP', 'ATW', 'ITH', 'TUL', 'COS', 'ELP',
       'ABQ', 'SMF', 'STL', 'IAD', 'DTW', 'RDU', 'RSW', 'OAK', 'ROC',
       'IND', 'CVG', 'MDW', 'SDF', 'ABY', 'TRI', 'XNA', 'ROA', 'MLI',
       'LYH', 'EVV', 'HPN', 'FAY', 'EWN', 'CSG', 'GPT', 'MLU', 'MOB',
       'OAJ', 'CHO', 'ILM', 'BMI', 'PHF', 'ACY', 'JAN', 'CID', 'GRK',
       'HOU', 'CRW', 'HTS', 'PSC', 'BOI', 'SBP', 'CLD', 'PSP', 'SBA',
       'MEM', 'MRY', 'GEG', 'RDD', 'PAH', 'CMX', 'SPI', 'EUG', 'CIC',
       'PIH', 'SGU', 'COD', 'MIA', 'MHT', 'GRB', 'FSD', 'SJU', 'AVL',
       'BFL', 'RAP', 'DRO', 'PIA', 'OGG', 'SIT', 'TXK', 'RNO', 'DAL',
       'SCE', 'MEI', 'MDT', 'FCA', 'SJC', 'KOA', 'PLN', 'SAN', 'GNV',
       'HLN', 'GJT', 'CPR', 'FSM', 'CMI', 'GTF', 'HDN', 'ITO', 'MTJ',
       'HSV', 'BTM', 'BIL', 'COU', 'MSO', 'SMX', 'TWF', 'ISP', 'GCC',
       'LIH', 'LNK', 'DAB', 'SNA', 'MQT', 'LGB', 'CWA', 'LSE', 'BUR',
       'ACT', 'MHK', 'MOT', 'IDA', 'SUN', 'GTR', 'MLB', 'SRQ', 'JAC',
       'ASE', 'LCH', 'JNU', 'ROW', 'BQK', 'YUM', 'FLG', 'EGE', 'GUC',
       'EYW', 'RKS', 'BGR', 'ELM', 'ADQ', 'OTZ', 'OTH', 'STT', 'KTN',
       'BET', 'SJT', 'CDC', 'CEC', 'SPS', 'SCC', 'STX', 'OME', 'MKG',
       'WRG', 'TYR', 'BRW', 'GGG', 'PSG', 'BKG', 'YAK', 'CLL', 'SAF',
       'CYS', 'LWB', 'CDV', 'FLO', 'BLI', 'DBQ', 'TOL', 'UTM', 'PIE',
       'ADK', 'ABR', 'TEX', 'MMH', 'GUM'),key="1")
        airportTo = st.sidebar.selectbox('Airport To',('PHX', 'SFO', 'LAX', 'ANC', 'LAS', 'SLC', 'DEN', 'ONT', 'FAI',
       'BQN', 'PSE', 'HNL', 'BIS', 'IYK', 'EWR', 'BOS', 'MKE', 'GFK',
       'OMA', 'GSO', 'LMT', 'SEA', 'MCO', 'TPA', 'DLH', 'MSP', 'FAR',
       'MFE', 'MSY', 'VPS', 'BWI', 'MAF', 'LWS', 'RST', 'ALB', 'DSM',
       'CHS', 'MSN', 'JAX', 'SAT', 'PNS', 'BHM', 'LIT', 'SAV', 'BNA',
       'ICT', 'ECP', 'DHN', 'MGM', 'CAE', 'PWM', 'ACV', 'EKO', 'PHL',
       'ATL', 'PDX', 'RIC', 'BTR', 'HRL', 'MYR', 'TUS', 'SBN', 'CAK',
       'TVC', 'CLE', 'ORD', 'DAY', 'MFR', 'BTV', 'TLH', 'TYS', 'DFW',
       'FLL', 'AUS', 'CHA', 'CMH', 'LRD', 'BRO', 'CRP', 'LAN', 'PVD',
       'FWA', 'JFK', 'LGA', 'OKC', 'PIT', 'PBI', 'ORF', 'DCA', 'AEX',
       'SYR', 'SHV', 'VLD', 'BDL', 'FAT', 'BZN', 'RDM', 'LFT', 'IPL',
       'EAU', 'ERI', 'BUF', 'IAH', 'MCI', 'AGS', 'ABI', 'GRR', 'LBB',
       'CLT', 'LEX', 'MBS', 'MOD', 'AMA', 'SGF', 'AZO', 'ABE', 'SWF',
       'BGM', 'AVP', 'FNT', 'GSP', 'ATW', 'ITH', 'TUL', 'COS', 'ELP',
       'ABQ', 'SMF', 'STL', 'IAD', 'DTW', 'RDU', 'RSW', 'OAK', 'ROC',
       'IND', 'CVG', 'MDW', 'SDF', 'ABY', 'TRI', 'XNA', 'ROA', 'MLI',
       'LYH', 'EVV', 'HPN', 'FAY', 'EWN', 'CSG', 'GPT', 'MLU', 'MOB',
       'OAJ', 'CHO', 'ILM', 'BMI', 'PHF', 'ACY', 'JAN', 'CID', 'GRK',
       'HOU', 'CRW', 'HTS', 'PSC', 'BOI', 'SBP', 'CLD', 'PSP', 'SBA',
       'MEM', 'MRY', 'GEG', 'RDD', 'PAH', 'CMX', 'SPI', 'EUG', 'CIC',
       'PIH', 'SGU', 'COD', 'MIA', 'MHT', 'GRB', 'FSD', 'SJU', 'AVL',
       'BFL', 'RAP', 'DRO', 'PIA', 'OGG', 'SIT', 'TXK', 'RNO', 'DAL',
       'SCE', 'MEI', 'MDT', 'FCA', 'SJC', 'KOA', 'PLN', 'SAN', 'GNV',
       'HLN', 'GJT', 'CPR', 'FSM', 'CMI', 'GTF', 'HDN', 'ITO', 'MTJ',
       'HSV', 'BTM', 'BIL', 'COU', 'MSO', 'SMX', 'TWF', 'ISP', 'GCC',
       'LIH', 'LNK', 'DAB', 'SNA', 'MQT', 'LGB', 'CWA', 'LSE', 'BUR',
       'ACT', 'MHK', 'MOT', 'IDA', 'SUN', 'GTR', 'MLB', 'SRQ', 'JAC',
       'ASE', 'LCH', 'JNU', 'ROW', 'BQK', 'YUM', 'FLG', 'EGE', 'GUC',
       'EYW', 'RKS', 'BGR', 'ELM', 'ADQ', 'OTZ', 'OTH', 'STT', 'KTN',
       'BET', 'SJT', 'CDC', 'CEC', 'SPS', 'SCC', 'STX', 'OME', 'MKG',
       'WRG', 'TYR', 'BRW', 'GGG', 'PSG', 'BKG', 'YAK', 'CLL', 'SAF',
       'CYS', 'LWB', 'CDV', 'FLO', 'BLI', 'DBQ', 'TOL', 'UTM', 'PIE',
       'ADK', 'ABR', 'TEX', 'MMH', 'GUM'),key="2")
        day_of_the_week = st.sidebar.slider('Day of the week', 1,5,7)
        time = st.sidebar.slider('Time (hr)', 5,500,1500)
        length = st.sidebar.slider('Length of Journey', 0,300,660)
        data = {'Airline': airline,
                'AirportFrom': airportFrom,
                'AirportTo': airportTo,
                'DayOfWeek': day_of_the_week,
                'Time': time,
                'Length': length}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

airlines_raw = pd.read_csv('Airlines.csv')
airlines = airlines_raw.drop(columns=['id','Flight','Delay'])
df = pd.concat([input_df,airlines],axis=0)

encode = ['Airline','AirportFrom','AirportTo']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1] # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)

# Reads in saved classification model
load_clf = myModel()

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

st.subheader('Prediction')
my_airlines = np.array(['No Delay','Delay'])
st.write(my_airlines[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)

