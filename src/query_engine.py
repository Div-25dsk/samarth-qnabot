
import pandas as pd
import numpy as np
import re
from difflib import get_close_matches

crop_data = pd.read_csv("E:\DIV FOLDER\project_samarth\data\Crop-Profile_-State-wise-Area,-Production-&-Yield-of-All-Crops,-(2020-21-to-2024-25).csv")

irri_data = pd.read_html("E:\DIV FOLDER\project_samarth\data\source_irrigated_area_report.xls")[0]

#renaming the columns  for simplicity
irri_data.columns= [col[-1] if isinstance(col,tuple) else col for col in irri_data.columns]

irri_data.columns = [
    "SNo", "District", "Canal_Govt", "Canal_Private", "Canal_Total",
    "Tank_Irrigated", "Tubewell_Irrigated", "Other_Wells", "Wells_Total",
    "Other_Sources", "Net_Irrigated_Area", "Gross_Canal_Govt",
    "Gross_Canal_Private", "Gross_Canal_Total", "Gross_Tank",
    "Gross_Tubewell", "Gross_Other_Well", "Gross_Well_Total",
    "Gross_Other_Source", "Gross_Irrigated_Area"
]
print(irri_data.head())
print(irri_data.columns)

#check for missing values
print("\n missing vales - crop ",crop_data.isnull().sum())

print("\nmissing values- irrigation",irri_data.isnull().sum())

#drop the one with most

crop_data = crop_data.drop(columns=['Area-2024-25','Production-2024-25','Yield-2024-25'])

#fill the missing values with median

crop_data = crop_data.fillna(crop_data.median(numeric_only=True))

print(crop_data.isnull().sum())

#to extract state headers in irrigation table

irri= irri_data.copy()
irri['District']= irri['District'].astype(str).str.strip()
irri['is_state_row'] = irri['District'].str.contains(r'(?i)^\s*state\s*[:\-\s]',na = False)
print('State header rows count:',irri["is_state_row"].sum())

# take thse headers into state
# Step 1: Create a new column 'State' with empty values
irri['State'] = None

#  Go row by row
for i in irri.index:
    # Check if this row is a state header
    if irri.loc[i, 'is_state_row']:
        # Get the 'District' value (which actually contains the state name)
        district_value = str(irri.loc[i, 'District']).strip()

        # Remove the word 'State' and any punctuation
        state_name = district_value.replace('State:', '').replace('State -', '').replace('State', '').strip()

        # Save the cleaned name into the 'State' column
        irri.loc[i, 'State'] = state_name

# Fill state name down to district rows
irri['State'] = irri['State'].ffill()

# Remove state header rows from the irri df copying which are false back to irri
irri = irri[~irri["is_state_row"]].copy()
#dropping the column which is now not needed
irri.drop(columns=['is_state_row'], inplace=True)


#sample
print(irri[['State','District']].head(12))

# Define function to filter by clean state name
def get_state_data(df, state_name):
    df['State_clean'] = df['State'].str.replace(r'(?i)^state[:\-\s]*', '', regex=True).str.strip()
    return df[df['State_clean'] == state_name]

#sample
get_state_data(irri, 'Tamil Nadu')
print(get_state_data(irri, 'Tamil Nadu'))

#conert numeric to numbers
non_numeric= ['SNo','District','State']
to_numeric_col = [c for c in irri.columns if c not in non_numeric]
print("sample",to_numeric_col)

for c in to_numeric_col:
    irri[c]= irri[c].astype(str).str.replace(',','').str.strip()
    irri[c]= pd.to_numeric(irri[c],errors='coerce')

print(irri[to_numeric_col].info())
print(irri[to_numeric_col].head())
#now take the coverted numeric col and group by stae names  to aggregate this is duplicate list
irri_numeric_cols =[c for c in to_numeric_col]
irri_state_agg= irri.groupby('State')[irri_numeric_cols].sum(min_count=1).reset_index()

print("Aggregated state sample:")
print(irri_state_agg[['State','Net_Irrigated_Area','Gross_Irrigated_Area']].head(12))

#normalise reutrn nan if nan strip the whitespaces - remove extra sapes and conver to title case
def normalise_state_name(s):
    if pd.isna(s):
        return s
    return re.sub(r'\s+', ' ', str(s).strip()).title()

crop_data['State_clean'] = crop_data['State'].astype(str).apply(normalise_state_name)
irri_state_agg['State_clean'] = irri_state_agg['State'].astype(str).apply(normalise_state_name)

print("Example crop states:", sorted(crop_data['State_clean'].unique())[:6])
print("Example irri states:", sorted(irri_state_agg['State_clean'].unique())[:6])

crop_states = set(crop_data['State_clean'].unique())
irri_states = set(irri_state_agg['State_clean'].unique())
in_crop_not_irri = sorted(list(crop_states - irri_states))
in_irri_not_crop = sorted(list(irri_states - crop_states))
print("In crop_data but NOT in irrigation:", in_crop_not_irri[:20])
print("In irrigation but NOT in crop_data:", in_irri_not_crop[:20])

merged = crop_data.merge(irri_state_agg.drop(columns=['State']),
                         on='State_clean',how='left',suffixes=('','_irri'))

merged.to_csv("merged_agri_data.csv", index=False)

print("Merged shape:", merged.shape)
print("Columns (sample):", merged.columns.tolist()[:12])
print(merged[['State','State_clean','Crop','Area-2021-22','Net_Irrigated_Area']].dropna(subset=['Net_Irrigated_Area']).head())

# compute total cropped area per state (2021-22) from crop_data
state_area_2021_22 = crop_data.groupby('State_clean')['Area-2021-22'].sum(min_count=1).reset_index()
state_area_2021_22.rename(columns={'Area-2021-22':'Total_Cropped_Area_2021_22'}, inplace=True)


#  Multi-year crop area aggregation
area_cols = ['Area-2020-21', 'Area-2021-22', 'Area-2022-23']
state_area_multi_year = crop_data.groupby('State_clean')[area_cols].sum(min_count=1).reset_index()

# Step 8: Merge with irrigation data
irri_multi = irri_state_agg.merge(state_area_multi_year, on='State_clean', how='left')

# Step 9: Compute irrigation coverage for each year
for col in area_cols:
    irri_multi[col] *= 1000  # Convert to hectares
    irri_multi[f'Irrigation_per_Ha_{col[-7:]}'] = irri_multi['Net_Irrigated_Area'] / irri_multi[col]
    irri_multi[f'Irrigation_%_{col[-7:]}'] = irri_multi[f'Irrigation_per_Ha_{col[-7:]}'] * 100


#intelligent layer 

def normalise_text(s:str)-> str:
    if s is None:
        return ""
    s =str(s).lower().strip()
    s =re.sub(r'[^a-z0-9\s\-]','',s)
    s= re.sub(r'\s+',' ',s)
    return s

CROP_CANDIDATES = [normalise_text(x)  for x in crop_data['Crop'].unique()]
STATE_CANDIDATES = [normalise_text(x) for x in crop_data['State_clean'].unique()]

def match_crop(query_text, cutoff = 0.6):
    q = normalise_text(query_text)
    for orig in crop_data['Crop'].unique():
        if normalise_text(orig) in q or q in normalise_text(orig):
            return orig
        matches = get_close_matches(q,CROP_CANDIDATES,cutoff=cutoff)
        if matches:
            matched_norm = matches[0]
            for orig in crop_data['Crop'].unique():
                if normalise_text(orig) == matched_norm:
                    return orig
                return None
            
def match_state(query_text,cutoff = 0.6):
    q = normalise_text(query_text)
    for orig in crop_data['State_clean'].unique():
        if normalise_text(orig) in q or q in normalise_text(orig):
            return orig
        matches = get_close_matches(q,STATE_CANDIDATES,cutoff=cutoff)
        if matches:
            matched_norm = matches[0]
            for orig in crop_data['State_clean'].unique():
                if normalise_text(orig)== matched_norm:
                    return orig
                return None
            
def parse_year(query_text):
    q = str(query_text)
    m = re.search(r'(20\d{2})\*s[-/]\*s(\d{2})',q)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    m2=re.search(r'(20\d{2})',q)
    if m2:
         y = int(m2.group(1))
         return f"{y}-{str(y+1)[2:]}"
    return None

print("match_crop('rice') ->", match_crop("rice"))
print("match_state('tn') ->", match_state("tn"))
print("parse_year('production in 2021-22') ->", parse_year("production in 2021-22"))

def classify_query(user_text):
    txt = normalise_text(user_text)

    if any(w in txt for w in ["area", "acre", "sown","hectare", "cultivated", "coverage", "land used"]):
        return "AREA"
    elif any(w in txt for w in [ "production", "produce", "output", "harvest", "quantity", "total production", "metric tons"]):
        return "PRODUCTION"
    elif any(w in txt for w in ["yield", "productivity", "per hectare", "efficiency", "crop yield", "output per acre"] ):
        return 'YIELD'
    elif any(w in txt for w in ["irrigation", "irrigated", "water coverage", "canal", "drip", "sprinkler", "water source", "rainfed"]):
        return"IRRIGATION"
    else:
        return "UNKNOWN"
    
def validate_query(question_type, state, crop, year):
    missing = []

    if question_type == "UNKNOWN":
        missing.append("question type (area, production, yield, irrigation)")
    if state is None:
        missing.append("state")
    if question_type != "IRRIGATION" and crop is None:
        missing.append("crop")
    if question_type != "IRRIGATION" and year is None:
        missing.append("year")

    if missing:
        return f"Please provide: {', '.join(missing)}."
    return None

# ----------  answer_query ----------
def answer_query(user_query, merged):
    question_type = classify_query(user_query)
    crop = match_crop(user_query)
    state = match_state(user_query)
    year = parse_year(user_query)

    # Validate input
    error = validate_query(question_type, state, crop, year)
    if error:
        return error

    # Normalize for matching
    state_norm = normalise_text(state)
    crop_norm = normalise_text(crop)

    # IRRIGATION questions
    if question_type == "IRRIGATION":
        result = merged[merged['State_clean'].apply(normalise_text) == state_norm]
        if result.empty:
            return f"No irrigation data available for {state}."
        irr = result.iloc[0]["Net_Irrigated_Area"]
        return f"The net irrigated area in {state} is {irr} hectares."

    # AREA / PRODUCTION / YIELD questions
    result = merged[
        (merged['State_clean'].apply(normalise_text) == state_norm) &
        (merged['Crop'].apply(normalise_text) == crop_norm)
    ]
    if result.empty:
        return f"No data found for crop '{crop}' in state '{state}'."

    if question_type == "AREA":
        col = f"Area-{year}"
        value = result.iloc[0][col]
        return f"The cultivated area for {crop} in {state} during {year} was {value} hectares."
    elif question_type == "PRODUCTION":
        col = f"Production-{year}"
        value = result.iloc[0][col]
        return f"The production of {crop} in {state} during {year} was {value} metric tons."
    elif question_type == "YIELD":
        col = f"Yield-{year}"
        value = result.iloc[0][col]
        return f"The yield of {crop} in {state} during {year} was {value} tons per hectare."
    else:
        return "Sorry, I could not understand your question."

# ----------  orchestration layer ----------
def process_query(user_question, merged):
    # Classify question, match state/crop, extract year
    question_type = classify_query(user_question)
    crop = match_crop(user_question)
    state = match_state(user_question)
    year = parse_year(user_question)

    # Validate input 
    error = validate_query(question_type, state, crop, year)
    if error:
        return error

    # Get final answer
    return answer_query(user_question, merged)

q1 = "What is the cultivated area of rice in Tamil Nadu in 2021-22?"
q2 = "Net irrigated area in Karnataka for rice in 2021?"
q3 = "Yield of wheat in Andhra Pradesh 2022?"

print(process_query(q1, merged))
print(process_query(q2, merged))
print(process_query(q3, merged))

