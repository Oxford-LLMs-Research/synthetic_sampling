
import json
import os
import pandas as pd
from survey_profile_generator import SurveyProfileGenerator

#################################
# 1) load survey-specific mappings
#################################

## ESS
### 2023
directory = "mappings/ESS/2023"
ess2023 = pd.read_csv("DRIVE_LINK")

### 2020
# directory = "mappings/ESS/2020"
# ess2020 = pd.read_csv("DRIVE_LINK")


# survey_mappings = {}

# for filename in os.listdir(directory):
#     if filename.endswith('.json'):
#         section_name = os.path.splitext(filename)[0]

#         with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
#             survey_mappings[section_name] = json.load(file)



## WVS
# directory = "mappings/WVS"
# wvs = pd.read_csv("DRIVE_LINK")

# survey_mappings = {}

# with open(directory, 'r', encoding='utf-8') as file:
#     survey_mappings = json.load(file)


#################################
# 2) generate profiles
#################################

cntry_spec_vars_path = "mappings/country_specific_variables.json"

prof_generator = SurveyProfileGenerator(
    data=ess2023,
    respondent_id='idno',
    survey_mappings=survey_mappings,
    country_specific_variables=country_specific_variables['2023'],
    max_sections=4,
    max_features=3,
    fixed_features=['cntry', 'gndr', 'agea', 'essround'],
    country_field='cntry',
    random_state=42
)

profiles = prof_generator.generate_profiles(num_profiles_per_respondent=5)


#################################
# 3) save profiles
#################################

ids = []
prof_descriptions = []
for profile in profiles:
    id = profile['respondent_id']
    preambule, question, response = prof_generator.profile_to_text(profile)
    prof_text = f"Profile: \n{preambule}. \nQuestion: {question} \nResponse: {response}"

    ids.append(id)
    prof_descriptions.append(prof_text)

df = pd.DataFrame({'id': ids, 'text': prof_descriptions})
df.to_csv('profiles_ess23.csv', index=False)
