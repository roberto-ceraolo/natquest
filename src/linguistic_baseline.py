import pandas as pd
import re
import nltk
from nltk import pos_tag, word_tokenize

# Ensure NLTK data is downloaded
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

# Function to classify a question as causal or non-causal
def classify_causation_question(question):
    patterns = [
        (r'\bwhy\b', 'R1'),
        (r'\bcause(?:s)?\b', 'R2'),
        (r'\bhow come\b|\bhow did\b', 'R3'),
        (r'\beffect(?:s)?\b|\baffect(?:s)?\b', 'R4'),
        (r'\blead(?:s)? to\b', 'R5'),
        (r'what (?:will|might)? happen(?:s)? (?:if|when)', 'R6'),
        (r'what (?:to do|should be done) (?:if|to|when)', 'R7')
    ]
    
    question = question.lower()
    
    for pattern, label in patterns:
        if re.search(pattern, question):
            return True
    
    return False

causal_lexico_syntactic = {
    'give rise', 'give rise to', 'induce', 'produce', 'generate', 'effect', 'bring about', 
    'provoke', 'arouse', 'elicit', 'lead', 'lead to', 'trigger', 'derive', 'derive from', 'associate', 'associate with', 'relate', 'relate to',
    'link', 'link to', 'stem', 'stem from', 'originate', 'bring forth', 'lead up', 'trigger off', 'bring on', 'result', 'result from',
    'stir up', 'entail', 'contribute to', 'set up', 'trigger off', 'commence', 'set off', 'set in motion', 'bring on', 'conduce', 'conduce to',  'educe',
    'originate in', 'lead off', 'spark', 'spark off', 'evoke', 'link up', 'implicate', 'implicate in', 'activate', 'actuate',
    'kindle', 'fire up', 'create', 'launch', 'develop', 'bring', 'stimulate', 'call forth', 'unleash', 'effectuate', 'kick up', 'give birth', 'give birth to', 'call down',
    'put forward', 'cause', 'start', 'make', 'begin', 'rise'
}

def check_causal_patterns(text):
    text = text.lower()
    for pattern in causal_lexico_syntactic:
        if re.search(r'\b' + re.escape(pattern) + r'\b', text):
            return True
    return False

def check_end_ify_verbs(text):
    words = word_tokenize(text)
    tagged_words = pos_tag(words)
    
    end_pattern = re.compile(r'\w+end$', re.IGNORECASE)
    ify_pattern = re.compile(r'\w+ify$', re.IGNORECASE)
    
    for word, tag in tagged_words:
        if tag.startswith('VB'):
            if end_pattern.match(word) or ify_pattern.match(word):
                return True
    return False

# Load the CSV file
df = pd.read_csv('causalquest.csv')

# Apply the functions to each question
df['morph_causative'] = df['shortened_query'].apply(check_end_ify_verbs)
df['lexico_synt_pattern'] = df['shortened_query'].apply(check_causal_patterns)
df['causalqa_rule'] = df['shortened_query'].apply(classify_causation_question)

# Function to calculate percentages
def calculate_percentages(group):
    total = len(group)
    return pd.Series({
        'morph_causative': (group['morph_causative'].sum() / total) * 100,
        'lexico_synt_pattern': (group['lexico_synt_pattern'].sum() / total) * 100,
        'causalqa_rule': (group['causalqa_rule'].sum() / total) * 100
    })

# Calculate overall percentages
overall_percentages = calculate_percentages(df)

# Calculate percentages for causal and non-causal groups
causal_percentages = calculate_percentages(df[df['is_causal'] == True])
non_causal_percentages = calculate_percentages(df[df['is_causal'] == False])

# Create the result table
result_table = pd.DataFrame({
    'contains morph causative (-en and -ify)': [
        f"{overall_percentages['morph_causative']:.2f}%",
        f"{causal_percentages['morph_causative']:.2f}%",
        f"{non_causal_percentages['morph_causative']:.2f}%"
    ],
    'contains lexico synt pattern (list found by Giriju)': [
        f"{overall_percentages['lexico_synt_pattern']:.2f}%",
        f"{causal_percentages['lexico_synt_pattern']:.2f}%",
        f"{non_causal_percentages['lexico_synt_pattern']:.2f}%"
    ],
    'match a rule from CausalQA (regex)': [
        f"{overall_percentages['causalqa_rule']:.2f}%",
        f"{causal_percentages['causalqa_rule']:.2f}%",
        f"{non_causal_percentages['causalqa_rule']:.2f}%"
    ]
}, index=['overall', 'causal (original label)', 'noncausal (original label)'])

# Print the result table
print(result_table.to_string())

# Optionally, save the result table to a CSV file
result_table.to_csv('causal_analysis_results.csv')

# Print some additional information
print(f"\nTotal number of questions: {len(df)}")
print(f"Number of causal questions (original label): {len(df[df['is_causal'] == True])}")
print(f"Number of non-causal questions (original label): {len(df[df['is_causal'] == False])}")