from typing import Any, Dict


def define_bloom_taxonomy_function() -> Dict[str, any]:
    """
    Returns a structured JSON schema for the function that classifies the question
    according to Bloom's taxonomy.
    """
    return {
        "name": "bloom_taxonomy_classification",
        "description": (
            "Classify a given question based on Anderson and Krathwohl’s Taxonomy of the Cognitive Domain. "
            "Categories include Remembering, Understanding, Applying, Analyzing, Evaluating, and Creating."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "statement": {
                    "type": "string",
                    "description": "Input question or statement to be classified."
                },
                "reasoning": {
                    "type": "string",
                    "description": "Step-by-step reasoning to classify the question."
                },
                "classification": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "enum": [
                                "Remembering",
                                "Understanding",
                                "Applying",
                                "Analyzing",
                                "Evaluating",
                                "Creating"
                            ],
                            "description": "The taxonomy category that best fits the question."
                        },
                        "explanation": {
                            "type": "string",
                            "description": "Brief justification for the chosen classification."
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 100,
                            "description": "Confidence level of the classification (0-100%)."
                        }
                    },
                    "required": ["category", "explanation", "confidence"]
                }
            },
            "required": ["statement", "reasoning", "classification"]
        }
    }


def define_needs_function() -> Dict[str, any]:
    # Define the function schema for the JSON response
    function = {
        "name": "categorize_question",
        "description": "Categorize the user's question into one of the primary need categories.",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "The primary need category the question falls into. One of: 'Knowledge and Information', 'Problem-Solving and Practical Skills', 'Personal Well-being', 'Professional and Social Development', 'Leisure and Creativity'."
                },
                "explanation": {
                    "type": "string",
                    "description": "A brief explanation of why this category was chosen."
                },
                "confidence_level": {
                    "type": "integer",
                    "description": "The confidence level (0-100) of the categorization.",
                    "minimum": 0,
                    "maximum": 100
                }
            },
            "required": ["category", "explanation", "confidence_level"]
        }
    }
    return function


def define_uniqueness_function() -> Dict[str, any]:
    """
    Returns a structured JSON schema for the function that classifies the question based on uniqueness.
    """
    return {
        "name": "uniqueness_classification",
        "description": (
            "Classify a given question based on the uniqueness of the answer. "
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "statement": {
                    "type": "string",
                    "description": "Input question or statement to be classified."
                },
                "reasoning": {
                    "type": "string",
                    "description": "Step-by-step reasoning to classify the question."
                },
                "classification": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "enum": [
                                "Multiple Valid Answers",
                                "Unique Answer"
                            ],
                            "description": "The category that best fits the question."
                        },
                        "explanation": {
                            "type": "string",
                            "description": "Brief justification for the chosen classification."
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 100,
                            "description": "Confidence level of the classification (0-100%)."
                        }
                    },
                    "required": ["category", "explanation", "confidence"]
                }
            },
            "required": ["statement", "reasoning", "classification"]
        }
    }


def define_domain_function() -> Dict[str, any]:
    """
    Returns a structured JSON schema for the function that classifies the question based on domain.
    """
    return {
        "name": "domain_classification",
        "description": (
            "Classify a given question based on the knowledge domain it belongs to. "
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "statement": {
                    "type": "string",
                    "description": "Input question or statement to be classified."
                },
                "reasoning": {
                    "type": "string",
                    "description": "Step-by-step reasoning to classify the question."
                },
                "classification": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "enum": [
                                "Natural and Formal Sciences",
                                "Society, Economy, Business",
                                "Health and Medicine",
                                "Computer Science and Technology",
                                "Psychology and Behavior",
                                "Historical Events and Hypothetical Scenarios",
                                "Everyday Life and Personal Choices",
                                "Arts and Culture"
                            ],
                            "description": "The category that best fits the question."
                        },
                        "explanation": {
                            "type": "string",
                            "description": "Brief justification for the chosen classification."
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 100,
                            "description": "Confidence level of the classification (0-100%)."
                        }
                    },
                    "required": ["category", "explanation", "confidence"]
                }
            },
            "required": ["statement", "reasoning", "classification"]
        }
    }


def define_causality_function() -> Dict[str, any]:
    """
    Returns a structured JSON schema for the function that classifies the question based on causality.
    """
    return {
        "name": "causality_classification",
        "description": (
            "Classify a given question based on whether it is causal or non-causal."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "statement": {
                    "type": "string",
                    "description": "Input question or statement to be classified."
                },
                "reasoning": {
                    "type": "string",
                    "description": "Step-by-step reasoning to classify the question."
                },
                "classification": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "enum": [
                                "Causal",
                                "Non-causal"
                            ],
                            "description": "The category that best fits the question."
                        },
                        "explanation": {
                            "type": "string",
                            "description": "Brief justification for the chosen classification."
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 100,
                            "description": "Confidence level of the classification (0-100%)."
                        }
                    },
                    "required": ["category", "explanation", "confidence"]
                }
            },
            "required": ["statement", "reasoning", "classification"]
        }
    }


def get_prompt_needs(question_data: Dict[str, str]) -> str:
    if isinstance(question_data, dict):
        question = question_data['shortened_query']
    else:
        question = question_data

    prompt = f"""Analyze the following question and identify the primary user need category it falls into. Consider the broad categories of user needs as defined below:

Knowledge and Information: Seeking factual information, understanding concepts, or exploring ideas. Questions falling in this category are looking for knowledge for its own sake, as far as we can infer from the question. We do not see an underlying need of the user except for curiosity.
Problem-Solving and Practical Skills: Troubleshooting issues, learning new skills, managing daily life, and handling technology. Anything that is actionable, how to do something or solve a problem. 
Personal Well-being: Improving mental and physical health, managing finances, and seeking support.
Professional and Social Development: Advancing career, job searching, and improving social interactions. Any information request about work, school, academia (but that is not actionable, in that case it’s 2. Problem-Solving and Practical Skills). 
Leisure and Creativity: Finding recreational activities, pursuing hobbies, and seeking creative inspiration.
Categorize this question based on the user's primary need asked in the question, choosing among the categories above. 

Examples

1. Q: "What is the chemical symbol for gold?"
    A: Knowledge and Information
2. Q: "How do I fix a leaky faucet?"
    A: Problem-Solving and Practical Skills
3. Q: "What does the paracetamol do to the body?"
    A: Personal Well-being
4. Q: "Create a story about a magical kingdom."
    A: Leisure and Creativity
5. Q: "How do I improve my resume?"
    A: Professional and Social Development

Question: "{question}"
"""
    return prompt

def get_prompt_uniqueness(question_data: Dict[str, str]) -> str:

    if isinstance(question_data, dict):
        question = question_data['shortened_query']
    else:
        question = question_data
    
    prompt = f"""You are tasked with classifying answerable questions based on the uniqueness of their answers. This classification helps understand the nature of the question and the potential diversity of valid responses.

For answerable questions:
1. Unique Answer: There is a single, specific correct answer that is widely accepted based on current human knowledge. Questions about fictional characters are most likely "Unique answer" because most of the times we can assume the answer is in the book / movie
2. Multiple Valid Answers: There are several plausible, valid answers that could be considered correct depending on perspective, context, or interpretation. Multiple valid answers means either (1) there is a subjective judgment involved (the answer varies depending on who answers), or (2) it is a creative task that can be solved in several different ways, or (3) we as humans do not have access to a unique correct answer. Trivial different wordings of the same concept do not qualify as "Multiple answers". e.g. "What's the meaning of bucolic?"

Ambiguous questions are not  necessarily "multiple answers" just because they are ambiguous (i.e. since different people could understand the query differently, they might give different answers). In such cases we should assume a meaning for the question, and reason about the possible ways to answer it.

For each given hypothetically answerable question, classify it as either:
1. Unique Answer
2. Multiple Valid Answers

If the question is looking for a list of items, but the list is unique and well defined, it should be classified as "Unique Answer". 
If the question is looking for a number which can be found or computed, it should be classified as "Unique Answer".
If the question is asking how something can be achieved, and there is only one way to achieve it, it should be classified as "Unique Answer", and "Multiple Valid Answers" otherwise.

Examples

1. Q: "What is the chemical symbol for gold?"
  Classification: Unique Answer
  Explanation: There is a single, universally accepted answer in chemistry: Au.

2. Q: "What is the best programming language for web development?"
  Classification: Multiple Valid Answers
  Explanation: There are several programming languages suitable for web development, and the "best" can depend on project requirements, developer preference, and other factors.

3. Q: "Who was the first president of the United States?"
  Classification: Unique Answer
  Explanation: There is a single, historically accepted answer: George Washington.

4. Q: "Tell me some short bedtime stories"
  Classification: Multiple Valid Answers
  Explanation: There are various short stories that can be told at bedtime, and the choice can vary based on cultural background, personal preference, and other factors.

5. Q: "What is the meaning of life?"
    Classification: Multiple Valid Answers
    Explanation: This question has multiple valid answers based on different philosophical, religious, and personal perspectives.

6 Q: "What does it mean when an economy is in a recession?"
    Classification: Unique Answer
    Explanation: There is a specific definition of a recession in economics, making this a question with a unique answer.

7 Q: "Name the three primary colors"
    Classification: Unique Answer
    Explanation: There are three primary colors in the RGB color model: red, green, and blue.

8 Q: "What criteria should I consider when buying a new laptop?"
    Classification: Multiple Valid Answers
    Explanation: The criteria for buying a laptop can vary based on individual needs, preferences, and budget constraints.

9. Q: "Who is the best neurosurgeon in New York?"
    Classification: Multiple Valid Answers
    Explanation: The best neurosurgeon can vary based on specialization, patient reviews, and other factors.



Please classify the following question:

{question}
"""
    return prompt

def get_prompt_bloom_taxonomy(question_data: Dict[str, str]) -> str:
    # if it is a dict, take the shortened_query key, else if it is a sstr take the value
    if isinstance(question_data, dict):
        question = question_data['shortened_query']
    else:
        question = question_data


    prompt = f"""
    Your task is to classify given statements or questions according to Anderson and Krathwohl’s Taxonomy of the Cognitive Domain. Use the following six categories and their descriptions:

Remembering: Recognizing or recalling knowledge from memory. Remembering is when memory is used to produce or retrieve definitions, facts, or lists, or to recite previously learned information. Factual questions, that do not require reasoning fall into this category. 
Understanding: Constructing meaning from different types of functions be they written or graphic messages or activities like interpreting, exemplifying, classifying, summarizing, inferring, comparing, or explaining. Questions asking for the meaning or explanation of a concept fall into this category.
Applying: Carrying out or using a procedure through executing, or implementing. Applying relates to or refers to situations where learned material is used, applied in a concrete situation, is used to present or show something. Questions that require the application of some theory or rule. For example, requiring some calculation, formula, light reasoning, applied to something in the real world. They do not entail a creative effort, but instead applying some rule or principle. Asking to generate a code with a specific goal (e.g. a cmd code that does yyy) is "apply" whereas asking to build a website requires a creative effort. How to do something, how to make something, how to solve something, how to apply some principle etc.
Analyzing: Breaking materials or concepts into parts, determining how the parts relate to one another, or how the parts relate to an overall structure or purpose. Mental actions included in this function are differentiating, organizing, and attributing, as well as being able to distinguish between the components or parts. When one is analyzing, he/she can illustrate this mental function by creating spreadsheets, surveys, charts, or diagrams, or graphic representations. Questions requiring deeper, more complex considerations on a certain thing. e.g. considering several aspects of something, considering pros and cons etc. Explaining why something is the way it is, by providing evidence or logical reasoning.
Evaluating: Making judgments based on criteria and standards through checking and critiquing. Critiques, recommendations, and reports are some of the products that can be created to demonstrate the processes of evaluation. Evaluating comes before creating as it is often a necessary part of the precursory behavior before one creates something. Questions asking to make judgements, suggestions, recommendations. Also making an hypothesis about something uncertain. Judging whether something is better than something else, or the best. 
Creating: Putting elements together to form a coherent or functional whole; reorganizing elements into a new pattern or structure through generating, planning, or producing. Creating requires users to put parts together in a new way, or synthesize parts into something new and different creating a new form or product. This process is the most difficult mental function in the taxonomy. Questions asking for generation tasks that require a creative effort fall into this category. 

Examples

1. Q: What does the term 'photosynthesis' mean?
    Classification: Understanding
    Explanation: The question asks for the meaning of a term, which falls under the 'Understanding' category.
2. Q: Calculate the area of a circle with a radius of 5 meters.
    Classification: Applying
    Explanation: The question requires applying a formula to calculate the area of a circle, which falls under the 'Applying' category.
3. Q: Compare and contrast the advantages and disadvantages of renewable energy sources.
    Classification: Evaluating
4. Q: Design a new logo for a tech startup company.
    Classification: Creating
5. Q: Explain the causes of World War II.
    Classification: Analyzing
6. Q: What category does the word 'dog' belong to?
    Classification: Remembering
7. Q: Is surfing easier to learn than snowboarding?
    Classification: Evaluating

Please classify the following question:

{question}
"""
    return prompt


def get_prompt_domain(question_data: Dict[str, str]) -> str:
    if isinstance(question_data, dict):
        question = question_data['shortened_query']
    else:
        question = question_data

    prompt = f"""Below you’ll find a question. Classify it in one of the following categories:

1. Natural and Formal Sciences: This category encompasses questions related to the physical world and its phenomena, including, but not limited to, the study of life and organisms (Biology), the properties and behavior of matter and energy (Physics), and the composition, structure, properties, and reactions of substances (Chemistry); also formal sciences belong to this category, such as Mathematics and Logic. Questions in this category seek to understand natural laws, the environment, and the universe at large.
2. Society, Economy, Business: Questions in this category explore the organization and functioning of human societies, including their economic and financial systems. Topics may cover Economics, Social Sciences, Cultures and their evolution, Political Science and Law. Questions regarding business, sales, companies’ choices and governance fall into this category. 
3. Health and Medicine: This category focuses on questions related to human health, diseases, and the medical treatments used to prevent or cure them. It covers a wide range of topics from the biological mechanisms behind diseases, the effectiveness of different treatments and medications, to strategies for disease prevention and health promotion. It comprises anything related or connected to human health.
4. Computer Science and Technology: Questions in this category deal with the theoretical foundations of information and computation, along with practical techniques for the implementation and application of these foundations. Topics include, but are not limited to, theoretical computer science, coding and optimization, hardware and software technology and innovation in a broad sense. This category includes the development, capabilities, and implications of computing technologies.
6. Psychology and Behavior: This category includes questions about the mental processes and behaviors of humans. Topics range from understanding why people engage in certain behaviors, like procrastination, to the effects of social factors, and the developmental aspects of human psychology, such as language acquisition in children. The focus is on understanding the workings of the human mind and behavior in various contexts, also in personal lives.
7. Historical Events and Hypothetical Scenarios: This category covers questions about significant past events and their impact on the world, as well as hypothetical questions that explore alternative historical outcomes or future possibilities. Topics might include the effects of major wars on global politics, the potential consequences of significant historical events occurring differently, and projections about future human endeavors, such as space colonization. This category seeks to understand the past and speculate on possible futures or alternative historical happenings.
8. Everyday Life and Personal Choices: Questions in this category pertain to practical aspects of daily living and personal decision-making. Topics can range from career advice, cooking tips, and financial management strategies to advice on maintaining relationships and organizing daily activities. This category aims to provide insights and guidance on making informed choices in various aspects of personal and everyday life. Actionable tips fall into this category. 
9. Arts and Culture: This category includes topics in culture across various mediums such as music, television, film, art, games, and social media, sports, celebrities. 

Assign one of the above categories to the given question. 

Question: {question}
"""    
    return prompt





def get_prompt_causality(question_data: Dict[str, str]) -> str:
    if isinstance(question_data, dict):
        question = question_data['shortened_query']
    else:
        question = question_data

    """
    Generate a prompt for causal classification. Last iteration of prompt engineering. Added preliminary concepts
    """
    prompt= f"""
        The following is a question that a human asked online. Classify the question in one of the following two categories: 

        Category 1: Causal. This category includes questions that suggest a cause-and-effect relationship broadly speaking, requiring the use of cause-and-effect knowledge or reasoning to provide an answer.
        A cause is a preceding thing, event, or person that contributes to the occurrence of a later thing or event, to the extent that without the preceding one, the later one would not have occurred.
        A causal question can have different mechanistic natures. It can be:
        1. Given the cause, predict the effect: seeking to understand the impact or outcome of a specific cause, which might involve predictions about the future or hypothetical scenarios.
        2. Given the effect, predict the cause: asking "Why" something occurs (e.g. Why do apples fall?), probing the cause of a certain effect, asking about the reasons behind something, or the actions needed to achieve a specific result or goal, "How to" do something, explicitly or implicitly (e.g. Why does far right politics rise nowadays? How to earn a million dollars? How to learn a language in 30 days?). 
        This also includes the cases in which the effect is not explicit: any request with a purpose, looking for ways to fulfill it. It means finding the action (cause) to best accomplish a certain goal (effect), and the latter can also be implicit. If someone asks for a restaurant recommendation, what she’s looking for is the best cause for a certain effect which can be, e.g., eating healthy. If asking for a vegan recipe, she’s looking for the recipe that causes the best possible meal. Questions asking for “the best” way to do something, fall into this category. 
        Asking for the meaning of something that has a cause, like a song, a book, a movie, is also causal, because the meaning is part of the causes that led to the creation of the work. A coding task which asks for a very specific effect to be reached, is probing for the cause (code) to obtain that objective. 
        3. Given variables, judge their causal relation: questioning the causal link among the given entities (e.g. Does smoking cause cancer? Did my job application get rejected because I lack experience?)
        Be careful: causality might also be implicit! Some examples of implicit causal questions: 
        - the best way to do something
        - how to achieve an effect
        - what’s the effect of an action (which can be in the present, future or past)
        - something that comes as a consequence of a condition (e.g. how much does an engineer earn, what is it like to be a flight attendant)
        - when a certain condition is true, does something happen?
        - where can I go to obtain a certain effect?
        - who was the main cause of a certain event, author, inventor, founder?
        - given an hypothetical imaginary condition, what would be the effect?
        - what’s the feeling of someone after a certain action?
        - what’s the code to obtain a certain result?
        - when a meaning is asked, is it because an effect was caused by a condition (what’s the meaning of <effect>)?
        - the role, the use, the goal of an entity, an object, is its effect

        Category 2: Non-causal. This category encompasses questions that do not imply in any way a cause-effect relationship.

        Let's think step by step.
        Question: {question}
        """
    return prompt


def get_prompt_causality_types(question_data: Dict[str, str]) -> str:
    if isinstance(question_data, dict):
        question = question_data['shortened_query']
    else:
        question = question_data

    prompt = f"""
You are tasked with classifying causal questions into one of four types based on the following taxonomy. Carefully read each type's definition and examples before proceeding to classify the given question. Provide a brief explanation for your classification.

### Causal Question Types:

#### Type 1: About Variables

- Definition: Questions that ask about the variables in the causal graph. They inquire about the causes or contributing factors of a phenomenon.
- Characteristics:
  - Seek to identify variables that play a role in causing an effect.
  - Often start with "What are the causes of..." or "What factors contribute to..."
- Examples:
  - "What are the causes of fire?"
  - "What nutrition do athletes need?"
  - "What factors contribute to climate change?"

#### Type 2: About Relations

- Definition: Questions that ask about the existence of directed edges (causal relationships) among variables. They inquire whether one variable directly affects another.
- Characteristics:
  - Investigate if a causal link exists between specific variables.
  - Often phrased as "Does X cause Y?" or "Is there a relationship between X and Y?"
- Examples:
  - "Does smoking interfere with the drug effect?"
  - "Was my application rejected due to my lack of work experience?"
  - "Does stress lead to heart disease?"

#### Type 3: About Average Effects

- Definition: Questions that ask about the quantification of the average change in an effect given a cause. They often involve measuring the magnitude of an effect.
- Characteristics:
  - Focus on the extent or degree to which one variable affects another on average.
  - Often include phrases like "How much," "To what extent," or "What is the effect size of..."
- Examples:
  - "How much do COVID vaccines decrease hospitalization risk?"
  - "Are small or big classrooms better for kids?"
  - "What is the average improvement in test scores due to tutoring?"

#### Type 4: About Mechanisms

- Definition: Questions that ask about the functions or mechanisms among variables, or require understanding the underlying processes. This includes counterfactual questions.
- Characteristics:
  - Delve into how and why a causal relationship occurs.
  - May involve hypothetical or counterfactual scenarios.
  - Often start with "How does..." or "What would happen if..."
- Examples:
  - "Had I not done a PhD, would my life be different?"
  - "How do scientists prepare rockets for missions to the Moon?"
  - "What is the biochemical process by which insulin regulates blood sugar?"

---

### Instructions:

1. Read the Question Carefully: Understand what the question is asking.
2. Think Step by Step: Analyze the question to identify its underlying causal nature.
3. Match with Definitions and Characteristics: Compare the question with the definitions above.
4. Provide Classification: Assign the question to one of the four types.
5. Explain Your Reasoning: Briefly justify why the question fits that type.

---

### Example Classification:

- Question: "Does regular exercise improve mental health?"
- Analysis:
  - The question is asking about the existence of a causal relationship between regular exercise and mental health.
  - It inquires whether one variable (exercise) affects another (mental health).
- Classification: Type 2: About Relations
- Explanation: The question seeks to determine if there is a causal link between two variables.

---

Now, please classify the following question:

Causal Question: "{question}"

"""
    return prompt


def get_prompt_causality_types_it2(question_data: Dict[str, str]) -> str:
    if isinstance(question_data, dict):
        question = question_data['shortened_query']
    else:
        question = question_data

    prompt = f"""
You are tasked with classifying causal questions into one of four types based on the following taxonomy. Carefully read each type's definition and examples before proceeding to classify the given question. Provide a brief explanation for your classification.


Causal Question Types:
Type 1: About Variables
Definition:
- Questions that ask about the variables in the causal graph. E.g., they inquire about the causes or contributing factors of a phenomenon.
Characteristics:
 - Seek to identify variables that play a role in causing an effect in a specific case or  scenario.
 - Often start with "Why did X ..." or "Where does X come from ...", “What will happen if…”
Examples:
"Why did company X close in Europe?”
“What will happen if Y gets elected?”
Type 2: About Relations
Definition: 
Questions that ask about the existence of directed edges (causal relationships) among variables. They inquire whether one variable directly affects another.
Characteristics:
 - Investigate if a causal link exists between specific variables.
 - Often phrased as "Does X cause Y?" or "Is there a relationship between X and Y?"
Examples:
 - "Does smoking interfere with the drug effect?"
 - "Was my application rejected due to my lack of work experience?"
 - "Does stress lead to heart disease?"


Type 3: About Average Effects
Definition: 
Questions that ask about the quantification of the average change in an effect given a cause. They often involve measuring the magnitude of an effect.
Characteristics:
 - Focus on the extent or degree to which one variable affects another on average.
 - Often include phrases like "How much," "To what extent," or "What is the effect size of..."
Examples:
 - "How much do COVID vaccines decrease hospitalization risk?"
 - "Are small or big classrooms better for kids?"
 - "What is the average improvement in test scores due to tutoring?"


Type 4: About Mechanisms
Definition: 
Questions that ask about the functions or mechanisms among variables, or require understanding the underlying processes. This includes counterfactual questions. 
Characteristics:
 - These are questions about a causal relation in a phenomenon which holds always, or most of the time
 - May involve hypothetical or counterfactual scenarios.
 - Often start with "How does..." or "What would happen if...", “Why does X happen?”, “How can someone achieve Y”, “What are the causes of”
Examples:
 - "Had I not done a PhD, would my life be different?"
 - "How do scientists prepare rockets for missions to the Moon?"
 - "What is the biochemical process by which insulin regulates blood sugar?"


---


Instructions:


1. Read the Question Carefully: Understand what the question is asking.
2. Think Step by Step: Analyze the question to identify its underlying causal nature.
3. Match with Definitions and Characteristics: Compare the question with the definitions above.
4. Provide Classification: Assign the question to one of the four types.
5. Explain Your Reasoning: Briefly justify why the question fits that type.


---


Example Classification:


- Question: "Does regular exercise improve mental health?"
- Analysis:
 - The question is asking about the existence of a causal relationship between regular exercise and mental health.
 - It inquires whether one variable (exercise) affects another (mental health).
- Classification: Type 2: About Relations
- Explanation: The question seeks to determine if there is a causal link between two variables.


—


To recap, remember the following: 
Variables = when the question wants to know the “content” of a node of the causal graph, e.g. a cause or an effect, or another node like a mediator, in a specific case. It is asking for an element of a graph in a one-time phenomenon, like why did X happen, how did Y happen. 
Edge = question about the existence or not of a causal relationship.
Avg effect = question about quantification of a causal effect. 
Mechanism =  question about how a phenomenon works in general - so once again about the content of the nodes / structure of the graph but not of a specific case, but for something that is always in the same way, governs how things happen. It is asking for what normally, usually happens - what happens if I do X, why does X happen, How does X happen, why would X happen, where can X happen, how can X be achieved. 



Now, please classify the following question:


Causal Question: "{question}"

"""
    return prompt


def get_prompt_causality_types_it3(question_data: Dict[str, str]) -> str:
    if isinstance(question_data, dict):
        question = question_data['shortened_query']
    else:
        question = question_data

    prompt = f"""
You are tasked with classifying causal questions into one of four types based on the following taxonomy. Carefully read each type's definition and examples before proceeding to classify the given question. Provide a brief explanation for your classification.


Causal Question Types:
Type 1: About Variables
Definition:
- Questions that ask about the variables in the causal graph. E.g., they inquire about the causes or contributing factors of a phenomenon.
Characteristics:
 - Seek to identify variables that play a role in causing an effect in a specific case or  scenario.
 - Often start with "Why did X ..." or "Where does X come from ...", “What will happen if…”
Examples:
"Why did company X close in Europe?”
“What will happen if Y gets elected?”
Type 2: About Relations
Definition: 
Questions that ask about the existence of directed edges (causal relationships) among variables. They inquire whether one variable directly affects another.
Characteristics:
 - Investigate if a causal link exists between specific variables.
 - Often phrased as "Does X cause Y?" or "Is there a relationship between X and Y?"
Examples:
 - "Does smoking interfere with the drug effect?"
 - "Was my application rejected due to my lack of work experience?"
 - "Does stress lead to heart disease?"


Type 3: About Average Effects
Definition: 
Questions that ask about the quantification of the average change in an effect given a cause. They often involve measuring the magnitude of an effect.
Characteristics:
 - Focus on the extent or degree to which one variable affects another on average.
 - Often include phrases like "How much," "To what extent," or "What is the effect size of..."
Examples:
 - "How much do COVID vaccines decrease hospitalization risk?"
 - "Are small or big classrooms better for kids?"
 - "What is the average improvement in test scores due to tutoring?"


Type 4: About Mechanisms
Definition: 
Questions that ask about the functions or mechanisms among variables, or require understanding the underlying processes. This includes counterfactual questions. 
Characteristics:
 - These are questions about a causal relation in a phenomenon which holds always, or most of the time
 - May involve hypothetical or counterfactual scenarios.
 - Often start with "How does..." or "What would happen if...", “Why does X happen?”, “How can someone achieve Y”, “What are the causes of”
Examples:
  - "Had I not done a PhD, would my life be different?"
 - "How do scientists prepare rockets for missions to the Moon?"
 - "What is the biochemical process by which insulin regulates blood sugar?"
- “What makes a good doctor?”
- “Why are some people X?”
- “How can someone achieve Y?”
- “What are the causes of Z?”
---


Instructions:


1. Read the Question Carefully: Understand what the question is asking.
2. Think Step by Step: Analyze the question to identify its underlying causal nature.
3. Match with Definitions and Characteristics: Compare the question with the definitions above.
4. Provide Classification: Assign the question to one of the four types.
5. Explain Your Reasoning: Briefly justify why the question fits that type.


---


Example Classification:


- Question: "Does regular exercise improve mental health?"
- Analysis:
 - The question is asking about the existence of a causal relationship between regular exercise and mental health.
 - It inquires whether one variable (exercise) affects another (mental health).
- Classification: Type 2: About Relations
- Explanation: The question seeks to determine if there is a causal link between two variables.


—


To recap, remember the following: 
Variables = when the question wants to know the “content” of a node of the causal graph, e.g. a cause or an effect, or another node like a mediator, in a specific case. It is asking for an element of a graph in a one-time phenomenon, like why did X happen, how did Y happen. 
Edge = question about the existence or not of a causal relationship.
Avg effect = question about quantification of a causal effect. 
Mechanism =  question about how a phenomenon works in general - so once again about the content of the nodes / structure of the graph but not of a specific case, but for something that is always in the same way, governs how things happen. It is asking for what normally, usually happens - what happens if I do X, why does X happen, How does X happen, why would X happen, where can X happen, how can X be achieved. 



Now, please classify the following question:


Causal Question: "{question}"

"""
    return prompt




def define_causality_types_function() -> Dict[str, any]:
    """
    Returns a structured JSON schema for the function that classifies the question based on causality types.
    """
    return {
        "name": "categorize_causal_question",
        "description": (
            "Classify a given question based on the type of causal relationship it represents."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "statement": {
                    "type": "string",
                    "description": "Input question or statement to be classified."
                },
                "reasoning": {
                    "type": "string",
                    "description": "Step-by-step reasoning to classify the question."
                },
                "classification": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "enum": [
                                "About Variables",
                                "About Relations",
                                "About Average Effects",
                                "About Mechanisms"
                            ],
                            "description": "The type of causal question that best fits the input."
                        },
                        "explanation": {
                            "type": "string",
                            "description": "Brief justification for the chosen classification."
                        },
                        "confidence_level": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 100,
                            "description": "Confidence level of the classification (0-100%)."
                        }
                    },
                    "required": ["category", "explanation", "confidence_level"]
                }
            },
            "required": ["statement", "reasoning", "classification"]
        }
    }