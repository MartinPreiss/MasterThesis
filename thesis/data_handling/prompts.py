system_prompt_valid_invalid = f"""Your task is to evaluate the factual correctness of a given answer to a question. Read the entire answer and question carefully, then analyze its factual accuracy step by step. After your analysis, provide a final verdict of either "True" if the entire answer is factually correct, or "False" if any part of the answer contains inaccuracies or hallucinations.

Evaluation Process:
1. Read the question.
2. Read the entire answer.
3. Use Chain of Thought reasoning to analyze the answer's factual accuracy.
4. Provide your reasoning steps.
5. Conclude with a final verdict: "True" or "False".

Examples:

Question: How do wind turbines generate electricity?
Answer: Wind turbines generate electricity by converting the kinetic energy of wind into electrical energy. The large blades of the turbine are designed to catch the wind, causing them to spin. This spinning motion drives a shaft connected to a generator, which uses electromagnetic induction to produce electricity. The amount of electricity generated depends on wind speed and the size of the turbine. Most modern wind turbines start generating electricity at wind speeds of about 3-4 meters per second and reach maximum power output at around 15 meters per second. Wind turbines are most efficient when grouped together in wind farms, where they can take advantage of consistent wind patterns.
Thought process:
    1. The answer correctly explains the basic principle of wind turbines converting wind's kinetic energy to electrical energy.
    2. The description of blades catching wind and spinning is accurate.
    3. The connection between the spinning motion and the generator is correctly explained.
    4. The mention of electromagnetic induction as the method of electricity production is accurate.
    5. The statement about electricity generation depending on wind speed and turbine size is correct.
    6. The wind speeds mentioned for starting generation and reaching maximum output are within the typical range for modern wind turbines.
    7. The point about wind farms being more efficient due to consistent wind patterns is generally accepted in the field.
After analyzing each part of the answer, I don't detect any inaccuracies or hallucinations. All the information provided appears to be factually correct and aligns with current understanding of wind turbine technology.
Final Verdict: True

Question: How does solar energy work?
Answer: Solar energy works by capturing light from the sun and converting it into electricity through solar panels. These panels use photovoltaic cells to absorb sunlight and generate electrical current. The process works during the day when sunlight is available, but it also continues at night by using starlight to generate energy. The energy captured by solar panels during the day can be stored in batteries to be used later, and this stored energy helps power homes and businesses even after the sun sets.
Thought process:
    1. The explanation of capturing sunlight and converting it into electricity is correct.
    2. The mention of photovoltaic cells absorbing sunlight and generating current is accurate.
    3. The claim about using starlight to generate energy at night is incorrect, as solar panels cannot generate power from starlight.
    4. The information about storing energy in batteries for later use is accurate.
    5. The description of using stored energy to power homes and businesses after sunset is correct.
While the answer provides accurate information about how solar panels function and how energy is stored for later use, the statement about generating energy at night through starlight is incorrect. Photovoltaic cells require sunlight to function, and the light from stars is insufficient to power solar panels. This factual error impacts the overall correctness of the answer.
Final Verdict: False
"""

user_prompt_valid_invalid = """Task:

Question: {question}
Answer: {answer}
Thought process: 
"""

simple_prompt = """Your task is to evaluate the factual correctness of a given answer to a question. Provide only a final verdict of either [TRUE] if the entire answer is factually correct, or [FALSE] if any part of the answer contains inaccuracies or hallucinations. 
Only output the Final Verdict. No Explanation. 
Question: {question}
Answer: {answer}
Final Verdict: The answer is ["""

one_example_prompt = """
You will get an user_question and an user_answer. Your task is to fact check the user_answer. 
So, does the User_Answer contain only factually correct statements? 
Only output True or False!

Example: 
    User_Question: Where is Berlin located ? 
    User_Answer: Berlin is located in France. 
    Output: The User_Answer is False.

User_Question: {question}

User_Answer: {answer}

Output: The User_Answer is """
