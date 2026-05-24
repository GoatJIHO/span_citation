from ..src.preprocessor import pre_process

def get_label_information(data_type:str):
    if "acl_arc" == data_type or "act2" == data_type :
        return {
            0: "BACKGROUND",
            1: "COMPARES_CONTRASTS",
            2: "EXTENSION",
            3: "FUTURE",
            4: "MOTIVATION",
            5: "USES"
        }

    if "scicite" == data_type:
        return {
            0: "METHOD",
            1: "BACKGROUND",
            2: "RESULT"
        }
    
def get_allowed_label(data_type:str):
    if "acl_arc" == data_type or "act2" == data_type :
        return {
            "BACKGROUND", "COMPARES_CONTRASTS", "EXTENSION",
            "FUTURE", "MOTIVATION", "USES"
        }

    if "scicite" == data_type:
        return {
            "METHOD", "BACKGROUND", "RESULT"
        }
    
def get_regex_label(data_type:str):
    if "acl_arc" == data_type or "act2" == data_type :
        return r"\[(BACKGROUND|COMPARES_CONTRASTS|EXTENSION|FUTURE|MOTIVATION|USES)\]"

    if "scicite" == data_type:
        return r"\[(METHOD|BACKGROUND|RESULT)\]"
        

   

def get_label_description(data_type: str):
    if data_type in ["acl_arc", "act2"]:
        return """
        The six citation function classes are:
        - [BACKGROUND] Q provides relevant information or is part of the general literature.
        - [COMPARES_CONTRASTS] P compares, contrasts, or disagrees with Q.
        - [EXTENSION] P extends data, methods, or results from Q.
        - [FUTURE] Q represents a potential direction for future work.
        - [MOTIVATION] Q directly motivates the research in P.
        - [USES] P uses tools, methods, or datasets from Q.
        """.strip()

    if data_type == "scicite":
        return """
        The three citation function classes are:
        - [METHOD] P cites Q for a method, tool, model, dataset, framework, or other technical resource used in the work.
        - [BACKGROUND] Q provides background information, general context, related work, or supporting knowledge.
        - [RESULT] P cites Q for a finding, result, observation, or conclusion reported in Q.
        """.strip()


def generate_prompt(train_type: str, data_type: str, context: str):
    if train_type == "origin_training_data":
        instruction = f"""
            You are provided a citation context from a paper P citing a paper Q, with the specific citation denoted by the token `#CITATION_TAG`. 

            Your task is to classify the citation function — i.e., the rhetorical purpose for citing Q in this context — into one of the six categories below. 

            Respond with only the label enclosed in square brackets, e.g., [BACKGROUND].

            {get_label_description(data_type)}

            Here is the citation context:

            "{context}"

            Please classify the purpose of citing `#CITATION_TAG` in this context.

            Only output one of the six labels above, enclosed in square brackets.
        """.strip()
        return instruction
        
        
    elif train_type == "new_ours":
        instruction = f"""
            You are provided a citation context from a paper P citing a paper Q, with the specific citation denoted by the token `#CITATION_TAG`.

            Your task is to:
            1) Choose ONE EVIDENCE SPAN from the context that best indicates the citation function of `#CITATION_TAG`.
            2) Classify the citation function into one of the categories below.

            EVIDENCE SPAN rules:
            - Copy a contiguous span verbatim from the context (case/punctuation preserved).
            - The span must come from the context itself and must NOT include `#CITATION_TAG`.
            - Pick the span that most strongly supports the label.

            {get_label_description(data_type)}

            ### Internal Step-by-Step Process
            Before generating the final output, you must:
            1) Scan the context and extract a contiguous EVIDENCE SPAN that clearly indicates the author's intent.
            2) Assign a temporary LABEL based on that span.
            3) **Re-evaluate the connection**: Ask yourself, "Does this specific span directly support the definition of the chosen label?" If it feels more like another category, refine both the span and the label until they are perfectly aligned.
            4) Ensure the span is exactly verbatim from the context (no paraphrasing).

            ### Output Format
            <EVIDENCE SPAN>\t[LABEL]
            
            ### Example
            Context: We adopted the architecture from ( #CITATION_TAG ) due to its efficiency...
            Output: <adopted the architecture>\t[USES]

            Here is the citation context:
            "{pre_process(context)}"

            Please provide the final EVIDENCE SPAN and LABEL after your internal re-evaluation.
            """.strip()
        
        return instruction
    
    else:
        raise ValueError(f"Invalid train_type: {train_type}, Only possible value is ['origin_training_data', 'new_ours']")
