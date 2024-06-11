import json
from transformers import AutoTokenizer
from huggingface_hub import login
from typing import (Literal, Sequence,TypedDict)

### --- Template format -----------------------
Role = Literal["system", "user", "assistant"]

class Message(TypedDict):
    role: Role
    content: str

Dialog = Sequence[Message]
### ------------------------------------------

class ChatTemplate:
    def __init__(self, model, token, cache_dir):
        # login
        login(token)

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model, cache_dir=cache_dir)
        

    def apply_chat_template(self, dataset, add_generation_prompt=False):
        system_message = "You are a lexicographer familiar with providing concise definitions of word meanings."
        # Always answer by provide identical sense definitions for similar usages of a word, and distinct sense definitions for different usages. Infer the meaning of the word by considering the context in which it is used. 
        
        template = 'Please provide a concise definition for the meaning of the word "{}" in the following sentence: {}'
                    
        def apply_chat_template_func(record):
                        
            # conversation
            if add_generation_prompt is False:
                dialog: Dialog = (Message(role='system', content=system_message),
                                  Message(role='user', content=template.format(record['target'], record['example'])),
                                  Message(role='assistant', content=record['gloss']))
            else:
                dialog: Dialog = (Message(role='system', content=system_message),
                                  Message(role='user', content=template.format(record['target'], record['example'])))
                
            # generate prompt
            prompt = self.tokenizer.decode(self.tokenizer.apply_chat_template(dialog, add_generation_prompt=add_generation_prompt))
            return {'text': prompt}
        
        return dataset.map(apply_chat_template_func)
