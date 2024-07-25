import os
import re
import json
import torch
import openai
import argparse
import tiktoken
import logging
import transformers
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

model_list = [
    'llama-3-8b', 
    'llama-3-70b', 
    'gpt-3.5-turbo-0125', 
    'gpt-3.5-turbo', 
    'gpt-4-turbo', 
    'gpt-4-turbo-2024-04-09',
    'gpt-4o', 
    'gpt-4o-2024-05-13',
]

OPEN_API_KEY = os.environ.get('OPENAI_API_KEY')
OPEN_API_URL = os.environ.get('OPEN_API_URL')
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
DEVICE_ID = 0

client = None
pipeline = None


def num_tokens_from_message(messages, model='gpt-3.5-turbo'):
    ''' Return the number of tokens used by a list of messages. '''

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        logging.warning('model not found. Using gpt-3.5-turbo encoding.')
        encoding = tiktoken.get_encoding('gpt-3.5-turbo')
    
    tokens_per_message = 3
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == 'role':
                num_tokens += tokens_per_message
    num_tokens += 3 # every reply is primed with <|start|>assistant<|message|>

    return num_tokens


def get_prompt(item: dict, prompt_dict: dict, psg_num: int=5, doc_key: str='text'):
    ''' generate prompt '''

    if all(key in prompt_dict for key in ['instruction', 'demo_sep', 'demo_prompt', 'doc_prompt']):

        if all(key in item for key in ['question', 'docs']):
            prompt = prompt_dict['demo_prompt'].replace('{INST}', prompt_dict['instruction'])
            prompt = prompt.replace('{Q}', item['question'])

            doc_prompt_list = []
            for doc_id, doc in enumerate(item['docs'][:psg_num]):
                doc_key = doc_key if doc_key in doc else 'text'

                doc_prompt = prompt_dict['doc_prompt'].replace('{ID}', str(doc_id + 1))
                doc_prompt = doc_prompt.replace('{T}', doc['title'])
                doc_prompt = doc_prompt.replace('{P}', doc[doc_key])
                doc_prompt_list.append(doc_prompt)
            
            prompt = prompt.replace('{D}', prompt_dict['demo_sep'].join(doc_prompt_list))
            prompt = prompt.replace('{A}', '').rstrip()

            return prompt
        else:
            logging.error('The data item has some mistakes.')
            return None
    else:
        logging.error('The prompt config has some mistakes.')
        return None


def get_messages(
        prompt: str, 
        instruction: str='You are ChatGPT, a large language model trained by OpenAI.'
    ) -> list:
    ''' generate messages '''
    
    messages = [
        {'role': 'system', 'content': instruction},
        {'role': 'user', 'content': prompt}
    ]

    return messages


def generate_openai(messages: list, model_name: str, temperature: float=0.5) -> str:
    ''' using api to get openai model response '''

    global client

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
        )

        return completion.choices[0].message.content, completion.usage.prompt_tokens # number of input tokens
    
    except Exception as e:
        logging.error(f'Exception occurred during calling {model_name}')
        return '', 0


def generate_llama_3(messages: list, temperature: float=0.5) -> str:
    ''' generate response using llama-3 '''

    global pipeline

    prompt = pipeline.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    result = pipeline(
        prompt,
        do_sample=True,
        eos_token_id=terminators,
        remove_invalid_values=True,
        top_k=10,
        num_return_sequences=1,
        # max_length=1024,
        max_new_tokens=400,
        temperature=temperature,
    )

    return re.findall(r'<\|start_header_id\|>assistant<\|end_header_id\|>(.*)', result[0]['generated_text'].replace('\n', ' '))[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True, help='The dataset file.')
    parser.add_argument('--prompt_config', required=True, help='The prompt configuration file.')
    parser.add_argument('--output_path', required=True, help='The file of generated responses.')
    parser.add_argument('--model_name', required=True, choices=model_list, help='The name of model.')
    parser.add_argument('--no_call', action='store_true', help='Just calculate predicted number of total tokens. No calling api.')
    parser.add_argument('--psg_num', type=int, default=5, help='The number of passages used in generation.')
    parser.add_argument('--use_sum', action='store_true', help='Use passage\'s summary.')
    parser.add_argument('--use_snippet', action='store_true', help='Use passages\' snippet.')
    args = parser.parse_args()

    with open(args.data_path, 'r') as file:
        data = json.load(file)

    with open(args.prompt_config, 'r') as file:
        prompt_dict = json.load(file)

    total_tokens = 0
    for item in tqdm(data):

        if args.no_call:
            total_tokens += num_tokens_from_message(get_messages(get_prompt(item, prompt_dict)), args.model_name)
        
        else:
            if 'gpt' in args.model_name:
                # using openai
                global client

                if client is None:
                    client = openai.OpenAI(
                        base_url=OPEN_API_URL,
                        api_key=OPEN_API_KEY
                    )

                instruction = 'You are ChatGPT, a large language model trained by OpenAI.'

                if args.use_sum:
                    prompt = get_prompt(item, prompt_dict, args.psg_num, 'summary')
                elif args.use_snippet:
                    prompt = get_prompt(item, prompt_dict, args.psg_num, 'extraction')
                else:
                    prompt = get_prompt(item, prompt_dict, args.psg_num)
                
                messages = get_messages(prompt, instruction)

                output, num_input_tokens = generate_openai(messages, args.model_name)
            
            else:
                # using llama
                global pipeline

                if pipeline is None:
                    pipeline = transformers.pipeline(
                        'text-generation',
                        model=args.model_name,
                        torch_dtype=torch.float16,
                        device_map='auto'
                    )

                instruction = 'You are a helpful AI assistant for QA task.'

                if args.use_sum:
                    prompt = get_prompt(item, prompt_dict, args.psg_num, 'summary')
                elif args.use_snippet:
                    prompt = get_prompt(item, prompt_dict, args.psg_num, 'extraction')
                else:
                    prompt = get_prompt(item, prompt_dict, args.psg_num)
                
                messages = get_messages(prompt, instruction)

                num_input_tokens = num_tokens_from_message(messages)
                output = generate_llama_3(messages)

            item['output'] = output
            total_tokens += num_input_tokens
    
    if args.no_call:
        logging.info(f'The predicted number of tokens is {total_tokens}')
    else:
        logging.info(f'The actual number of tokens is {total_tokens}')

        with open(args.output_path, 'w') as file:
            json.dump(data, file, indent=4)


if __name__ == '__main__':
    main()

