"implements search-and-learn (sal) recipy for creating a reasoning architecture, see https://github.com/huggingface/search-and-learn"
"please clone https://github.com/huggingface/search-and-learn and install dependencies

"the reasoning architecture uses llm, prm (reward model) and search strategy (e.g. best-of-n) to define the best answer after the reasoning process"

"to use this reasoning backend you need to utilize the following model_reagistry"
"""
{
    "model_name": "llama-8b-sal",
    "backend": "vllm-reasoning-sal",
    "huggingface_id": "meta-llama/Llama-3.1-8B-Instruct", #or use another huggingface LLM
    "release_date": "",
    "open_weight": true,
    "parameters": "8B",
    "languages": ["en"],
    "license": {
      "name": "",
      "url": ""
    },
    "model_config": {
      "requires_api_key": true,
      "premade_chat_template": true,
      "eos_to_cull": "<\\|eot_id\\|>",
      "method": best_of_n #or beam_search, or dvts, see https://github.com/huggingface/search-and-learn"
      "vllm_args": {
       "gpu_memory_utilization": 0.5 #recommended parameter since enough gpu capacity has to be left for prm
       }
    },
    "prm_config": {
      "prm_name": "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data" #reward model recommended by sal
    }
  }
"""
"Note: temperature and max_tokens are playpen cli-arguments as usual"

import logging
import re
import torch
import vllm

from typing import List, Dict, Tuple, Any, Union
from transformers import AutoTokenizer, AutoConfig

from jinja2 import TemplateError
import clemcore.backends as backends
from clemcore.backends.utils import ensure_alternating_roles

logger = logging.getLogger(__name__)

FALLBACK_CONTEXT_SIZE = 256

#these modules are from sal; clone the repository https://github.com/huggingface/search-and-learn and install the dependencies to make the following modules available
from sal.models.reward_models import RLHFFlow
from sal.config import Config
from sal.search import beam_search, best_of_n, dvts

def load_config_and_tokenizer(model_spec: backends.ModelSpec) -> Union[AutoTokenizer, AutoConfig, int]:
    """
    Load a HuggingFace model's standard config and tokenizer, and get context token limit from config.
    :param model_spec: The ModelSpec for the model.
    :return: Tokenizer, model config and context token limit (int).
    """
    logger.info(f'Loading model config and tokenizer from HuggingFace: {model_spec.model_name}')

    use_api_key = False
    api_key = None
    if 'requires_api_key' in model_spec['model_config']:
        if model_spec['model_config']['requires_api_key']:
            # load HF API key:
            creds = backends.load_credentials("huggingface")
            api_key = creds["huggingface"]["api_key"]
            use_api_key = True
        else:
            requires_api_key_info = (f"{model_spec['model_name']} registry setting has requires_api_key, "
                                     f"but it is not 'true'. Please check the model entry.")
            print(requires_api_key_info)
            logger.info(requires_api_key_info)

    hf_model_str = model_spec['huggingface_id']

    # use 'slow' tokenizer for models that require it:
    if 'slow_tokenizer' in model_spec['model_config']:
        if model_spec['model_config']['slow_tokenizer']:
            tokenizer = AutoTokenizer.from_pretrained(hf_model_str, device_map="auto", torch_dtype="auto",
                                                      verbose=False, use_fast=False)
        else:
            tokenizer = None
            slow_tokenizer_info = (f"{model_spec['model_name']} registry setting has slow_tokenizer, "
                                   f"but it is not 'true'. Please check the model entry.")
            print(slow_tokenizer_info)
            logger.info(slow_tokenizer_info)
    elif use_api_key:
        tokenizer = AutoTokenizer.from_pretrained(hf_model_str, token=api_key, device_map="auto",
                                                  torch_dtype="auto", verbose=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(hf_model_str, device_map="auto", torch_dtype="auto",
                                                  verbose=False)

    # apply proper chat template:
    if not model_spec['model_config']['premade_chat_template']:
        if 'custom_chat_template' in model_spec['model_config']:
            tokenizer.chat_template = model_spec['model_config']['custom_chat_template']
        else:
            logger.info(
                f"No custom chat template for {model_spec.model_name} found in model settings from model registry "
                f"while model has no pre-made template! Generic template will be used, likely leading to "
                f"bad results.")

    if use_api_key:
        model_config = AutoConfig.from_pretrained(hf_model_str, token=api_key)
    else:
        model_config = AutoConfig.from_pretrained(hf_model_str)

    # get context token limit for model:
    if hasattr(model_config, 'max_position_embeddings'):  # this is the standard attribute used by most
        context_size = model_config.max_position_embeddings
    elif hasattr(model_config, 'n_positions'):  # some models may have their context size under this attribute
        context_size = model_config.n_positions
    else:  # few models, especially older ones, might not have their context size in the config
        context_size = FALLBACK_CONTEXT_SIZE

    # stopping transformers pad_token_id warnings
    # check if tokenizer has no set pad_token_id:
    if not tokenizer.pad_token_id:  # if not set, pad_token_id is None
        # preemptively set pad_token_id to eos_token_id as automatically done to prevent warning at each generation:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer, model_config, context_size


def load_model(model_spec: backends.ModelSpec) -> Any:
    """
    Load model weights from HuggingFace, onto the number of GPUs specified in the ModelSpec/model registry entry.
    :param model_spec: The ModelSpec for the model.
    :return: The vLLM LLM class instance of the loaded model.
    """
    assert "model_config" in model_spec, "vllm model requires model_config entry in model spec"
    model_config = model_spec.model_config

    default_args = dict(tensor_parallel_size=model_config['number_gpus'] if 'number_gpus' in model_config else 1)
    max_model_len = int(model_spec.context_size) if 'context_size' in model_spec and model_spec.context_size else None
    if max_model_len is not None:
        default_args["max_model_len"] = max_model_len

    vllm_args = model_config['vllm_args'] if 'vllm_args' in model_config else {}
    model_args = {**default_args, **vllm_args}
    logger.info(f"Number of GPUs used for model: {model_args['tensor_parallel_size']}")
    if "max_model_len" in model_args:
        logger.info(f"Context size forcefully limited to {model_args['max_model_len']} tokens.")

    logger.info(f'Start loading model weights from HuggingFace: {model_spec.model_name}')
    model = vllm.LLM(model_spec.huggingface_id, **model_args)
    logger.info(f"Finished loading model weights from HuggingFace: {model_spec.model_name}")
    return model

def load_prm(model_spec: backends.ModelSpec) -> Any:
    assert "prm_config" in model_spec, "reasoning vllm model requires prm"
    prm_config = model_spec.prm_config
    prm_name = prm_config['prm_name']
    prm = RLHFFlow(prm_name)
    return prm

class ReasoningLocal(backends.Backend):
    """
    Model/backend handler class for locally-run models using vLLM and sal: https://github.com/huggingface/search-and-learn
    """

    def __init__(self):
        super().__init__()

    def get_model_for(self, model_spec: backends.ModelSpec) -> backends.Model:
        """
        Get a ReasoingLocalModel instance with the passed model and settings.
        :param model_spec: The ModelSpec for the model.
        :return: The Model class instance of the model.
        """
        torch.set_num_threads(1)
        return ReasoningLocalModel(model_spec)


class ReasoningLocalModel(backends.Model):
    """
    Class for loaded reasoning vLLM models ready for generation.
    Implementation based on the existing VLLM backend and sal: https://github.com/huggingface/search-and-learn
    """
    def __init__(self, model_spec: backends.ModelSpec):
        super().__init__(model_spec)
        # fail-fast
        self.tokenizer, self.config, self.context_size = load_config_and_tokenizer(model_spec)
        self.model = load_model(model_spec)
        self.prm = load_prm(model_spec)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def generate_response(self, messages: List[Dict],
                          return_full_text: bool = False,
                          log_messages: bool = False) -> Tuple[Any, Any, str]:
        """
        :param messages: for example
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Who won the world series in 2020?"},
                    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                    {"role": "user", "content": "Where was it played?"}
                ]
        :param return_full_text: If True, whole input context is returned.
        :param log_messages: If True, raw and cleaned messages passed will be logged.
        :return: the continuation
        """
        # log current given messages list:
        if log_messages:
            logger.info(f"Raw messages passed: {messages}")

        current_messages = ensure_alternating_roles(messages)

        # log current flattened messages list:
        if log_messages:
            logger.info(f"Flattened messages: {current_messages}")

        # apply chat template & tokenize:
        prompt_tokens = self.tokenizer.apply_chat_template(current_messages, add_generation_prompt=True,
                                                           return_tensors="pt")

        # decode again to get properly formatted prompt text:
        prompt_text = self.tokenizer.batch_decode(prompt_tokens)[0]

        prompt = {"inputs": prompt_text, "max_new_tokens": self.max_tokens,
                  "temperature": self.temperature, "return_full_text": return_full_text}

        # check context limit:
        context_check = _check_context_limit(self.context_size, prompt_tokens[0],
                                             max_new_tokens=self.max_tokens)
        if not context_check[0]:  # if context is exceeded, context_check[0] is False
            logger.info(f"Context token limit for {self.model_spec.model_name} exceeded: "
                        f"{context_check[1]}/{context_check[3]}")
            # fail gracefully:
            raise backends.ContextExceededError(f"Context token limit for {self.model_spec.model_name} exceeded",
                                                tokens_used=context_check[1], tokens_left=context_check[2],
                                                context_size=context_check[3])
        
        #sal formating of the prompt
        input_batch = {"problem": [prompt_text]}

        #sal config
        config = Config()
        
        #search related options
        config.n = 4  # Number of answers to generate during the search
        config.temperature = self.temperature # identified by playpen cli-argument -T, recommended > 0.0 (not greedy decoding)
        config.max_tokens=self.max_tokens # identified by playpen cli-argument -L
        config.system_prompt = "You are going to play language games. Please follow the instructions PRECISELY. It is of vital ipportance to yield all the answers in a format EXACTLY ACCORDING to the instructions."
        config.custom_chat_template = None
        
        #beam / dvts search options:
        config.num_iterations = 10 #number of levels in a tree search (tree depth)        

        #for other sal config parameters see search-and-learn -> src -> sal -> config.py

        #you can specify what search to use in model_config
        if self.model_spec.model_config["method"] == "best_of_n":
            print("*****best_of_n*****")
            model_output = best_of_n(x=input_batch, config=config, llm=self.model, prm=self.prm)
        elif self.model_spec.model_config["method"] == "beam_search":
            print("*****beam_search*****")
            model_output = beam_search(examples=input_batch, config=config, llm=self.model, prm=self.prm)
        elif self.model_spec.model_config["method"] == "dvts":
            model_output = dvts(examples=input_batch, config=config, llm=self.model, prm=self.prm)

        response_text = model_output["pred"][0]
        response = {'response': prompt_text+response_text}

        return prompt, response, response_text


def _check_context_limit(context_size, prompt_tokens, max_new_tokens: int = 100) -> Tuple[bool, int, int, int]:
    """
    Internal context limit check to run in generate_response.
    :param prompt_tokens: List of prompt token IDs.
    :param max_new_tokens: How many tokens to generate ('at most', but no stop sequence is defined).
    :return: Tuple with
            Bool: True if context limit is not exceeded, False if too many tokens
            Number of tokens for the given messages and maximum new tokens
            Number of tokens of 'context space left'
            Total context token limit
    """
    prompt_size = len(prompt_tokens)
    tokens_used = prompt_size + max_new_tokens  # context includes tokens to be generated
    tokens_left = context_size - tokens_used
    fits = tokens_used <= context_size
    return fits, tokens_used, tokens_left, context_size


def check_messages(messages: List[Dict], model_spec: backends.ModelSpec) -> bool:
    """
    Message checking for clemgame development. This checks if the model's chat template accepts the given messages
    as passed, before the standard flattening done for generation. This allows clemgame developers to construct
    message lists that are sound as-is and are not affected by the indiscriminate flattening of the generation
    method. Deliberately verbose.
    :param model_spec: The ModelSpec for the model.
    :param messages: for example
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Who won the world series in 2020?"},
                {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                {"role": "user", "content": "Where was it played?"}
            ]
    :return: True if messages are sound as-is, False if messages are not compatible with the model's template.
    """
    tokenizer, _, _ = load_config_and_tokenizer(model_spec)

    # bool for message acceptance:
    messages_accepted: bool = True

    # check for system message:
    has_system_message: bool = False
    if messages[0]['role'] == "system":
        print("System message detected.")
        has_system_message = True
        if not messages[0]['content']:
            print(f"Initial system message is empty. It will be removed when generating responses.")
        else:
            print(f"Initial system message has content! It will not be removed when generating responses. This "
                  f"will lead to issues with models that do not allow system messages.")
        """
        print("Checking model system message compatibility...")
        # unfortunately Mistral models, which do not accept system message, currently do not raise a distinct 
        # exception for this...
        try:
            self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        except TemplateError:
            print("The model's chat template does not allow for system message!")
            messages_accepted = False
        """

    # check for message order:
    starts_with_assistant: bool = False
    double_user: bool = False
    double_assistant: bool = False
    ends_with_assistant: bool = False

    for msg_idx, message in enumerate(messages):
        if not has_system_message:
            if msg_idx == 0 and message['role'] == "assistant":
                starts_with_assistant = True
        else:
            if msg_idx == 1 and message['role'] == "assistant":
                starts_with_assistant = True
        if msg_idx > 0 and message['role'] == "user" and messages[msg_idx - 1]['role'] == "user":
            double_user = True
        elif msg_idx > 0 and message['role'] == "assistant" and messages[msg_idx - 1]['role'] == "assistant":
            double_assistant = True
    if messages[-1]['role'] == "assistant":
        ends_with_assistant = True

    if starts_with_assistant or double_user or double_assistant or ends_with_assistant:
        print("Message order issue(s) found:")
        if starts_with_assistant:
            print("First message has role:'assistant'.")
        if double_user:
            print("Messages contain consecutive user messages.")
        if double_assistant:
            print("Messages contain consecutive assistant messages.")
        if ends_with_assistant:
            print("Last message has role:'assistant'.")

    # proper check of chat template application:
    try:
        tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    except TemplateError:
        print(f"The {model_spec.model_name} chat template does not accept these messages! "
              f"Cleaning applied before generation might still allow these messages, but is indiscriminate and "
              f"might lead to unintended generation inputs.")
        messages_accepted = False
    else:
        print(
            f"The {model_spec.model_name} chat template accepts these messages. Cleaning before generation is still "
            f"applied to these messages, which is indiscriminate and might lead to unintended generation inputs.")

    return messages_accepted


def check_context_limit(messages: List[Dict], model_spec: backends.ModelSpec,
                        max_new_tokens: int = 100, clean_messages: bool = False,
                        verbose: bool = True) -> Tuple[bool, int, int, int]:
    """
    Externally-callable context limit check for clemgame development.
    :param messages: for example
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Who won the world series in 2020?"},
                {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                {"role": "user", "content": "Where was it played?"}
            ]
    :param model_spec: The ModelSpec for the model.
    :param max_new_tokens: How many tokens to generate ('at most', but no stop sequence is defined).
    :param clean_messages: If True, the standard cleaning method for message lists will be applied.
    :param verbose: If True, prettyprint token counts.
    :return: Tuple with
            Bool: True if context limit is not exceeded, False if too many tokens
            Number of tokens for the given messages and maximum new tokens
            Number of tokens of 'context space left'
            Total context token limit
    """
    tokenizer, _, context_size = load_config_and_tokenizer(model_spec)

    # optional messages processing:
    if clean_messages:
        current_messages = ensure_alternating_roles(messages)
    else:
        current_messages = messages
    # the actual tokens, including chat format:
    prompt_tokens = tokenizer.apply_chat_template(current_messages, add_generation_prompt=True)
    context_check_tuple = _check_context_limit(context_size, prompt_tokens, max_new_tokens=max_new_tokens)
    tokens_used = context_check_tuple[1]
    tokens_left = context_check_tuple[2]
    if verbose:
        print(f"{tokens_used} input tokens, {tokens_left} tokens of {context_size} left.")
    fits = context_check_tuple[0]
    return fits, tokens_used, tokens_left, context_size
