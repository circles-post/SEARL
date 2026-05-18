

def str_to_camel(s):
    return ''.join(word.capitalize() for word in s.split('_'))


class EnvCodesGenerator:
    def __init__(self, env_name):
        self.env_name = env_name

        self.init_code_str = self.get_init_code()

    def generate_env_codes(self):
        pass

    def add_attribute(self, attribute_name, attribute_value):
        temp_code_str = f"""
        {attribute_name} = {attribute_value}
"""
        self.init_code_str += temp_code_str

    def add_method(self, method_name, method_body):
        pass
    
    def get_init_code(self):
        code_str = """
import re
import json
import string
import random
import torch
from .base import Env


class {str_to_camel(self.env_name)}Env(Env):
    def __init__(self, config, centralized_actor=None):
        super().__init__(config, centralized_actor)

"""
        return code_str
    
    def _get_count_rule_code(self):
        code_str = """

"""
        return code_str
    
    def _get_length_rule_code(self):
        code_str = """

"""
        return code_str

    def _get_format_rule_code(self):
        code_str = """

"""
        return code_str
    
    
    
