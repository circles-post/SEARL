import time
import asyncio
import aiohttp
from openai import OpenAI
from easydict import EasyDict
from .base_generator import BaseGenerator, register_generator
from verl.utils.vllm_request import vllm_generate


def register_api_method(name):
    """Decorator for registering methods into the dictionary"""
    def decorator(func):
        # Do not directly access the class in the decorator, handle it in the class instead
        func._api_method_name = name
        return func
    return decorator


def register_async_api_method(name):
    """Decorator for registering async methods into the dictionary"""
    def decorator(func):
        # Do not directly access the class in the decorator, handle it in the class instead
        func._async_api_method_name = name
        return func
    return decorator


@register_generator('api')
class APIGenerator(BaseGenerator):
    # Create a function mapping dictionary
    api_methods = {}
    async_api_methods = {}

    def __init__(self, config: EasyDict):
        super().__init__(config)
        self.api_method = config.api_method

        # Register all methods with _api_method_name attribute during initialization
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if callable(attr) and hasattr(attr, '_api_method_name'):
                self.api_methods[attr._api_method_name] = attr
            if callable(attr) and hasattr(attr, '_async_api_method_name'):
                self.async_api_methods[attr._async_api_method_name] = attr

        if self.api_method not in self.api_methods:
            raise ValueError(f"Unsupported API method: {self.api_method}. The API method should be in [{', '.join(self.api_methods.keys())}]")

        if hasattr(config, 'port'):
            self.port = config.port
        else:
            self.port = 9000
        
        if hasattr(config, 'model_name'):
            self.model_name = config.model_name
        else:
            self.model_name = None
        
        self.selected_method = self.api_methods[self.api_method]
        # If there is an async version, select the corresponding async method
        if self.api_method in self.async_api_methods:
            self.selected_async_method = self.async_api_methods[self.api_method]
        else:
            # If no specific async method is implemented, provide an async wrapper based on the sync method
            self.selected_async_method = None
    
    @register_api_method('qwq')
    def get_response_qwq(self, message_list, temperature=0.7):
        client = OpenAI(api_key='dummy_key', base_url='http://0.0.0.0:8000/v1')
        model_name = 'deepseek-r1'
        response = client.chat.completions.create(
            model=model_name,
            messages=message_list,
            temperature=temperature, 
            stream=True
        )

        response_str = ""
        for chunk in response:
            for choice in chunk.choices:
                response_str += choice.delta.content
        
        return response_str

    @register_async_api_method('qwq')
    async def get_response_qwq_async(self, message_list, temperature=0.7):
        # Use thread pool to run the synchronous version
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self.get_response_qwq, message_list, temperature
        )

    @register_api_method('deepseek-r1')
    def get_response_r1(self, message_list, temperature=0.7):
        client = OpenAI(api_key='dummy_key', base_url='http://0.0.0.0:8000/v1')
        model_name = 'DeepSeek-R1'
        response = client.chat.completions.create(
            model=model_name,
            messages=message_list,
            temperature=temperature, 
            stream=True
        )

        response_str = ""
        for chunk in response:
            for choice in chunk.choices:
                response_str += choice.delta.content
        
        return response_str

    @register_async_api_method('deepseek-r1')
    async def get_response_r1_async(self, message_list, temperature=0.7):
        # Use thread pool to run the synchronous version
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self.get_response_r1, message_list, temperature
        )
    
    @register_api_method('local')
    def get_response_local(self, message_list, temperature=0.7):
        stream = False
        response = ""
        
        # Maximum number of attempts to call the API
        max_attempts = 3
        
        for attempt in range(max_attempts):
            try:
                print(f"Trying to call API service (attempt {attempt+1}/{max_attempts}), port: {self.port}")
                generation_stream = vllm_generate(
                    f'http://0.0.0.0:{self.port}', message_list, model=self.model_name, 
                    temperature=temperature, stream=stream
                )
                
                if stream:
                    if generation_stream is not None:
                        for chunk in generation_stream:
                            response += chunk
                else:
                    response = generation_stream
                
                # Check if the response is empty
                if response is None or response == "":
                    print(f"Warning: API returned an empty response (attempt {attempt+1}/{max_attempts})")
                    if attempt < max_attempts - 1:
                        print("Waiting 1 second before retrying...")
                        time.sleep(1)
                        continue
                    else:
                        print("All attempts failed, returning default response")
                        return "API service is temporarily unavailable, please try again later."
                else:
                    # Successfully obtained response, break the loop
                    break
                
            except Exception as e:
                print(f'API call error: {e} (attempt {attempt+1}/{max_attempts})')
                if attempt < max_attempts - 1:
                    print("Waiting 1 second before retrying...")
                    time.sleep(1)
                else:
                    print("All attempts failed, returning default response")
                    return "API service is temporarily unavailable, please try again later."
        
        return response

    @register_async_api_method('local')
    async def get_response_local_async(self, message_list, temperature=0.7):
        """Asynchronous version of local VLLM API call"""
        stream = False
        response = ""
        max_attempts = 3
        
        for attempt in range(max_attempts):
            try:
                print(f"Trying to call API service (attempt {attempt+1}/{max_attempts}), port: {self.port}")
                generation_stream = await vllm_generate(
                    f'http://0.0.0.0:{self.port}', 
                    message_list, 
                    model=self.model_name, 
                    temperature=temperature, 
                    stream=stream,
                    is_async=True
                )
                
                if stream:
                    if generation_stream is not None:
                        for chunk in generation_stream:
                            response += chunk
                else:
                    response = generation_stream
                
                # Check if the response is empty
                if response is None or response == "":
                    print(f"Warning: API returned an empty response (attempt {attempt+1}/{max_attempts})")
                    if attempt < max_attempts - 1:
                        print("Waiting 1 second before retrying...")
                        await asyncio.sleep(1)
                        continue
                    else:
                        print("All attempts failed, returning default response")
                        return "API service is temporarily unavailable, please try again later."
                else:
                    # Successfully obtained response, break the loop
                    break
                
            except Exception as e:
                print(f'API call error: {e} (attempt {attempt+1}/{max_attempts})')
                if attempt < max_attempts - 1:
                    print("Waiting 1 second before retrying...")
                    await asyncio.sleep(1)
                else:
                    print("All attempts failed, returning default response")
                    return "API service is temporarily unavailable, please try again later."
        
        return response

    def generate(self, input_data, temperature=0.7):
        """Synchronous generation method"""
        # Check the type of input. If it is a string, convert to message_list format
        if isinstance(input_data, str):
            message_list = [{'role': 'system', 'content': input_data}]
        elif isinstance(input_data, list):
            message_list = input_data
        else:
            raise ValueError("Input must be either a list or a string.")

        # Call the selected method
        return self.selected_method(message_list, temperature)

    async def generate_async(self, input_data, temperature=0.7):
        """Asynchronous generation method"""
        # Check the type of input. If it is a string, convert to message_list format
        if isinstance(input_data, str):
            message_list = [{'role': 'system', 'content': input_data}]
        elif isinstance(input_data, list):
            message_list = input_data
        else:
            raise ValueError("Input must be either a list or a string.")

        # If there is an async method, call the corresponding async method
        if self.selected_async_method:
            return await self.selected_async_method(message_list, temperature)
        else:
            # If there is no corresponding async method, use thread pool to run the sync method
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, self.selected_method, message_list, temperature
            )
