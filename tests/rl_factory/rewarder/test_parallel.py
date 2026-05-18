from easydict import EasyDict
from multiprocessing import Pool
from generator import get_generator
from verl.utils.vllm_request import vllm_generate


def test_parallel_rewarder():
    questions = [
        "Python中如何实现多线程编程？",
        "解释一下Python的GIL（全局解释器锁）及其影响",
        "Python中的装饰器是什么？请举例说明",
        "如何用Python处理JSON数据？",
        "Python中列表(list)和元组(tuple)有什么区别？",
        "解释Python的生成器(generator)和它们的优势",
        "Python中如何处理异常？try-except块如何使用？",
        "Python的虚拟环境(virtualenv)有什么作用？如何创建和使用？"
    ]

    generator = get_generator('api')(
        config=EasyDict({
            'api_method': 'local',
            'port': 9000
        }))
    
    print('Start')
    with Pool(processes=8) as pool:
        results = []
        for question in questions:
            result = pool.apply_async(
                vllm_generate, args=('http://0.0.0.0:8080', question, '/your/path/to/Qwen/QwQ-32B')
            )
            results.append(result)

        datasets_processed = [result.get() for result in results]


if __name__ == '__main__':
    test_parallel_rewarder()
