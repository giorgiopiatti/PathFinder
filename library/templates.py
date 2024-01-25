LLAMA_CHAT_TEMPLATE = (
    "{% for message in messages %}{% if message['role'] == 'system' %}{{ '[INST]"
    " <<SYS>>\n' + message['content'] + ' \n<</SYS>>\n\n' }}{% elif message['role'] =="
    " 'user' %}{% if messages[loop.index0-1]['role'] == 'system' %}{{"
    " message['content'] + '  [/INST]' }}{% else %}{{ '[INST] ' + message['content'] +"
    " '  [/INST]' }}{% endif %}{% elif message['role'] == 'assistant' %}{{ ' '+"
    " message['content']}}{% if loop.index0 != (loop.length -1) %} {{bos_token +"
    " eos_token}}{% endif %}{% endif %}{% endfor %}"
)

MIXTRAL_INSTRUCT_TEMPLATE = (
    "{{bos_token}}{% for message in messages %}{% if message['role'] == 'system' %}{{"
    " '[INST] ' + message['content'] + ' \n' }}{% elif message['role'] == 'user' %}{%"
    " if messages[loop.index0-1]['role'] == 'system' %}{{ message['content'] + '"
    " [/INST]' }}{% else %}{{ ' [INST]  ' + message['content'] + '  [/INST]' }}{% endif"
    " %}{% elif message['role'] == 'assistant' %}{{ ' '+ message['content']}}{% if"
    " loop.index0 != (loop.length -1) %}{{eos_token}}{% endif %}{% endif %}{% endfor %}"
)
