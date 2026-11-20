import ast,pathlib
from types import SimpleNamespace as NS
R=pathlib.Path(__file__).resolve().parents[5]

def helper():
    tree=ast.parse((R/'core/app/task_pipeline/easy_ui_based_generate_task_pipeline.py').read_text());method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_usage_metadata');module=ast.Module(body=[method],type_ignores=[]);scope={'jsonable_encoder':lambda x:dict(x)};exec(compile(module,'<actual_pipeline_method>','exec'),scope);return scope['_usage_metadata']

def test_tokenify_model_and_quote_reach_usage_metadata():
    obj=NS(_model_config=NS(provider='yuanma/yuanma_extended/yuanma_extended',model='gpt-6-luna'),_task_state=NS(llm_result=NS(model='gpt-6-luna-tokenify',system_fingerprint='yuanma-credit-route/v1:tokenify',usage={'total_price':'0.000007','currency':'USD'})))
    x=helper()(obj);assert x['actual_provider']=='Tokenify' and x['model_name']=='gpt-6-luna-tokenify' and x['total_price']=='0.000007' and x['cost_basis']=='supplier_quote_70pct_estimate'

def test_official_terminal_stays_official():
    obj=NS(_model_config=NS(provider='yuanma/yuanma_extended/yuanma_extended',model='gpt-6-luna'),_task_state=NS(llm_result=NS(model='gpt-6-luna',system_fingerprint='yuanma-credit-route/v1:paid',usage={'total_price':'0.00001'})))
    x=helper()(obj);assert x['actual_provider']=='OpenAI' and x['cost_basis']=='catalog_estimate'

def test_unrelated_provider_has_no_changed_fields():
    usage={'total_price':'0.00001'}
    obj=NS(_model_config=NS(provider='unrelated',model='gpt-6-luna'),_task_state=NS(llm_result=NS(model='gpt-6-luna-tokenify',system_fingerprint='yuanma-credit-route/v1:tokenify',usage=usage)))
    assert helper()(obj)==usage


def test_stopped_stream_and_direct_answer_do_not_invent_supplier():
    for model in ['gpt-6-luna', 'gpt-6-luna-tokenify']:
        usage={'total_price':'0.00001'}
        obj=NS(_model_config=NS(provider='yuanma/yuanma_extended/yuanma_extended',model='gpt-6-luna'),_task_state=NS(llm_result=NS(model=model,system_fingerprint=None,usage=usage)))
        assert helper()(obj)==usage

def test_openrouter_reply_is_not_official():
    usage={'total_price':'0.00001'}
    obj=NS(_model_config=NS(provider='yuanma/yuanma_extended/yuanma_extended',model='gpt-6-luna'),_task_state=NS(llm_result=NS(model='openai/gpt-6-luna',system_fingerprint=None,usage=usage)))
    assert helper()(obj)==usage

def test_actual_runner_preserves_terminal_fingerprint():
    tree=ast.parse((R/'core/app/apps/base_app_runner.py').read_text());method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_handle_invoke_result_stream');method.returns=None
    for arg in method.args.args:arg.annotation=None
    scope={'LLMResult':NS,'AssistantPromptMessage':NS,'QueueMessageEndEvent':lambda **kw:NS(kind='end',**kw),'QueueLLMChunkEvent':lambda **kw:NS(kind='chunk',**kw),'PublishFrom':NS(APPLICATION_MANAGER='test')};exec(compile(ast.Module(body=[method],type_ignores=[]),'<actual_runner>','exec'),scope)
    events=[];queue=NS(publish=lambda event,*_:events.append(event))
    chunks=[NS(model='gpt-6-luna-tokenify',system_fingerprint='yuanma-credit-route/v1:tokenify',prompt_messages=[],delta=NS(message=NS(content='READY'),usage={'total_price':'0.00000259'}))]
    scope['_handle_invoke_result_stream'](NS(),iter(chunks),queue,False)
    assert events[-1].kind=='end' and events[-1].llm_result.system_fingerprint=='yuanma-credit-route/v1:tokenify'


def test_unknown_route_or_other_model_does_not_invent_supplier():
    for model, marker in [('gpt-6-luna', 'yuanma-credit-route/v1:credit_9'),
                          ('gpt-6-luna', 'unrelated:paid'), ('other-model', 'yuanma-credit-route/v1:paid')]:
        usage={'total_price':'0.00001'}
        obj=NS(_model_config=NS(provider='yuanma/yuanma_extended/yuanma_extended',model=model),
               _task_state=NS(llm_result=NS(system_fingerprint=marker,usage=usage)))
        assert helper()(obj)==usage


def test_all_official_slots_preserve_usage_fields():
    for slot in ['paid', *(f'credit_{i}' for i in range(1,9))]:
        usage={'total_price':'0.00001','prompt_tokens':12,'completion_tokens':5,'cached_tokens':3}
        obj=NS(_model_config=NS(provider='yuanma/yuanma_extended/yuanma_extended',model='gpt-6-luna'),
               _task_state=NS(llm_result=NS(system_fingerprint='yuanma-credit-route/v1:'+slot,usage=usage)))
        result=helper()(obj)
        assert result['provider_route']==slot and result['actual_provider']=='OpenAI'
        assert all(result[key]==value for key,value in usage.items())
