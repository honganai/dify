"""Exercise the real runner invocation site without booting Flask or a database."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[4]
APP = '11111111-1111-4111-8111-111111111111'
MESSAGE = '22222222-2222-4222-8222-222222222222'
PROVIDER = 'yuanma/postplus_vertex/postplus_vertex'


def load_helper():
    path = ROOT / 'core/app/apps/postplus_route_context.py'
    spec = importlib.util.spec_from_file_location('postplus_context', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.postplus_model_parameters


@pytest.mark.parametrize('kind', ['completion', 'chat'])
@pytest.mark.parametrize('provider', [PROVIDER, 'langgenius/vertex_ai/vertex_ai'])
def test_runner_binds_only_custom_provider_without_changing_shared_parameters(kind, provider):
    path = ROOT / 'core/app/apps' / kind / 'app_runner.py'
    tree = ast.parse(path.read_text())
    invocation = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'invoke_result' for t in node.targets))
    parameters = {'temperature': 0.1, '_postplus_message_id': 'untrusted'}
    captured = {}
    def invoke(**kwargs):
        captured.update(kwargs)
    scope = dict(model_instance=NS(invoke_llm=invoke), prompt_messages=['unchanged'], stop=[],
                 application_generate_entity=NS(model_conf=NS(provider=provider, parameters=parameters), stream=True, user_id='unchanged'),
                 app_record=NS(id=APP), message=NS(id=MESSAGE))
    # The first RED is the unmodified runner passing the stale ID unchanged.
    if any(isinstance(n, ast.Name) and n.id == 'postplus_model_parameters' for n in ast.walk(invocation)):
        scope['postplus_model_parameters'] = load_helper()
    exec(compile(ast.fix_missing_locations(ast.Module(body=[invocation], type_ignores=[])), str(path), 'exec'), scope)
    assert captured['prompt_messages'] == ['unchanged']
    assert parameters == {'temperature': 0.1, '_postplus_message_id': 'untrusted'}
    if provider == PROVIDER:
        assert captured['model_parameters'] == {'temperature': 0.1, '_postplus_app_id': APP, '_postplus_message_id': MESSAGE}
    else:
        assert captured['model_parameters'] == parameters
