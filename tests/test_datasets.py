import json
import pytest
from fastapi.testclient import TestClient
from app.core.datasets import DATASETS
from app.services.detector import build_detection_prompt

@pytest.mark.parametrize('dataset,anomaly', [('n24_news','Food'),('bbc','entertainment')])
def test_dataset_detection_uses_its_own_memory_and_categories(dataset, anomaly, monkeypatch, tmp_path):
    from app.api import detect as api, dashboard
    from app.services.memory import Neighbour
    from app.services.llm_client import LLMResponse
    from app.services.stats import StatsStore
    from app.main import app
    requested=[]
    def retrieve(key,text,k):
        requested.append(key)
        return [Neighbour(0,'trn_00000',DATASETS[key].normal_categories[0],f'{key} reference text',.1)]
    class Client:
        def generate(self,prompt,**kwargs):
            return LLMResponse(json.dumps({'reason':'fixture','anomaly_score':.2,'citations':[1]}),'fixture')
    monkeypatch.setattr(api,'retrieve_neighbours',retrieve)
    monkeypatch.setattr(api,'get_llm_client',lambda:Client())
    monkeypatch.setattr(dashboard,'store',StatsStore(tmp_path))
    with TestClient(app) as client:
        response=client.post('/detect/compare',json={'dataset':dataset,'text':'New article','setting':'normal_anomaly'})
    assert response.status_code==200
    result=response.json()
    assert requested==[dataset]
    assert result['dataset']==dataset
    assert f'### Anomaly Category(ies):\n- {anomaly}' in result['rag']['prompt']
    assert f'{dataset} reference text' in result['rag']['prompt']
    assert f'{dataset} reference text' not in result['baseline']['prompt']
    other='Food' if dataset=='bbc' else 'entertainment'
    assert f'### Anomaly Category(ies):\n- {other}' not in result['rag']['prompt']
