import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.services.detector import build_detection_prompt, Exemplar, parse_detection_response, parse_citations
from app.services.evaluation import metrics, selection_metrics
from app.services.memory import TrainMemory
from app.services.model_selection import build_selection_prompt, grounding_score, parse_selection
from app.services.profiling import profile_training
from app.services.research_data import Split, eligible_test_rows, normalize, load_split
from app.services.baselines import score_baseline


@pytest.fixture
def training():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(30,8)).astype(np.float32)
    records = [{"text":f"Training article {i} "*(i%3+1),"label":0,"original_label":str(i%3)} for i in range(30)]
    return records,x


def test_training_profile_is_reproducible_and_withholds_test_features(training):
    records,x=training
    a=profile_training(records,x,seed=2,max_samples=20)
    b=profile_training(records,x,seed=2,max_samples=20)
    assert a==b
    assert a['features']['normal_anomaly_separability'] is None
    assert a['features']['test_anomaly_ratio'] is None
    records[0]['label']=1
    with pytest.raises(ValueError): profile_training(records,x)


def test_geometry_on_known_clusters():
    x=np.array([[1,0],[1,0],[0,1],[0,1]],dtype=float)
    rs=[{'text':'some text','label':0,'original_label':str(i//2)} for i in range(4)]
    f=profile_training(rs,x)['features']
    assert f['intra_class_compactness']==pytest.approx(0)
    assert f['inter_class_separability']==pytest.approx(1)
    assert f['category_silhouette']==pytest.approx(1)


@pytest.mark.parametrize('value',['NaN','Infinity','true','-0.2','1.5','"0.7"'])
def test_invalid_scores_are_not_silently_clamped(value):
    assert parse_detection_response('{"reason":"x","anomaly_score":'+value+'}')[1] is None


def test_valid_score_and_citations():
    assert parse_detection_response('```json\n{"reason":"Out of category","anomaly_score":0.8}\n```')==('Out of category',.8)
    assert parse_citations('{"citations":[1,1,9,0,true,2.5,2]}',3)==[1,2]


def test_grounding_rejects_invented_duplicate_and_rounded_values():
    f={'density':.31234,'unavailable':None}
    es=[{'feature':'density','value':.3123,'implication':'local density'},
        {'feature':'density','value':.3123,'implication':'duplicate'},
        {'feature':'unavailable','value':1,'implication':'invented'},
        {'feature':'unknown','value':2,'implication':'invented'}]
    assert grounding_score(es,f)['g_score']==.25
    assert grounding_score([],f)['g_score']==0


def test_four_conditions_and_ranking_contract(training):
    rs,x=training
    profile=profile_training(rs,x)
    prompt,features=build_selection_prompt(profile,'test',['lof','iforest'],'full')
    assert 'intra_class_compactness' in prompt and 'limitations' in prompt
    prompt,basic=build_selection_prompt(profile,'test',['lof','iforest'],'baseline')
    assert 'intra_class_compactness' not in basic and 'limitations' not in prompt
    good=json.dumps({'ranking':['lof','iforest'],'reason':'density','evidence':[]})
    assert parse_selection(good,['lof','iforest'],features)['selected_model']=='lof'
    with pytest.raises(ValueError): parse_selection(good,['lof','ecod'],features)
    with pytest.raises(ValueError): parse_selection(json.dumps({'ranking':['lof','lof'],'reason':'x','evidence':[]}),['lof','iforest'],features)


def test_metrics_preserve_label_alignment_and_report_failures():
    m=metrics([0,1,0,1],[.1,None,.2,.9],bootstrap=20)
    assert m['auroc']==1 and m['auprc']==1 and m['n_errors']==1 and m['coverage']==.75
    assert metrics([0,0],[.1,.2])['auroc'] is None
    m=selection_metrics(['lof','iforest'],{'lof':.7,'iforest':.9})
    assert m['selected_rank']==2 and m['auroc_regret']==pytest.approx(.2)


def test_overlap_and_duplicate_filter():
    def s(texts): return Split([{'text':t} for t in texts],np.ones((len(texts),2)),Path('x'),Path('y'))
    rows,audit=eligible_test_rows(s(['Normal text']),s([' normal   TEXT ','other','other','third']))
    assert rows.tolist()==[1,3]
    assert audit=={'train_test_overlap_removed':1,'test_duplicates_removed':1}


def test_memory_retrieval_and_invalid_vectors(tmp_path):
    p=tmp_path/'v.npy';j=tmp_path/'r.jsonl'
    np.save(p,np.array([[1,0],[0,1],[-1,0]]))
    j.write_text('\n'.join(json.dumps({'text':str(i),'label':0,'original_label':'normal'}) for i in range(3)))
    memory=TrainMemory(p,j)
    assert memory.search(np.array([1,0]),2)[0].doc_id=='trn_00000'
    with pytest.raises(ValueError): memory.search(np.zeros(2),1)
    with pytest.raises(ValueError): memory.search(np.ones(3),1)
    with pytest.raises(ValueError): memory.search(np.ones(2),0)
    with pytest.raises(ValueError): normalize(np.array([[np.nan,0]]))


@pytest.mark.parametrize('model',['iforest','lof','knn'])
def test_baseline_score_orientation(model):
    rng=np.random.default_rng(42)
    x=rng.normal(0,.1,(50,4))+np.array([1,0,0,0])
    z=np.array([[1,0,0,0],[-5,8,7,6]])
    scores,_=score_baseline(model,x,z)
    assert scores[1]>scores[0]


def test_selection_api_with_injected_llm(training,monkeypatch):
    from app.api import selection
    from app.main import app
    from app.services.llm_client import LLMResponse
    rs,x=training
    profile=profile_training(rs,x)
    monkeypatch.setattr(selection,'get_profile',lambda *a,**k:profile)
    class Client:
        def generate(self,*a,**k):
            return LLMResponse(json.dumps({'ranking':['lof','iforest'],'reason':'Training size is 30.',
                'evidence':[{'feature':'n_train','value':30,'implication':'Few samples limit neural training.'}]}),'test-double')
    monkeypatch.setattr(selection,'get_llm_client',lambda:Client())
    with TestClient(app) as client:
        r=client.post('/selection/recommend',json={'candidates':['lof','iforest']})
        assert r.status_code==200 and r.json()['grounding']['g_score']==1
        assert client.post('/selection/recommend',json={'candidates':['fake']}).status_code==422
        assert client.get('/model-selection').status_code==200
        assert client.get('/selection/cards').status_code==200


def test_detection_prompt_settings():
    plain=build_detection_prompt('text',['normal'],'bad','normal_only')
    rag=build_detection_prompt('text',['normal'],'bad','normal_anomaly',[Exemplar('example','normal')])
    assert 'Reference Examples' not in plain and 'Reference Examples' in rag
    assert '### Anomaly Category(ies)' not in plain
    assert '### Anomaly Category(ies)' in rag


def test_detection_api_validation_and_real_pipeline_shape(training,monkeypatch,tmp_path):
    from app.api import detect as api, dashboard
    from app.main import app
    from app.services.llm_client import LLMResponse
    from app.services.memory import Neighbour
    from app.services.stats import StatsStore
    monkeypatch.setattr(api,'retrieve_neighbours',lambda *a:[Neighbour(0,'trn_00000','Science','A normal science article.',.1)])
    monkeypatch.setattr(dashboard,'store',StatsStore(tmp_path))
    class Client:
        def generate(self,*a,**k):
            prompt = k.get('prompt', a[0] if a else '')
            if '## Reference Examples:' in prompt:
                assert 'A normal science article.' in prompt
                return LLMResponse('{"reason":"Compared with reference [1].","anomaly_score":0.9,"citations":[1]}','test-double')
            assert 'A normal science article.' not in prompt
            return LLMResponse('{"reason":"Category-only decision.","anomaly_score":0.7}','test-double')
    monkeypatch.setattr(api,'get_llm_client',lambda:Client())
    with TestClient(app) as client:
        r=client.post('/detect/compare',json={'text':'A food recipe.','setting':'normal_anomaly'})
        assert r.status_code==200
        d=r.json()
        assert d['baseline']['verdict']=='anomaly' and d['rag']['score']==.9
        assert d['baseline']['score']==.7 and d['baseline']['citations']==[]
        assert d['baseline']['reason']=='Category-only decision.'
        assert d['rag']['reason']=='Compared with reference [1].'
        assert d['baseline']['prompt'] != d['rag']['prompt']
        assert d['retrieved'][0]['doc_id']=='trn_00000' and d['rag']['citations']==[1]
        assert client.post('/detect/compare',json={'text':'   '}).status_code==422
        assert client.post('/detect/compare',json={'text':'text','run_baseline':False,'run_rag':False}).status_code==422


def test_runner_resume_interleaves_variants_and_keeps_budget(training,monkeypatch,tmp_path):
    from scripts import run_experiment as runner
    from app.core.datasets import DatasetSpec
    from app.services.llm_client import LLMResponse
    rs,x=training
    base=tmp_path/'fixture';base.mkdir()
    for split in ['train','test']:
        records=rs if split=='train' else [{'text':f'Test only {i}','label':i%2,'original_label':'other'} for i in range(8)]
        (base/f'fixture_{split}_data.jsonl').write_text('\n'.join(json.dumps(r) for r in records),encoding='utf-8')
        np.save(base/f'fixture_{split}_data_bert_base_uncased_feature.npy',x[:len(records)])
    monkeypatch.setitem(runner.DATASETS,'fixture',DatasetSpec('fixture',('normal',),'other','fixture','fixture'))
    monkeypatch.setattr(runner.settings,'llm_provider','test-fixture')
    calls=[]
    class Client:
        def generate(self,prompt,**k):
            calls.append(prompt)
            return LLMResponse('{"reason":"Fixture response","anomaly_score":0.3,"citations":[]}','test-double')
    monkeypatch.setattr(runner,'get_llm_client',lambda:Client())
    out=tmp_path/'run'
    args=['detect','--dataset','fixture','--data-dir',str(tmp_path),'--out',str(out),'--max-train','30','--max-test','8','--ks','0','3','--allow-api','--max-calls','2']
    runner.main(args)
    assert len(calls)==2
    saved=[json.loads(p.read_text()) for p in (out/'calls').glob('*.json')]
    assert len({p['row'] for p in saved})==1
    runner.main(args)
    assert len(calls)==4 and len(list((out/'calls').glob('*.json')))==4
    summary=json.loads((out/'summary.json').read_text())
    assert len(summary['common_valid_rows'])==2
    assert summary['logical_llm_calls_this_invocation']==2


def test_data_loader_rejects_misaligned_rows(tmp_path):
    records=[{'text':'x','label':0,'original_label':'normal'} for _ in range(3)]
    (tmp_path/'fixture_train_data.jsonl').write_text('\n'.join(map(json.dumps,records)))
    np.save(tmp_path/'fixture_train_data_bert_base_uncased_feature.npy',np.ones((2,3)))
    with pytest.raises(ValueError,match='mismatch'): load_split(tmp_path,'fixture','train')

def test_memory_excludes_all_normalized_self_matches(tmp_path):
    vectors = tmp_path / 'vectors.npy'
    records = tmp_path / 'records.jsonl'
    np.save(vectors, np.array([[1,0],[1,0],[0,1]], dtype=float))
    records.write_text('\n'.join(json.dumps({'text':text,'label':0,'original_label':'science'}) for text in ['A normal text',' a NORMAL   text ', 'Different reference']))
    memory = TrainMemory(vectors, records)
    refs = memory.search(np.array([1.,0.]), 3, exclude_text='A normal text')
    assert [r.row for r in refs] == [2]


def test_reference_category_labels_can_be_withheld():
    prompt = build_detection_prompt('query',['normal'],'bad','normal_only',
        [Exemplar('Reference content','SECRET_CATEGORY',doc_id='train_1')],include_reference_categories=False)
    assert 'Reference content' in prompt and 'train_1' in prompt
    assert 'SECRET_CATEGORY' not in prompt


def test_workspace_saved_predictions_and_allowlists(tmp_path,monkeypatch):
    from app.api import workspace
    from app.main import app
    folder = tmp_path / 'test-run'
    folder.mkdir()
    (folder/'summary.json').write_text(json.dumps({'dataset':'n24_news','command':'baselines','results':[]}))
    (folder/'predictions_iforest.json').write_text(json.dumps([{'row':4,'label':1,'score':.8},{'row':9,'label':0,'score':None,'error':'failed'}]))
    monkeypatch.setattr(workspace.settings,'runs_dir',str(tmp_path))
    with TestClient(app) as client:
        assert client.get('/api/workspace/experiments').json()['experiments'][0]['id']=='test-run'
        scores=client.get('/api/workspace/experiments/test-run/predictions/iforest').json()['predictions']
        assert scores[0]=={'id':'4','label':1,'score':.8,'error':None,'pending':False}
        assert scores[1]['score'] is None
        assert client.get('/api/workspace/experiments/unknown/predictions/iforest').status_code==404
        assert client.get('/api/workspace/experiments/test-run/predictions/manifest').status_code==404


def test_workspace_retrieval_hides_labels_and_excludes_query(monkeypatch):
    from app.api import workspace
    from app.main import app
    from app.services.memory import Neighbour
    class Memory:
        def search(self, vector, k, exclude_text=None):
            assert exclude_text == 'query'
            return [Neighbour(0,'trn_00000','SECRET_CATEGORY','full reference text',.2)]
    class Embedder:
        def embed(self,text):return np.array([1.,0.])
    monkeypatch.setattr(workspace,'get_memory',lambda d:Memory())
    monkeypatch.setattr(workspace,'get_bert_embedder',lambda:Embedder())
    with TestClient(app) as client:
        result=client.post('/api/workspace/retrieval',json={'text':'query'}).json()
        assert result['query_excluded'] and not result['reference_labels_in_prompt']
        assert 'SECRET_CATEGORY' not in result['prompt']
        assert result['references'][0]['text']=='full reference text'
        assert client.post('/api/workspace/retrieval',json={'text':'   '}).status_code==422
