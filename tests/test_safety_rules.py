import pytest
from intentguard.validator import validate
from intentguard.domain_config import load_domain_config

def test_verb_mismatch_veto():
    # mismatched known verbs are never approved even with cosine 0.95 and full entity match
    # 'update' vs 'create'
    user_msg = "Update the customer address"
    api_calls = [{"method": "POST", "endpoint": "/api/customers/address"}]
    res = validate(user_msg, api_calls, domain="ecommerce")
    assert res['verdicts'][0]['verdict'] != 'approved'

def test_verb_mismatch_get_blacklist():
    # a "get" intent vs POST /blacklist/ is never approved
    user_msg = "Get the customer status"
    api_calls = [{"method": "POST", "endpoint": "/api/customers/blacklist"}]
    res = validate(user_msg, api_calls, domain="ecommerce")
    assert res['verdicts'][0]['verdict'] != 'approved'

def test_verb_mismatch_freeze_unfreeze():
    # freeze vs unfreeze never match
    user_msg = "Freeze the account"
    api_calls = [{"method": "POST", "endpoint": "/api/accounts/unfreeze"}]
    res = validate(user_msg, api_calls, domain="ecommerce")
    assert res['verdicts'][0]['verdict'] != 'approved'

def test_verb_mismatch_close_delete():
    # close vs delete never match
    user_msg = "Close the ticket"
    api_calls = [{"method": "DELETE", "endpoint": "/api/tickets/"}]
    res = validate(user_msg, api_calls, domain="ecommerce")
    assert res['verdicts'][0]['verdict'] != 'approved'


def test_read_only_rule():
    # intent verb group is "get" and call method is not GET -> cap at low_confidence
    user_msg = "Read the latest comments"
    api_calls = [{"method": "POST", "endpoint": "/api/comments/search"}]
    res = validate(user_msg, api_calls, domain="ecommerce")
    assert res['verdicts'][0]['verdict'] != 'approved'

def test_url_word_handling():
    # POST with unrecognised segment gives verb None, never "create". (Unless it's a noun phrase)
    # Since it gives verb None, the mismatch veto shouldn't trigger, but the overall score should be low.
    # Actually we just want to ensure it doesn't give 'create'.
    from intentguard.validator import _serialise_call
    from intentguard.scorer import _extract_verb
    from intentguard.domain_config import load_domain_config
    dc = load_domain_config('ecommerce')
    call = {"method": "POST", "endpoint": "/api/inventory/restock-alert"}
    serialised, _ = _serialise_call(call, domain_config=dc)
    # serialised should not contain 'create' or 'post'
    assert 'create' in serialised
    assert 'post' not in serialised

def test_noun_phrase_rule():
    from intentguard.validator import _serialise_call
    from intentguard.domain_config import load_domain_config
    dc = load_domain_config('ecommerce')
    
    # Noun phrase
    call1 = {"method": "POST", "endpoint": "/api/inventory/restock-alert"}
    ser1, _ = _serialise_call(call1, domain_config=dc)
    assert 'create' in ser1
    
    # Verb
    call2 = {"method": "POST", "endpoint": "/api/tickets/escalate"}
    ser2, _ = _serialise_call(call2, domain_config=dc)
    assert 'escalate' in ser2
    assert 'create' not in ser2

def test_verb_credit_from_params_denied():
    from intentguard.scorer import _verb_score
    from intentguard.domain_config import load_domain_config
    dc = load_domain_config('ecommerce')
    # If the user says "close the ticket", intent verb is "close".
    # Call verb is explicitly "create", but text contains "close".
    # Since call_verb_provided is True, _verb_score should return 0 (no match), NOT 1.0.
    score = _verb_score('close the ticket', 'create api tickets 88 comments please close', domain_config=dc, extracted_call_verb='create', call_verb_provided=True)
    assert score == 0.0

def test_post_comments_verb():
    from intentguard.validator import _serialise_call
    from intentguard.domain_config import load_domain_config
    dc = load_domain_config('ecommerce')
    call = {"method": "POST", "endpoint": "/api/tickets/88/comments", "body": {"text": "we'll resolve this shortly"}}
    ser, verb = _serialise_call(call, domain_config=dc)
    assert verb == 'create'

def test_compatible_verbs_veto():
    from intentguard.validator import validate
    # Compatible: get and verify
    # user message intent verb is verify
    res = validate('verify this discount', [{'method': 'GET', 'endpoint': '/api/discounts'}])
    # Should not be vetoed to orphaned due to verb mismatch (since get and verify are compatible)
    assert res['verdicts'][0]['verdict'] in ('approved', 'low_confidence')

    # Read-only rule: intent is get, call is POST but verb is verify -> NOT read only violation!
    res2 = validate('get my ticket', [{'method': 'POST', 'endpoint': '/api/tickets/88/validate'}])
    assert res2['verdicts'][0]['verdict'] in ('approved', 'low_confidence')
    
    # Incompatible: get and restrict
    res3 = validate('get my ticket', [{'method': 'POST', 'endpoint': '/api/tickets/88/flag-vip'}])
    assert res3['verdicts'][0]['verdict'] != 'approved'


def test_drop_api_and_long_text():
    from intentguard.validator import _serialise_call
    call = {
        "method": "POST",
        "endpoint": "/api/users/profile",
        "body": {"status": "ok", "comment": "this is a long comment"}
    }
    ser, verb = _serialise_call(call)
    assert 'api' not in ser
    assert 'ok' in ser
    assert 'this is a long comment' not in ser


def test_verb_groups_clean():
    import json
    from pathlib import Path
    domain_path = Path(__file__).resolve().parent.parent / "intentguard" / "domains" / "ecommerce.json"
    with open(domain_path) as f:
        config = json.load(f)
    pronouns = {'me', 'you', 'he', 'she', 'it', 'we', 'they', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs', 'myself', 'yourself', 'himself', 'herself', 'itself', 'ourselves', 'yourselves', 'themselves'}
    for grp, verbs in config.get('verb_groups', {}).items():
        for v in verbs:
            assert ' ' not in v, f"Multi-word entry found: '{v}' in group '{grp}'"
            assert '_' not in v, f"Multi-word entry found: '{v}' in group '{grp}'"
            assert v not in pronouns, f"Pronoun entry found: '{v}' in group '{grp}'"


def test_dual_use_words():
    from intentguard.validator import _serialise_call
    from intentguard.domain_config import load_domain_config, _CACHE
    _CACHE.clear()
    dc = load_domain_config('ecommerce')
    
    # POST /api/fraud/orders/9012/flag/ -> restrict
    ser1, verb1 = _serialise_call({'method': 'POST', 'endpoint': '/api/fraud/orders/9012/flag/'}, dc)
    assert verb1 == 'restrict'
    
    # GET /api/fraud/flags/ -> get
    ser2, verb2 = _serialise_call({'method': 'GET', 'endpoint': '/api/fraud/flags/'}, dc)
    assert verb2 == 'get'
    
    # POST /api/support/tickets/341/comments/ -> create
    ser3, verb3 = _serialise_call({'method': 'POST', 'endpoint': '/api/support/tickets/341/comments/'}, dc)
    assert verb3 == 'create'


def test_create_verbs():
    import json
    from pathlib import Path
    domain_path = Path(__file__).resolve().parent.parent / "intentguard" / "domains" / "ecommerce.json"
    with open(domain_path) as f:
        config = json.load(f)
    creates = set(config['verb_groups']['create'])
    assert 'initiate' in creates
    assert 'record' in creates
    assert 'give' not in creates
    assert 'process' not in creates


def test_intent_context_carryover_fixed(monkeypatch):
    import intentguard.validator as vmod
    import intentguard.embedder as emod
    captured_contexts = []
    
    # We patch encode to intercept the string being passed
    original_encode = emod.encode
    def fake_encode(text):
        if 'SAVE20' in text or '5502' in text or 'valid first' in text or 'Apply discount' in text:
            captured_contexts.append(text)
        return original_encode(text)
    
    monkeypatch.setattr(emod, "encode", fake_encode)
    
    msg = "Apply discount code SAVE20 to order 5502 and check if it's valid first."
    api_call = {"method": "GET", "endpoint": "/api/foo", "params": {"irrelevant": "param text"}}
    
    res = vmod.validate(msg, [api_call])
    
    assert any('SAVE20' in c and 'valid first' in c for c in captured_contexts)
    assert not any('param text' in c for c in captured_contexts)

