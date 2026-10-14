import pytest
from intentguard.validator import _serialise_call, validate
from intentguard.domain_config import load_domain_config

@pytest.fixture
def ecommerce_config():
    return load_domain_config("ecommerce")

def test_fix1_verb_merge(ecommerce_config):
    # 'view' should be mapped to 'get'
    # 'resend' should be mapped to 'send'
    v_can = ecommerce_config['_verb_canonical']
    assert v_can.get('view') == 'get'
    assert v_can.get('resend') == 'send'

def test_fix2_hyphen_split(ecommerce_config):
    # 'apply-coupon' -> apply
    call = {"method": "post", "endpoint": "/api/apply-coupon"}
    serialised = _serialise_call(call, domain_config=ecommerce_config)
    assert "apply" in serialised

def test_fix3_read_only_rule():
    # Intent verb 'get', but method 'post'
    user_msg = "Show me my recent orders."
    api_calls = [{"method": "POST", "endpoint": "/api/orders/search"}]
    res = validate(user_msg, api_calls, domain="ecommerce")
    # Because intent is 'get' and method is 'post', it cannot be approved.
    # It should be low_confidence or orphaned.
    assert res['verdicts'][0]['verdict'] != 'approved'
