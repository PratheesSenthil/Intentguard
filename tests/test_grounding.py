import pytest
from intentguard.grounding import _extract_values, _normalise, _get_tokens, is_grounded, process_grounding

def test_normalise():
    assert _normalise("$29.99") == "29.99"
    assert _normalise("4,000.50") == "4000.5"
    assert _normalise("-10") == "10"
    assert _normalise("0.01") == "0.01"
    assert _normalise("SAVE20") == "save20"

def test_get_tokens():
    tokens = _get_tokens("customer 275's profile")
    assert "275" in tokens
    assert "275's" not in tokens
    
    tokens = _get_tokens("order 4821.")
    assert "4821" in tokens
    
    tokens = _get_tokens("($40)")
    assert "40" in tokens
    
    tokens = _get_tokens("SAVE20,")
    assert "save20" in tokens
    assert "20" in tokens
    
    tokens = _get_tokens("order #4821")
    assert "4821" in tokens
    
    tokens = _get_tokens("4,000 dollars")
    assert "4000" in tokens
    
def test_substring_not_grounded():
    tokens = _get_tokens("Order 482 needs cancelling")
    assert "482" in tokens
    assert not is_grounded("4822", tokens, set())
    assert not is_grounded("48210", tokens, set())

def test_extraction_and_exemptions():
    call = {
        "endpoint": "/api/orders/4821/SAVE20/info",
        "params": {
            "amount": 40,
            "is_active": True,
            "date": "2023-10-15",
            "comment": "Cancel my order 4822 please",
            "nested": {
                "email": "test@example.com",
                "code": "CODE1"
            },
            "lst": [10.5, False, "free text 99"]
        }
    }
    vals = _extract_values(call)
    
    assert "4821" in vals
    assert "SAVE20" in vals
    assert "40" in vals
    assert "test@example.com" in vals
    assert "CODE1" in vals
    assert "10.5" in vals
    
    assert "True" not in vals
    assert "False" not in vals
    assert "2023-10-15" not in vals
    assert "Cancel my order 4822 please" not in vals
    assert "free text 99" not in vals

def test_trusted_values():
    call = {"endpoint": "/api/orders/999/cancel"}
    user_msg = "cancel it"
    
    all_g, objs = process_grounding(user_msg, call, [])
    assert not all_g
    assert any(o["value"] == "999" and not o["grounded"] for o in objs)
    
    all_g, objs = process_grounding(user_msg, call, ["999"])
    assert all_g
    assert any(o["value"] == "999" and o["grounded"] for o in objs)

def test_punct():
    from intentguard.grounding import _get_tokens
    
    assert "sam.lee@mail.com" in _get_tokens("to sam.lee@mail.com.")
    assert "sam.lee@mail.com" in _get_tokens("(sam.lee@mail.com)")
    assert "sam.lee@mail.com" in _get_tokens("email sam.lee@mail.com, then")
    assert "sam.lee@mail.com" in _get_tokens("sam.lee@mail.com!")
    assert "save20" in _get_tokens("SAVE20.")
    assert "4821" in _get_tokens("order 4821).")
    assert "29.99" in _get_tokens("$29.99.")

    toks = _get_tokens("sam.lee@mail.com")
    assert "sam.lee@mail.co" not in toks
    assert "sam.lee@mail.com" in toks

def test_punct():
    from intentguard.grounding import _get_tokens
    
    assert "sam.lee@mail.com" in _get_tokens("to sam.lee@mail.com.")
    assert "sam.lee@mail.com" in _get_tokens("(sam.lee@mail.com)")
    assert "sam.lee@mail.com" in _get_tokens("email sam.lee@mail.com, then")
    assert "sam.lee@mail.com" in _get_tokens("sam.lee@mail.com!")
    assert "save20" in _get_tokens("SAVE20.")
    assert "4821" in _get_tokens("order 4821).")
    assert "29.99" in _get_tokens("$29.99.")

    toks = _get_tokens("sam.lee@mail.com")
    assert "sam.lee@mail.co" not in toks
    assert "sam.lee@mail.com" in toks
