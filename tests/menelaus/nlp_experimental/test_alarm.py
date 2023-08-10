import pytest
from menelaus.nlp_experimental.alarm import Alarm, KullbackLeiblerAlarm


def test_abstract_alarm_evaluate():
    """ ensure parent alarm class does not implement evaluate() """
    alarm = Alarm(threshold=0)
    with pytest.raises(NotImplementedError) as _:
        alarm.evaluate()

def test_kl_alarm_experimental():
    """ placeholder test for checking KL alarm logic when implemented """
    threshold = 2
    alarm = KullbackLeiblerAlarm(threshold)
    assert alarm._state == "baseline"
    alarm.evaluate(rep_reference=1, rep_test=2)
    assert alarm._state == "baseline"
    alarm.evaluate(rep_reference=1, rep_test=3)
    assert alarm._state == "warning"
    alarm.evaluate(rep_reference=1, rep_test=4)
    assert alarm._state == "drift"
